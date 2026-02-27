"""
   2026, Nj (Nirjhar) Mukherjee, CMU, PDL
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dlio_benchmark.common.enumerations import Compression
from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.utils.utility import progress, gen_random_tensor


"""
Generator for creating data in Parquet format.
"""
class ParquetGenerator(DataGenerator):
    def __init__(self):
        super().__init__()
        # Get row group size from config or use default
        self.rowgroupsize = 8192 # 0 = use parquet default

    def generate(self):
        """
        Generate parquet data for training.
        Mirrors CSVGenerator behavior.
        """
        super().generate()

        np.random.seed(10)
        rng = np.random.default_rng()

        dim = self.get_dimension(self.total_files_to_generate)

        for i in range(self.my_rank, int(self.total_files_to_generate), self.comm_size):

            progress(i + 1, self.total_files_to_generate, "Generating Parquet Data")

            dim_ = dim[2 * i]

            if isinstance(dim_, list):
                shape = dim_
            else:
                dim1 = dim[2 * i]
                dim2 = dim[2 * i + 1]
                shape = (dim1, dim2)

            total_size = np.prod(shape)

            import sys
            print(f"Generating data with shape {shape} and total size {total_size} for file {i}", file=sys.stderr)
            # Generate distinct records for each sample instead of duplicating
            # This avoids trivial compression while keeping generation fast
            records = []
            for _ in range(self.num_samples):
                record = gen_random_tensor(
                    shape=total_size,
                    dtype=self._args.record_element_dtype,
                    rng=rng
                )
                records.append(record)
            
            print(f"Gen Done", file=sys.stderr)

            # array = np.array(records)

            # print(f"Np created", file=sys.stderr)

            # # Convert to Arrow table
            # table = pa.Table.from_arrays(
            #     [pa.array(col) for col in array.T],
            #     names=[f"col_{j}" for j in range(array.shape[1])]
            # )
            columns = []
            for col_idx in range(total_size):
                # print(f"Processing column {col_idx+1}/{total_size}", file=sys.stderr)
                col_data = [record[col_idx] for record in records]
                columns.append(pa.array(col_data))
            table = pa.Table.from_arrays(columns, names=[f"col_{j}" for j in range(total_size)])

            print(f"arrow convert Done", file=sys.stderr)
            out_path_spec = self.storage.get_uri(self._file_list[i])

            compression = None

            print(f"Compressing", file=sys.stderr)

            if self.compression != Compression.NONE:
                if self.compression == Compression.GZIP:
                    compression = "gzip"
                elif self.compression == Compression.BZIP2:
                    compression = "brotli"   # closest supported alternative
                elif self.compression == Compression.ZIP:
                    compression = "zstd"
                elif self.compression == Compression.XZ:
                    compression = "lz4"
            else:
                compression = None

            # Parquet extension (no automatic extension like CSV)
            if not out_path_spec.endswith(".parquet"):
                out_path_spec = out_path_spec + ".parquet"

            # Build write options with row group size if specified
            write_kwargs = {
                'compression': compression
            }
            if self.rowgroupsize > 0:
                write_kwargs['row_group_size'] = self.rowgroupsize

            print(f"Writing out ", file=sys.stderr)

            pq.write_table(
                table,
                out_path_spec,
                **write_kwargs
            )

        np.random.seed()
