"""
   2026, Nj (Nirjhar) Mukherjee, CMU, PDL
"""

import numpy as np
import pyarrow as pa
import vortex as vx

from dlio_benchmark.common.enumerations import Compression
from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.utils.utility import progress, gen_random_tensor


class VortexGenerator(DataGenerator):
    """
    Generator for creating data in Vortex format via an Arrow Table.
    """

    def __init__(self):
        super().__init__()

    def generate(self):
        """
        Generate Vortex data for training.
        """
        super().generate()

        np.random.seed(10)
        rng = np.random.default_rng()
        dim = self.get_dimension(self.total_files_to_generate)

        for i in range(self.my_rank, int(self.total_files_to_generate), self.comm_size):
            progress(i + 1, self.total_files_to_generate, "Generating Vortex Data")

            dim_ = dim[2 * i]
            if isinstance(dim_, list):
                shape = dim_
            else:
                dim1 = dim[2 * i]
                dim2 = dim[2 * i + 1]
                shape = (dim1, dim2)

            total_size = np.prod(shape)

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

            array = np.array(records)

            # Convert to Arrow Table: one column per feature
            columns = {}
            for col_idx in range(array.shape[1]):
                columns[f"col_{col_idx}"] = array[:, col_idx]
            arrow_table = pa.table(columns)

            out_path_spec = self.storage.get_uri(self._file_list[i])
            if not out_path_spec.endswith(".vortex"):
                out_path_spec += ".vortex"

            # Write the Arrow Table *directly* to a vortex file
            vx.io.write(arrow_table, out_path_spec)

        np.random.seed()
