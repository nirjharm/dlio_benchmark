
"""
   2026, Nj (Nirjhar) Mukherjee, CMU, PDL
   
   Parquet Reader with Feature Selection for DLRM workloads.
   Supports reading only a subset of dense/sparse columns.
"""

import pyarrow.parquet as pq
import numpy as np
import sys
import time

from dlio_benchmark.common.constants import MODULE_DATA_READER

# Tracing configuration - set to True for detailed traces
_TRACE_ENABLED = True
_TRACE_VERBOSE = False  # Set True for per-operation traces, False for aggregated only
_sample_counter = 0
_rg_read_counter = 0
_rg_total_read_time = 0.0
_dense_total_time = 0.0
_sparse_total_time = 0.0

def _trace(msg, verbose_only=False):
    """Print trace message. If verbose_only=True, only print when _TRACE_VERBOSE is True."""
    if _TRACE_ENABLED and (not verbose_only or _TRACE_VERBOSE):
        print(f"[PQ {time.time():.3f}] {msg}", file=sys.stderr, flush=True)

# =============================================================================
# MODULE-LEVEL CACHE - Shared across all reader instances to avoid repeated opens
# =============================================================================
_global_file_cache = {}  # filename -> ParquetFile
_global_row_group_sizes = {}  # filename -> list of row group sizes

def _get_cached_parquet_file(filename):
    """Get or open a ParquetFile, caching at module level to avoid repeated opens."""
    global _global_file_cache, _global_row_group_sizes
    
    if filename not in _global_file_cache:
        t0 = time.time()
        pf = pq.ParquetFile(filename)
        _global_file_cache[filename] = pf
        _global_row_group_sizes[filename] = [
            pf.metadata.row_group(i).num_rows
            for i in range(pf.num_row_groups)
        ]
        _trace(f"OPEN: {filename.split('/')[-1]} in {time.time()-t0:.3f}s, {pf.num_row_groups} row groups")
    
    return _global_file_cache[filename], _global_row_group_sizes[filename]

from dlio_benchmark.utils.utility import Profile, dft_ai
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.reader.feature_config import (
    initialize_feature_selection,
    get_selected_columns,
    get_all_selected_column_names,
    format_sample_for_dlio,
)

dlp = Profile(MODULE_DATA_READER)


class ParquetReader(FormatReader):
    """
    Parquet reader with lazy file open for each worker.
    Reads only the row group containing the requested sample.
    Supports column projection for dense/sparse feature selection.
    Uses module-level cache to avoid repeated file opens across instances.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        self._row_group_cache = {}  # filename -> (rg_idx, dense_data, sparse_data)
        self._current_filename = None  # track current active file
        
        # Initialize feature selection (uses config defaults or previously set values)
        initialize_feature_selection()
        self._dense_columns, self._sparse_columns = get_selected_columns()
        self._all_columns = get_all_selected_column_names()
        _trace(f"INIT: {len(self._dense_columns)} dense, {len(self._sparse_columns)} sparse cols")

    @dlp.log
    def open(self, filename):
        super().open(filename)
        pf, self.row_group_sizes = _get_cached_parquet_file(filename)
        self._current_filename = filename
        return None

    @dlp.log
    def get_sample(self, filename, sample_index):
        global _sample_counter, _rg_read_counter
        _sample_counter += 1
        
        super().get_sample(filename, sample_index)

        # Use module-level cache for ParquetFile
        pf, row_group_sizes = _get_cached_parquet_file(filename)

        # find row group containing sample
        rg_idx = 0
        idx_in_rg = sample_index
        for size in row_group_sizes:
            if idx_in_rg < size:
                break
            idx_in_rg -= size
            rg_idx += 1

        # load row group if not cached (with column projection)
        if filename not in self._row_group_cache or self._row_group_cache[filename][0] != rg_idx:
            global _rg_total_read_time, _dense_total_time, _sparse_total_time
            _rg_read_counter += 1
            t1 = time.time()
            
            # Read only selected columns from the row group
            table = pf.read_row_group(rg_idx, columns=self._all_columns)
            rg_read_time = time.time() - t1
            _rg_total_read_time += rg_read_time
            
            # Separate dense and sparse data
            t2 = time.time()
            dense_data = np.empty((table.num_rows, len(self._dense_columns)), dtype=np.float64)
            for i, col_name in enumerate(self._dense_columns):
                col_data = table[col_name]
                try:
                    dense_data[:, i] = col_data.to_numpy(zero_copy_only=False)
                except Exception:
                    dense_data[:, i] = np.asarray(col_data.to_numpy().tolist())
            dense_time = time.time() - t2
            _dense_total_time += dense_time
            
            # For sparse columns - extract full lists via to_pylist()
            # This is slow (~3s for 298 columns) but necessary for full sparse data
            t3 = time.time()
            sparse_data = [table[col_name].to_numpy().tolist() for col_name in self._sparse_columns]
            sparse_time = time.time() - t3
            _sparse_total_time += sparse_time
            
            self._row_group_cache[filename] = (rg_idx, dense_data, sparse_data)
            _trace(f"RG[{rg_idx}]: read={rg_read_time:.3f}s dense={dense_time:.3f}s sparse={sparse_time:.3f}s | TOTAL: rg={_rg_total_read_time:.2f}s dense={_dense_total_time:.2f}s sparse={_sparse_total_time:.2f}s ({_rg_read_counter} rgs)")

        rg_idx_cached, dense_data, sparse_data = self._row_group_cache[filename]
        
        # Extract the sample
        dense_values = dense_data[idx_in_rg]
        sparse_values = [sparse_col[idx_in_rg] for sparse_col in sparse_data]
        
        # Format for DLIO
        record = format_sample_for_dlio(dense_values, sparse_values)
        
        # Print progress every 5000 samples
        # if _sample_counter % 5000 == 0:
        #     _trace(f"PROGRESS: {_sample_counter} samples, {_rg_read_counter} rgs read")
        
        dft_ai.update(image_size=record.nbytes)
        return record

    @dlp.log
    def read_index(self, image_idx, step):
        # Use the global_index_map from parent class to get filename and sample_index
        filename, sample_index = self.global_index_map[image_idx]
        
        # Lazy open file if not already opened
        if filename not in self.open_file_map or self.open_file_map[filename] is None:
            self.open_file_map[filename] = self.open(filename)
            self._current_filename = filename
        
        return self.get_sample(filename, sample_index)

    @dlp.log
    def close(self, filename):
        super().close(filename)
        if filename in self._file_cache:
            del self._file_cache[filename]
        if filename in self._row_group_cache:
            del self._row_group_cache[filename]
        if self._current_filename == filename:
            self._current_filename = None

    @dlp.log
    def finalize(self):
        super().finalize()

    def next(self):
        # iterator over batches
        total_samples = self.num_samples(list(self._file_cache.keys())[0])
        filename = list(self._file_cache.keys())[0]
        for start_idx in range(0, total_samples, self.batch_size):
            batch = [
                self.get_sample(filename, i)
                for i in range(start_idx, min(start_idx + self.batch_size, total_samples))
            ]
            yield np.stack(batch)

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
