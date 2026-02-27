
"""
   2026, Nj (Nirjhar) Mukherjee, CMU, PDL
   
   Vortex Reader with Feature Selection for DLRM workloads.
   Reads Arrow-based Vortex files with lazy batch streaming.
   Supports reading only a subset of dense/sparse columns via projection.
   Uses RepeatedScan with row_range for efficient random access.
"""

import vortex as vx
import numpy as np
import pyarrow as pa
import sys
import time

from dlio_benchmark.common.constants import MODULE_DATA_READER

# Tracing configuration
_TRACE_ENABLED = False
_batch_read_counter = 0

def _trace(msg):
    if _TRACE_ENABLED:
        print(f"[VORTEX_READER {time.time():.3f}] {msg}", file=sys.stderr, flush=True)

# =============================================================================
# MODULE-LEVEL CACHE - Shared across all reader instances to avoid repeated opens
# =============================================================================
_global_file_cache = {}  # filename -> VortexFile
_global_file_lengths = {}  # filename -> total_rows
_global_repeated_scan_cache = {}  # filename -> RepeatedScan (with projection)

def _get_cached_vortex_file(filename, projection=None):
    """
    Get or open a VortexFile and create a RepeatedScan with projection.
    Caching at module level to avoid repeated opens.
    
    Args:
        filename: Path to the vortex file
        projection: List of column names to project (for efficient reads)
    
    Returns:
        Tuple of (VortexFile, file_length, RepeatedScan)
    """
    global _global_file_cache, _global_file_lengths, _global_repeated_scan_cache
    
    if filename not in _global_file_cache:
        _trace(f"  CACHE MISS: Opening VortexFile for {filename}")
        t0 = time.time()
        vxf = vx.open(filename)
        _global_file_cache[filename] = vxf
        _global_file_lengths[filename] = len(vxf)
        _trace(f"  VortexFile opened in {time.time()-t0:.3f}s, {len(vxf)} rows")
        
        # Create a RepeatedScan with column projection for efficient repeated access
        if projection:
            _trace(f"  Creating RepeatedScan with {len(projection)} columns projection")
            t1 = time.time()
            repeated_scan = vxf.to_repeated_scan(projection=projection)
            _global_repeated_scan_cache[filename] = repeated_scan
            _trace(f"  RepeatedScan created in {time.time()-t1:.3f}s")
        else:
            _global_repeated_scan_cache[filename] = vxf.to_repeated_scan()
    else:
        _trace(f"  CACHE HIT: Reusing VortexFile for {filename}")
    
    return (_global_file_cache[filename], 
            _global_file_lengths[filename], 
            _global_repeated_scan_cache.get(filename))

from dlio_benchmark.utils.utility import Profile, dft_ai
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.reader.feature_config import (
    initialize_feature_selection,
    get_selected_columns,
    get_all_selected_column_names,
    get_selected_indices,
    format_sample_for_dlio,
)

dlp = Profile(MODULE_DATA_READER)


class VortexReader(FormatReader):
    """
    Vortex Reader: reads Arrow-based Vortex files with lazy batch streaming.
    Uses RepeatedScan.execute(row_range=...) for efficient random access.
    Supports column projection for dense/sparse feature selection.
    Uses module-level cache to avoid repeated file opens across instances.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        _trace(f"__init__ START: dataset_type={dataset_type}, thread_index={thread_index}")
        t0 = time.time()
        super().__init__(dataset_type, thread_index)
        self._batch_cache = {}  # filename -> (batch_idx, dense_data, sparse_data)
        
        # Initialize feature selection (uses config defaults or previously set values)
        _trace("  Initializing feature selection...")
        initialize_feature_selection()
        self._dense_columns, self._sparse_columns = get_selected_columns()
        self._dense_indices, self._sparse_indices = get_selected_indices()
        self.c_projection = self._dense_columns + self._sparse_columns
        self._all_columns = get_all_selected_column_names()
        _trace(f"__init__ END: {len(self._dense_columns)} dense, {len(self._sparse_columns)} sparse cols, took {time.time()-t0:.3f}s")

    @dlp.log
    def open(self, filename):
        """
        Opens the Vortex file lazily using module-level cache.
        Creates a RepeatedScan with column projection for efficient batch reads.
        """
        _trace(f"open START: {filename}")
        t0 = time.time()
        super().open(filename)
        # Use module-level cache with projection
        vxf, file_len, repeated_scan = _get_cached_vortex_file(filename, projection=self.c_projection)
        _trace(f"open END: {file_len} rows, took {time.time()-t0:.3f}s")
        return None

    @dlp.log
    def close(self, filename):
        """
        Clears instance-level cached batches. Module-level cache persists.
        """
        super().close(filename)
        if filename in self._batch_cache:
            del self._batch_cache[filename]

    def _split_by_density(self, arrow_array):
        """
        Extract dense and sparse columns from Arrow struct array.
        The arrow_array is already projected to only contain selected columns.
        
        Args:
            arrow_array: PyArrow StructArray with projected columns
            
        Returns:
            Tuple of (dense_data, sparse_data) where:
            - dense_data: numpy array of shape (num_rows, num_dense_features)
            - sparse_data: list of lists for sparse features
        """
        num_rows = len(arrow_array)

        # Helper: try zero-copy first, fall back to safe copy if necessary
        def _col_to_numpy(col):
            try:
                return col.to_numpy(zero_copy_only=True)
            except pa.ArrowInvalid:
                return col.to_numpy(zero_copy_only=False)

        # Directly extract dense columns (assume projected columns are correct)
        if self._dense_columns:
            dense_arrays = [_col_to_numpy(arrow_array.field(col))
                            for col in self._dense_columns]
            dense_data = np.column_stack(dense_arrays)
        else:
            dense_data = np.empty((num_rows, 0))

        # Extract sparse columns, preferring zero-copy
        sparse_data = [_col_to_numpy(arrow_array.field(col))
                       for col in self._sparse_columns]

        return dense_data, sparse_data

    @dlp.log
    def get_sample(self, filename, sample_index):
        """
        Retrieves a single sample row from the file using lazy batch streaming.
        Uses RepeatedScan.execute(row_range=...) for efficient random access.
        Only reads the batch containing the requested sample.
        """
        global _batch_read_counter
        super().get_sample(filename, sample_index)

        VORTEX_BATCH_SIZE = 8192
        _trace(f"batch_size={VORTEX_BATCH_SIZE}")

        # Use module-level cache for VortexFile and RepeatedScan
        
        vxf, file_length, repeated_scan = _get_cached_vortex_file(filename, projection=self.c_projection)

        # Determine which batch contains this sample
        batch_idx = sample_index // VORTEX_BATCH_SIZE
        idx_in_batch = sample_index % VORTEX_BATCH_SIZE

        # Load batch if not cached - use lazy row_range access
        if filename not in self._batch_cache or self._batch_cache[filename][0] != batch_idx:
            batch_start = batch_idx * VORTEX_BATCH_SIZE
            batch_end = min(batch_start + VORTEX_BATCH_SIZE, file_length)
            
            _batch_read_counter += 1
            _trace(f"get_sample: LAZY READ batch {batch_idx} rows [{batch_start}, {batch_end}) for sample {sample_index}")
            t0 = time.time()
            
            # Use RepeatedScan.execute with row_range for efficient random access
            batch_result = repeated_scan.execute(row_range=(batch_start, batch_end)).read_all()
            batch_arrow = batch_result.to_arrow_array()
            _trace(f"  execute(row_range) took {time.time()-t0:.3f}s, got {len(batch_arrow)} rows")
            
            # Extract columns from the projected batch
            t1 = time.time()
            batch_dense, batch_sparse = self._split_by_density(batch_arrow)
            _trace(f"  column extraction took {time.time()-t1:.3f}s")
            
            self._batch_cache[filename] = (batch_idx, batch_dense, batch_sparse)

        _, dense_data, sparse_data = self._batch_cache[filename]
        
        # Extract the sample
        dense_values = dense_data[idx_in_batch]
        sparse_values = [sparse_col[idx_in_batch] for sparse_col in sparse_data]
        
        # Format for DLIO
        record = format_sample_for_dlio(dense_values, sparse_values)
        dft_ai.update(image_size=record.nbytes)
        return record

    @dlp.log
    def read_index(self, index, step):
        """
        Reads sample by global index. DLIO expects this interface.
        Uses the global_index_map to get filename and sample_index.
        """
        filename, sample_index = self.global_index_map[index]
        
        # Lazy open file if not already opened
        if filename not in self.open_file_map or self.open_file_map[filename] is None:
            self.open_file_map[filename] = self.open(filename)
        
        return self.get_sample(filename, sample_index)

    @dlp.log
    def next(self):
        """
        Returns an iterator over the dataset. Required by DLIO.
        """
        for batch in super().next():
            yield batch

    @dlp.log
    def finalize(self):
        """
        Finalize reader state.
        """
        return super().finalize()

    def is_index_based(self):
        # This reader now operates in a non-index-based mode: it returns the whole file.
        return True

    def is_iterator_based(self):
        return True
