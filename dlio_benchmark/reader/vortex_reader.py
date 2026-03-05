"""
2026, Nj (Nirjhar) Mukherjee, CMU, PDL

Vortex Reader with Feature Selection for DLRM workloads.
Reads Arrow-based Vortex files with lazy batch streaming.
Supports reading only a subset of dense/sparse columns via projection.
Uses the Vortex scan API to pull batches directly instead of per-index
repeated scans.
""" 

import vortex as vx
import numpy as np
import pyarrow as pa
import sys
import time

from dlio_benchmark.common.constants import MODULE_DATA_READER

# Tracing configuration
_TRACE_ENABLED = False
_fetch_counter = 0
_fetch_total_time = 0.0
_recent_cache = {"filename": None, "records": {}}
_cache_hits = 0
_cache_requests = 0
_missing_warned = set()
_file_lengths = {}
_cache_size = 1000

def _trace(msg):
    if _TRACE_ENABLED:
        print(f"[VORTEX_READER {time.time():.3f}] {msg}", file=sys.stderr, flush=True)

# =============================================================================
# MODULE-LEVEL CACHE - Shared across all reader instances to avoid repeated opens
# =============================================================================
_global_file_cache = {}  # filename -> VortexFile

def _get_cached_vortex_file(filename):
    """Get or open a VortexFile. Cached at module level to avoid reopens."""
    global _global_file_cache, _file_lengths

    if filename not in _global_file_cache:
        _trace(f"  CACHE MISS: Opening VortexFile for {filename}")
        t0 = time.time()
        vxf = vx.open(filename)
        _global_file_cache[filename] = vxf
        _file_lengths[filename] = 100000
    else:
        _trace(f"  CACHE HIT: Reusing VortexFile for {filename}")

    return _global_file_cache[filename]

from dlio_benchmark.utils.utility import Profile, dft_ai
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.reader.feature_config import (
    initialize_feature_selection,
    get_selected_columns,
    get_all_selected_column_names,
    get_selected_indices,
    extract_sparse_first_elements_batch,
    format_sample_for_dlio,
)

dlp = Profile(MODULE_DATA_READER)


class VortexReader(FormatReader):
    """
    Vortex Reader: reads Arrow-based Vortex files with lazy batch streaming.
    Uses the Vortex scan API with projection + indices for batch-aligned access.
    Supports column projection for dense/sparse feature selection.
    Uses module-level cache to avoid repeated file opens across instances.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        _trace(f"__init__ START: dataset_type={dataset_type}, thread_index={thread_index}")
        t0 = time.time()
        super().__init__(dataset_type, thread_index)
        
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
        """
        _trace(f"open START: {filename}")
        t0 = time.time()
        super().open(filename)
        _get_cached_vortex_file(filename)
        _trace(f"open END: took {time.time()-t0:.3f}s")
        return None

    @dlp.log
    def close(self, filename):
        """
        Clears instance-level cached batches. Module-level cache persists.
        """
        super().close(filename)

    def _scan_batch_into_cache(self, filename, start_index):
        """Scan up to 32 contiguous rows starting at start_index and cache them."""
        global _recent_cache, _fetch_counter, _fetch_total_time, _cache_size

        vxf = _get_cached_vortex_file(filename)

        max_len = _file_lengths.get(filename)
        if max_len is not None:
            if max_len <= 0:
                return
            start_index = min(start_index, max_len - 1)
            end_index = min(start_index + _cache_size, max_len)
        else:
            end_index = start_index + _cache_size

        indices = list(range(start_index, end_index))
        if not indices:
            return

        t_scan_start = time.time()
        scan_iter = vxf.scan(
            projection=self.c_projection if self.c_projection else None,
            indices=vx.array(indices)
        )

        t_read_start = time.time()
        arrow_array = scan_iter.read_all().to_arrow_array()
        t_read_end = time.time()

        _fetch_counter += 1
        _fetch_total_time += (t_read_end - t_scan_start)
        if _fetch_counter % 10 == 0:
            avg_ms = (_fetch_total_time / _fetch_counter) * 1000.0 if _fetch_counter else 0.0
            print(
                f"[VORTEX_READER] avg_scan_total_ms={avg_ms:.3f} after {_fetch_counter} scans",
                file=sys.stderr,
                flush=True,
            )

        dense_data, sparse_data = self._split_by_density(arrow_array)
        sparse_first = extract_sparse_first_elements_batch(sparse_data)

        records = {}
        for pos in range(len(dense_data)):
            record = format_sample_for_dlio(dense_data[pos], sparse_first[pos])
            dft_ai.update(image_size=record.nbytes)
            records[indices[pos]] = record

        _recent_cache = {"filename": filename, "records": records}

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
        # Normalize to a StructArray for uniform access
        if isinstance(arrow_array, pa.ChunkedArray):
            arrow_array = arrow_array.combine_chunks()
        if isinstance(arrow_array, pa.Table):
            # Convert table to struct array
            arrow_array = arrow_array.to_struct_array()

        num_rows = len(arrow_array)

        # Helper: try zero-copy first, fall back to safe copy if necessary
        def _col_to_numpy(col):
            try:
                return col.to_numpy(zero_copy_only=True)
            except pa.ArrowInvalid:
                return col.to_numpy(zero_copy_only=False)

        # Safely extract dense columns; fill zeros when missing
        if self._dense_columns:
            dense_arrays = []
            for col in self._dense_columns:
                idx = arrow_array.type.get_field_index(col)
                if idx == -1:
                    if col not in _missing_warned:
                        print(f"[VORTEX_READER] missing dense column '{col}' in scan result; filling zeros", file=sys.stderr, flush=True)
                        _missing_warned.add(col)
                    dense_arrays.append(np.zeros(num_rows, dtype=np.float64))
                else:
                    dense_arrays.append(_col_to_numpy(arrow_array.field(col)))
            dense_data = np.column_stack(dense_arrays)
        else:
            dense_data = np.empty((num_rows, 0))

        # Extract sparse columns, preferring zero-copy; fill zeros when missing
        sparse_data = []
        for col in self._sparse_columns:
            idx = arrow_array.type.get_field_index(col)
            if idx == -1:
                if col not in _missing_warned:
                    print(f"[VORTEX_READER] missing sparse column '{col}' in scan result; filling zeros", file=sys.stderr, flush=True)
                    _missing_warned.add(col)
                sparse_data.append(np.zeros(num_rows, dtype=np.int64))
            else:
                sparse_data.append(_col_to_numpy(arrow_array.field(col)))

        return dense_data, sparse_data

    @dlp.log
    def get_sample(self, filename, sample_index):
        """
        Retrieves a single sample row via scan(indices=...).
        """
        super().get_sample(filename, sample_index)

        global _cache_hits, _cache_requests

        is_hit = _recent_cache.get("filename") == filename and sample_index in _recent_cache.get("records", {})
        _cache_requests += 1
        if is_hit:
            _cache_hits += 1
            if _cache_requests % 8192 == 0:
                hit_rate = (_cache_hits / _cache_requests) * 100.0 if _cache_requests else 0.0
                print(f"[VORTEX_READER] cache_hit_rate={hit_rate:.2f}% ({_cache_hits}/{_cache_requests})", file=sys.stderr, flush=True)
            return _recent_cache["records"][sample_index]

        # Align to 32-sized block starting at sample_index
        self._scan_batch_into_cache(filename, sample_index)
        if _cache_requests % 10 == 0:
            hit_rate = (_cache_hits / _cache_requests) * 100.0 if _cache_requests else 0.0
            print(f"[VORTEX_READER] cache_hit_rate={hit_rate:.2f}% ({_cache_hits}/{_cache_requests})", file=sys.stderr, flush=True)
        return _recent_cache["records"].get(sample_index)

    @dlp.log
    def read_index(self, index, step):
        """
        Reads sample by global index. DLIO expects this interface for index-based loaders.
        """
        filename, sample_index = self.global_index_map[index]
        
        # Lazy open file if not already opened
        if filename not in self.open_file_map or self.open_file_map[filename] is None:
            self.open_file_map[filename] = self.open(filename)
        
        return self.get_sample(filename, sample_index)

    @dlp.log
    def next(self):
        """
        Iterator using the base class batching logic (index-based).
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
        return True

    def is_iterator_based(self):
        return True
