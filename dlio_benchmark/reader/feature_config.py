"""
   2026, Nj (Nirjhar) Mukherjee, CMU, PDL
   
   Feature Selection Configuration for DLRM workloads.
   Configures which dense and sparse columns to read from widetable data.
"""

import numpy as np
import sys
import time
from typing import List, Tuple, Optional

# Tracing configuration
_TRACE_ENABLED = True
_TRACE_VERBOSE = False  # Set True for detailed traces

def _trace(msg, verbose_only=False):
    if _TRACE_ENABLED and (not verbose_only or _TRACE_VERBOSE):
        print(f"[FC {time.time():.3f}] {msg}", file=sys.stderr, flush=True)

# =============================================================================
# FEATURE SELECTION CONFIGURATION - MODIFY THESE VALUES
# =============================================================================

# Number of dense features to randomly select (set to None to use all)
NUM_DENSE_FEATURES_TO_SELECT: Optional[int] = 1221

# Number of sparse features to randomly select (set to None to use all)
NUM_SPARSE_FEATURES_TO_SELECT: Optional[int] = 298

# Random seed for reproducible column selection (set to None for random each run)
RANDOM_SEED: Optional[int] = 42

# =============================================================================
# DATASET SCHEMA CONFIGURATION
# =============================================================================

# Total number of dense columns in the dataset
TOTAL_DENSE_COLUMNS = 12115

# Total number of sparse columns in the dataset
TOTAL_SPARSE_COLUMNS = 1763

# Column name prefixes
DENSE_COLUMN_PREFIX = "dense_"
SPARSE_COLUMN_PREFIX = "sparse_"

# =============================================================================
# INTERNAL STATE - DO NOT MODIFY
# =============================================================================

_selected_dense_indices: Optional[List[int]] = None
_selected_sparse_indices: Optional[List[int]] = None
_dense_column_names: Optional[List[str]] = None
_sparse_column_names: Optional[List[str]] = None
_initialized: bool = False


def initialize_feature_selection(
    n_dense: Optional[int] = None,
    m_sparse: Optional[int] = None,
    seed: Optional[int] = None,
    force_reinit: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Initialize random feature selection for dense and sparse columns.
    
    Args:
        n_dense: Number of dense features to select. If None, uses NUM_DENSE_FEATURES_TO_SELECT.
                 If that is also None, selects all dense columns.
        m_sparse: Number of sparse features to select. If None, uses NUM_SPARSE_FEATURES_TO_SELECT.
                  If that is also None, selects all sparse columns.
        seed: Random seed for reproducibility. If None, uses RANDOM_SEED.
        force_reinit: If True, reinitializes even if already initialized.
    
    Returns:
        Tuple of (dense_column_names, sparse_column_names)
    """
    global _selected_dense_indices, _selected_sparse_indices
    global _dense_column_names, _sparse_column_names, _initialized
    
    if _initialized and not force_reinit:
        return _dense_column_names, _sparse_column_names
    
    # Use provided values or fall back to module-level config
    n_dense = n_dense if n_dense is not None else NUM_DENSE_FEATURES_TO_SELECT
    m_sparse = m_sparse if m_sparse is not None else NUM_SPARSE_FEATURES_TO_SELECT
    seed = seed if seed is not None else RANDOM_SEED
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Select dense column indices
    if n_dense is None or n_dense >= TOTAL_DENSE_COLUMNS:
        _selected_dense_indices = list(range(TOTAL_DENSE_COLUMNS))
    else:
        _selected_dense_indices = sorted(
            np.random.choice(TOTAL_DENSE_COLUMNS, size=n_dense, replace=False).tolist()
        )
    
    # Select sparse column indices
    if m_sparse is None or m_sparse >= TOTAL_SPARSE_COLUMNS:
        _selected_sparse_indices = list(range(TOTAL_SPARSE_COLUMNS))
    else:
        _selected_sparse_indices = sorted(
            np.random.choice(TOTAL_SPARSE_COLUMNS, size=m_sparse, replace=False).tolist()
        )
    
    # Generate column names
    _dense_column_names = [f"{DENSE_COLUMN_PREFIX}{i}" for i in _selected_dense_indices]
    _sparse_column_names = [f"{SPARSE_COLUMN_PREFIX}{i}" for i in _selected_sparse_indices]
    
    _initialized = True
    _trace(f"INIT: {len(_dense_column_names)} dense, {len(_sparse_column_names)} sparse cols (seed={seed})")
    return _dense_column_names, _sparse_column_names


def get_selected_columns() -> Tuple[List[str], List[str]]:
    """
    Get the currently selected column names.
    Initializes with default config if not already initialized.
    
    Returns:
        Tuple of (dense_column_names, sparse_column_names)
    """
    if not _initialized:
        initialize_feature_selection()
    return _dense_column_names, _sparse_column_names


def get_all_selected_column_names() -> List[str]:
    """
    Get all selected column names (dense + sparse) in order.
    
    Returns:
        List of all selected column names
    """
    dense_cols, sparse_cols = get_selected_columns()
    return dense_cols + sparse_cols


def get_selected_indices() -> Tuple[List[int], List[int]]:
    """
    Get the selected column indices.
    
    Returns:
        Tuple of (dense_indices, sparse_indices)
    """
    if not _initialized:
        initialize_feature_selection()
    return _selected_dense_indices, _selected_sparse_indices


def get_num_selected_features() -> Tuple[int, int]:
    """
    Get the number of selected dense and sparse features.
    
    Returns:
        Tuple of (num_dense, num_sparse)
    """
    if not _initialized:
        initialize_feature_selection()
    return len(_selected_dense_indices), len(_selected_sparse_indices)


def reset_feature_selection():
    """
    Reset the feature selection state. Next call to get_selected_columns
    or initialize_feature_selection will reinitialize.
    """
    global _selected_dense_indices, _selected_sparse_indices
    global _dense_column_names, _sparse_column_names, _initialized
    
    _selected_dense_indices = None
    _selected_sparse_indices = None
    _dense_column_names = None
    _sparse_column_names = None
    _initialized = False



_format_call_counter = 0
_format_total_time = 0.0  # Total time spent in format_sample_for_dlio


def _first_element_or_zero(value):
    """Best-effort extraction of the first element from a sparse value."""
    if value is None:
        return 0
    # Common fast paths
    if isinstance(value, np.ndarray):
        return value[0] if value.size else 0
    if isinstance(value, (list, tuple)):
        return value[0] if value else 0

    # PyArrow scalars/arrays often expose as_py/tolist
    if hasattr(value, "as_py"):
        py_val = value.as_py()
        if isinstance(py_val, list):
            return py_val[0] if py_val else 0
        return py_val if py_val is not None else 0
    if hasattr(value, "tolist"):
        lst = value.tolist()
        if isinstance(lst, list):
            return lst[0] if lst else 0
        return lst if lst is not None else 0

    # Fallback for scalars/unknowns
    return value if np.isscalar(value) else 0


def extract_sparse_first_elements_batch(sparse_columns) -> np.ndarray:
    """
    Convert a list of sparse columns (each column is a sequence per row) into
    a dense 2D numpy array of the first element per row/column. Intended to be
    called once per batch/row group to amortize Python overhead.
    """
    if not sparse_columns:
        return np.empty((0, 0), dtype=np.float64)

    num_rows = len(sparse_columns[0])
    num_cols = len(sparse_columns)
    first_elems = np.empty((num_rows, num_cols), dtype=np.float64)

    for col_idx, col in enumerate(sparse_columns):
        # Fast path: numeric numpy array (no per-row Python work)
        if isinstance(col, np.ndarray) and col.dtype != object:
            first_elems[:, col_idx] = np.asarray(col, dtype=np.float64, copy=False)
            continue

        # General path: object arrays or Python lists of variable-length lists
        col_iter = col if isinstance(col, list) else np.asarray(col, dtype=object)
        first_elems[:, col_idx] = [
            _first_element_or_zero(val) for val in col_iter
        ]

    return first_elems

def format_sample_for_dlio(dense_values: np.ndarray, sparse_values) -> np.ndarray:
    """
    Format a sample's dense and sparse features into DLIO's expected format.
    
    DLIO expects a numpy array as the sample. For DLRM-style data, we concatenate:
    1. Dense features as a flat float array
    2. Sparse features - first element of each sparse column (for embedding lookup)
    
    Args:
        dense_values: Array of dense feature values (floats)
        sparse_values: Either:
            - numpy array of first elements (int64) - fast path from parquet_reader
            - List of sparse feature arrays (legacy path)
    
    Returns:
        numpy array suitable for DLIO
    """
    global _format_call_counter, _format_total_time
    _format_call_counter += 1
    t0 = time.time()
    
    # Convert dense to float64 without copying if already correct
    dense_arr = np.asarray(dense_values, dtype=np.float64, copy=False)
    
    # Handle sparse values - check if already numpy array (fast path)
    if isinstance(sparse_values, np.ndarray):
        sparse_arr = np.asarray(sparse_values, dtype=np.float64, copy=False)
    else:
        # Legacy path: list/tuple of sparse lists; extract first element of each
        sparse_arr = np.fromiter(
            (_first_element_or_zero(sv) for sv in sparse_values),
            dtype=np.float64,
            count=len(sparse_values),
        )

    # Preallocate output once instead of concatenate allocating twice
    combined = np.empty(dense_arr.size + sparse_arr.size, dtype=np.float64)
    combined[: dense_arr.size] = dense_arr
    combined[dense_arr.size :] = sparse_arr
    
    elapsed = time.time() - t0
    _format_total_time += elapsed
    
    # Log every 5000 calls with avg timing
    if _format_call_counter % 5000 == 0:
        avg_ms = (_format_total_time / _format_call_counter) * 1000
        _trace(f"FORMAT: {_format_call_counter} calls, avg={avg_ms:.3f}ms/call, total={_format_total_time:.1f}s")
    
    return combined
