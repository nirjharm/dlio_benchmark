"""Benchmark reading columns/rows from generated wide tables.

This script scans `data/` for files produced by `generate_data.py` and for each file
and each (rc, d_count, s_count) combination in `SAMPLE_CONFIG`, it reads
d_count dense features and s_count sparse features for rc rows (rc=0 means all rows),
measures elapsed time, CPU seconds used and disk IO (bytes -> sectors using SECTOR_SIZE),
stores results in `intermediate_results/results.csv` and writes plots into that directory.
"""
import os
import glob
import time
import math
import random
import pandas as pd
import numpy as np

try:
    import psutil
except Exception:
    psutil = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Config
# Each tuple: (rc, d_count, s_count)
#   rc: number of rows to query (0 = all rows)
#   d_count: number of dense features to pull
#   s_count: number of sparse features to pull
SAMPLE_CONFIG = [
    (0, 1221, 298),
    (0, 1221, 0),
    (0, 0, 298),
]

# Number of times to repeat each experiment for averaging
NUM_REPETITIONS = 3

SEED = 42
DATA_DIR = 'normal-zipf1_2_data'
#DATA_DIR = 'uniform_data' 
INTERMEDIATE_DIR = 'Results'
SECTOR_SIZE = 512  # bytes per sector
DATA_DEVICE = '/dev/nvme1n1'  # device to measure I/O from (e.g., /dev/nvme1n1, /dev/sda)

# Enable/disable file formats to test
ENABLE_UNCOMPRESSED_PARQUET = True
ENABLE_SNAPPY_PARQUET = True
ENABLE_VORTEX = True
ENABLE_LANCE = False


def ensure_dirs():
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)


def drop_caches():
    """
    Drop filesystem caches to ensure cold reads.
    Requires root/sudo on Linux. Silently skips if not available.
    """
    try:
        # Sync to flush writes
        os.system('sync')
        # Drop caches (requires root)
        with open('/proc/sys/vm/drop_caches', 'w') as f:
            f.write('3')
    except Exception:
        # Not Linux or no permission - try via sudo
        os.system('sync && sudo sh -c "echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null')


def list_data_files(data_dir=DATA_DIR):
    patterns = []
    if ENABLE_UNCOMPRESSED_PARQUET:
        patterns.append(os.path.join(data_dir, 'uncompressed', '*.parquet'))
    if ENABLE_SNAPPY_PARQUET:
        patterns.append(os.path.join(data_dir, 'snappy', '*.parquet'))
    if ENABLE_VORTEX:
        patterns.append(os.path.join(data_dir, 'vortex', '*'))
    if ENABLE_LANCE:
        patterns.append(os.path.join(data_dir, 'lance', '*.lance'))
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    return files


def get_backend_from_path(path):
    if os.path.join('data', 'uncompressed') in path:
        return 'uncompressed_parquet'
    if os.path.join('data', 'snappy') in path:
        return 'snappy_parquet'
    if os.path.join('data', 'vortex') in path:
        return 'vortex'
    if os.path.join('data', 'lance') in path:
        return 'lance'
    return 'unknown'


def read_subset_parquet(path, cols, nrows, seed=SEED):
    """
    Read specified columns from a Parquet file using PyArrow's native projection.
    Uses row slicing (not sampling) to limit rows without loading the full file.
    Returns PyArrow Table (no pandas conversion).
    """
    import pyarrow.parquet as pq

    # Use PyArrow directly for column projection at read time
    parquet_file = pq.ParquetFile(path)

    # Read only the requested columns
    table = parquet_file.read(columns=cols)

    # Limit rows if requested (slice from start, not random sampling)
    if nrows is not None and nrows < table.num_rows:
        table = table.slice(0, nrows)

    # Materialize: access nbytes to ensure data is read
    _ = table.nbytes
    return table


def read_subset_vortex(path, cols, nrows, seed=SEED):
    """
    Read specified columns from a Vortex file using native scan API with projection.
    Uses row limits at scan time to avoid loading the full file.
    Returns PyArrow Table (no pandas conversion).
    """
    import vortex as vx

    # Open file (metadata only)
    vxf = vx.open(path)

    # Build scan with column projection
    # If cols is None, read all columns
    if cols is not None:
        scanner = vxf.scan(projection=cols)
    else:
        scanner = vxf.scan()

    # Apply row limit if specified
    if nrows is not None:
        scanner = scanner.limit(nrows)

    # Execute scan and convert to Arrow
    result = scanner.read_all()
    table = result.to_arrow_table()

    # Materialize: access nbytes to ensure data is read
    _ = table.nbytes
    return table


def read_subset_lance(path, cols, nrows, seed=SEED):
    """
    Read specified columns from a Lance dataset using native scan API with projection.
    Uses row limits at scan time to avoid loading the full file.
    Returns PyArrow Table (no pandas conversion).
    """
    import lance

    # Open dataset (metadata only)
    ds = lance.dataset(path)

    # Build scanner with column projection and row limit
    scanner = ds.scanner(columns=cols, limit=nrows)

    # Execute scan to get Arrow table
    table = scanner.to_table()

    # Materialize: access nbytes to ensure data is read
    _ = table.nbytes
    return table


def get_device_stats(device=DATA_DEVICE):
    """
    Read device I/O stats from /sys/block/<device>/stat.
    Returns (read_sectors, write_sectors).
    
    The stat file format (space-separated fields):
      0: reads completed
      1: reads merged
      2: sectors read  <-- we want this
      3: time spent reading (ms)
      4: writes completed
      5: writes merged
      6: sectors written  <-- we want this
      7: time spent writing (ms)
      ...
    """
    # Extract device name from path (e.g., /dev/nvme1n1 -> nvme1n1)
    dev_name = os.path.basename(device)
    stat_path = f'/sys/block/{dev_name}/stat'
    
    with open(stat_path, 'r') as f:
        fields = f.read().split()
    
    read_sectors = int(fields[2])
    write_sectors = int(fields[6])
    return read_sectors, write_sectors


def measure_action(func, *args, **kwargs):
    proc = psutil.Process() if psutil else None
    
    # Get device-level I/O stats before
    dev_read_before, dev_write_before = get_device_stats()
    
    if proc:
        cpu_before = proc.cpu_times()
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    
    # Get device-level I/O stats after
    dev_read_after, dev_write_after = get_device_stats()
    
    if proc:
        cpu_after = proc.cpu_times()
        cpu_seconds = (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)
    else:
        cpu_seconds = 0
    
    read_sectors = dev_read_after - dev_read_before
    write_sectors = dev_write_after - dev_write_before
    return res, elapsed, cpu_seconds, read_sectors, write_sectors


def run_bench():
    ensure_dirs()
    files = list_data_files()
    results = []
    for path in files:
        backend = get_backend_from_path(path)
        print('Benchmarking', path, 'backend', backend)

        # Infer available columns from file metadata (without reading data)
        try:
            if backend == 'vortex':
                import vortex as vx
                vxf = vx.open(path)
                # Get schema from vortex file dtype
                dtype = vxf.dtype
                if hasattr(dtype, 'names'):
                    all_columns = list(dtype.names())
                else:
                    # Fallback: read 0 rows to get schema
                    schema_table = vxf.scan().limit(0).read_all().to_arrow_table()
                    all_columns = schema_table.schema.names
            elif backend == 'lance':
                import lance
                ds = lance.dataset(path)
                all_columns = ds.schema.names
            else:
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(path)
                all_columns = parquet_file.schema_arrow.names
        except Exception as e:
            print('Failed to read schema from', path, 'skipping:', e)
            continue

        # Separate dense and sparse columns by prefix
        dense_cols = [c for c in all_columns if c.startswith('dense_')]
        sparse_cols = [c for c in all_columns if c.startswith('sparse_')]
        total_dense = len(dense_cols)
        total_sparse = len(sparse_cols)
        
        # Get file size (handle directories like Lance datasets)
        if os.path.isdir(path):
            file_size = sum(
                os.path.getsize(os.path.join(dirpath, f))
                for dirpath, _, filenames in os.walk(path)
                for f in filenames
            )
        else:
            file_size = os.path.getsize(path) if os.path.exists(path) else 0

        for rc, d_count, s_count in SAMPLE_CONFIG:
            # Select columns randomly (but repeatable via seed)
            num_dense = min(d_count, total_dense)
            num_sparse = min(s_count, total_sparse)
            
            # Use seed + config to get repeatable random selection across runs
            col_rng = random.Random(SEED + hash((rc, d_count, s_count)))
            selected_dense = col_rng.sample(dense_cols, num_dense) if num_dense > 0 else []
            selected_sparse = col_rng.sample(sparse_cols, num_sparse) if num_sparse > 0 else []
            cols_to_fetch = selected_dense + selected_sparse
            
            if not cols_to_fetch:
                print(f'  Skipping config (rc={rc}, d={d_count}, s={s_count}): no columns to fetch')
                continue

            # rc=0 means all rows
            nrows = None if rc == 0 else rc

            # choose function
            if backend == 'vortex':
                read_fn = read_subset_vortex
            elif backend == 'lance':
                read_fn = read_subset_lance
            else:
                read_fn = read_subset_parquet

            # Run multiple repetitions
            for run_id in range(NUM_REPETITIONS):
                # Drop caches and wait to ensure cold read
                drop_caches()
                time.sleep(1)

                try:
                    (_, elapsed, cpu_sec, read_sectors, write_sectors) = measure_action(read_fn, path, cols_to_fetch, nrows, SEED)
                except Exception as e:
                    print(f'  Read failed for rc={rc}, d={d_count}, s={s_count}, run={run_id}:', e)
                    continue

                # Calculate CPU utilization as percentage (cpu_time / elapsed_time * 100)
                cpu_util = (cpu_sec / elapsed * 100) if elapsed > 0 else 0

                results.append({
                    'file': path,
                    'backend': backend,
                    'run_id': run_id,
                    'num_rows_requested': rc,
                    'num_dense_requested': d_count,
                    'num_sparse_requested': s_count,
                    'num_dense_fetched': num_dense,
                    'num_sparse_fetched': num_sparse,
                    'elapsed_s': elapsed,
                    'cpu_util_pct': cpu_util,
                    'read_sectors': read_sectors,
                    'write_sectors': write_sectors,
                    'file_size_bytes': file_size,
                })
                print(f'    Run {run_id + 1}/{NUM_REPETITIONS}: elapsed={elapsed:.3f}s, cpu_util={cpu_util:.1f}%, read_sectors={read_sectors}')

    df = pd.DataFrame(results)
    # Name output files after the data directory
    data_dir_name = os.path.basename(os.path.normpath(DATA_DIR))
    csv_path = os.path.join(INTERMEDIATE_DIR, f'{data_dir_name}_results.csv')
    df.to_csv(csv_path, index=False)
    print('Saved results to', csv_path)

    # plotting
    if plt is None:
        print('matplotlib not installed; skipping plots')
        return

    # Create config tuple labels for x-axis
    df['config_label'] = df.apply(
        lambda r: f"({int(r['num_rows_requested'])},{int(r['num_dense_requested'])},{int(r['num_sparse_requested'])})",
        axis=1
    )

    # Aggregate by backend and config: compute mean and std
    agg_df = df.groupby(['backend', 'config_label']).agg(
        elapsed_mean=('elapsed_s', 'mean'),
        elapsed_std=('elapsed_s', 'std'),
        cpu_util_mean=('cpu_util_pct', 'mean'),
        cpu_util_std=('cpu_util_pct', 'std'),
        sectors_mean=('read_sectors', 'mean'),
        sectors_std=('read_sectors', 'std'),
        file_size=('file_size_bytes', 'first'),  # same for all runs
    ).reset_index()

    # Fill NaN std (single run) with 0
    agg_df = agg_df.fillna(0)

    # Convert file size to MB
    agg_df['file_size_mb'] = agg_df['file_size'] / (1024 * 1024)

    # Get unique config labels in order (preserve SAMPLE_CONFIG order)
    config_labels = df['config_label'].unique().tolist()
    x_positions = np.arange(len(config_labels))
    backends = agg_df['backend'].unique()
    width = 0.25  # bar width
    offsets = np.linspace(-width, width, len(backends))

    # Compute normalized values (relative to snappy_parquet)
    # Build lookup for snappy_parquet values per config
    snappy_ref = agg_df[agg_df['backend'] == 'snappy_parquet'].set_index('config_label')

    def normalize(val, ref_val):
        return val / ref_val if ref_val > 0 else 0

    # Create figure with 2 rows x 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    (ax_elapsed, ax_cpu, ax_sectors, ax_size) = axes[0]
    (ax_elapsed_norm, ax_cpu_norm, ax_sectors_norm, ax_size_norm) = axes[1]

    # ===== ROW 1: Absolute values =====

    # Plot elapsed time with error bars
    for i, backend in enumerate(backends):
        sub = agg_df[agg_df['backend'] == backend].set_index('config_label')
        sub = sub.reindex(config_labels).fillna(0)
        ax_elapsed.bar(x_positions + offsets[i], sub['elapsed_mean'], width,
                       yerr=sub['elapsed_std'], capsize=3, label=backend)
    ax_elapsed.set_xlabel('Config (rc, d_count, s_count)')
    ax_elapsed.set_ylabel('Elapsed time (s)')
    ax_elapsed.set_title('Elapsed time (mean ± std)')
    ax_elapsed.set_xticks(x_positions)
    ax_elapsed.set_xticklabels(config_labels, rotation=45, ha='right')
    ax_elapsed.set_ylim(bottom=0)
    ax_elapsed.legend()

    # CPU utilization plot with error bars
    for i, backend in enumerate(backends):
        sub = agg_df[agg_df['backend'] == backend].set_index('config_label')
        sub = sub.reindex(config_labels).fillna(0)
        ax_cpu.bar(x_positions + offsets[i], sub['cpu_util_mean'], width,
                   yerr=sub['cpu_util_std'], capsize=3, label=backend)
    ax_cpu.set_xlabel('Config (rc, d_count, s_count)')
    ax_cpu.set_ylabel('CPU utilization (%)')
    ax_cpu.set_title('CPU utilization (mean ± std)')
    ax_cpu.set_xticks(x_positions)
    ax_cpu.set_xticklabels(config_labels, rotation=45, ha='right')
    ax_cpu.set_ylim(bottom=0)
    ax_cpu.legend()

    # Disk sectors read plot with error bars
    for i, backend in enumerate(backends):
        sub = agg_df[agg_df['backend'] == backend].set_index('config_label')
        sub = sub.reindex(config_labels).fillna(0)
        ax_sectors.bar(x_positions + offsets[i], sub['sectors_mean'], width,
                       yerr=sub['sectors_std'], capsize=3, label=backend)
    ax_sectors.set_xlabel('Config (rc, d_count, s_count)')
    ax_sectors.set_ylabel(f'Read sectors (sector={SECTOR_SIZE}B)')
    ax_sectors.set_title('Read sectors (mean ± std)')
    ax_sectors.set_xticks(x_positions)
    ax_sectors.set_xticklabels(config_labels, rotation=45, ha='right')
    ax_sectors.set_ylim(bottom=0)
    ax_sectors.legend()

    # File size plot (no error bars - same across runs)
    for i, backend in enumerate(backends):
        sub = agg_df[agg_df['backend'] == backend].set_index('config_label')
        sub = sub.reindex(config_labels).fillna(0)
        ax_size.bar(x_positions + offsets[i], sub['file_size_mb'], width, label=backend)
    ax_size.set_xlabel('Config (rc, d_count, s_count)')
    ax_size.set_ylabel('File size (MB)')
    ax_size.set_title('File size')
    ax_size.set_xticks(x_positions)
    ax_size.set_xticklabels(config_labels, rotation=45, ha='right')
    ax_size.set_ylim(bottom=0)
    ax_size.legend()

    # ===== ROW 2: Normalized to snappy_parquet =====

    # Normalized elapsed time
    for i, backend in enumerate(backends):
        sub = agg_df[agg_df['backend'] == backend].set_index('config_label')
        sub = sub.reindex(config_labels).fillna(0)
        snappy_vals = snappy_ref.reindex(config_labels)['elapsed_mean'].fillna(1)
        norm_vals = sub['elapsed_mean'] / snappy_vals
        norm_std = sub['elapsed_std'] / snappy_vals  # scale std by same factor
        ax_elapsed_norm.bar(x_positions + offsets[i], norm_vals, width,
                            yerr=norm_std, capsize=3, label=backend)
    ax_elapsed_norm.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    ax_elapsed_norm.set_xlabel('Config (rc, d_count, s_count)')
    ax_elapsed_norm.set_ylabel('Relative to snappy')
    ax_elapsed_norm.set_title('Elapsed time (normalized)')
    ax_elapsed_norm.set_xticks(x_positions)
    ax_elapsed_norm.set_xticklabels(config_labels, rotation=45, ha='right')
    ax_elapsed_norm.set_ylim(bottom=0)
    ax_elapsed_norm.legend()

    # Normalized CPU utilization
    for i, backend in enumerate(backends):
        sub = agg_df[agg_df['backend'] == backend].set_index('config_label')
        sub = sub.reindex(config_labels).fillna(0)
        snappy_vals = snappy_ref.reindex(config_labels)['cpu_util_mean'].fillna(1)
        norm_vals = sub['cpu_util_mean'] / snappy_vals
        norm_std = sub['cpu_util_std'] / snappy_vals
        ax_cpu_norm.bar(x_positions + offsets[i], norm_vals, width,
                        yerr=norm_std, capsize=3, label=backend)
    ax_cpu_norm.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    ax_cpu_norm.set_xlabel('Config (rc, d_count, s_count)')
    ax_cpu_norm.set_ylabel('Relative to snappy')
    ax_cpu_norm.set_title('CPU utilization (normalized)')
    ax_cpu_norm.set_xticks(x_positions)
    ax_cpu_norm.set_xticklabels(config_labels, rotation=45, ha='right')
    ax_cpu_norm.set_ylim(bottom=0)
    ax_cpu_norm.legend()

    # Normalized sectors
    for i, backend in enumerate(backends):
        sub = agg_df[agg_df['backend'] == backend].set_index('config_label')
        sub = sub.reindex(config_labels).fillna(0)
        snappy_vals = snappy_ref.reindex(config_labels)['sectors_mean'].fillna(1)
        norm_vals = sub['sectors_mean'] / snappy_vals
        norm_std = sub['sectors_std'] / snappy_vals
        ax_sectors_norm.bar(x_positions + offsets[i], norm_vals, width,
                            yerr=norm_std, capsize=3, label=backend)
    ax_sectors_norm.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    ax_sectors_norm.set_xlabel('Config (rc, d_count, s_count)')
    ax_sectors_norm.set_ylabel('Relative to snappy')
    ax_sectors_norm.set_title('Read sectors (normalized)')
    ax_sectors_norm.set_xticks(x_positions)
    ax_sectors_norm.set_xticklabels(config_labels, rotation=45, ha='right')
    ax_sectors_norm.set_ylim(bottom=0)
    ax_sectors_norm.legend()

    # Normalized file size
    for i, backend in enumerate(backends):
        sub = agg_df[agg_df['backend'] == backend].set_index('config_label')
        sub = sub.reindex(config_labels).fillna(0)
        snappy_vals = snappy_ref.reindex(config_labels)['file_size_mb'].fillna(1)
        norm_vals = sub['file_size_mb'] / snappy_vals
        ax_size_norm.bar(x_positions + offsets[i], norm_vals, width, label=backend)
    ax_size_norm.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    ax_size_norm.set_xlabel('Config (rc, d_count, s_count)')
    ax_size_norm.set_ylabel('Relative to snappy')
    ax_size_norm.set_title('File size (normalized)')
    ax_size_norm.set_xticks(x_positions)
    ax_size_norm.set_xticklabels(config_labels, rotation=45, ha='right')
    ax_size_norm.set_ylim(bottom=0)
    ax_size_norm.legend()

    fig.tight_layout()
    out = os.path.join(INTERMEDIATE_DIR, f'{data_dir_name}_benchmark.png')
    fig.savefig(out)
    plt.close(fig)
    print('Wrote', out)


if __name__ == '__main__':
    run_bench()
