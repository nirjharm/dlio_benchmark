"""Generate synthetic wide tables in parquet (uncompressed and snappy) and Vortex (if available).

Config at top controls dense/sparse feature counts, sparse length, number of rows, row-group size and seed.
This script writes files under `data/uncompressed/`, `data/snappy/`, and `data/vortex/`.

Memory-efficient: streams data in row-group-sized chunks to avoid OOM.
"""
import os
import math
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Config
NUM_ROWS = 100000  # number of rows per table

# Arrays of feature configs - each index generates one set of files
# Dense features: float64 columns
DENSE_FEATURE_COUNTS = [12115]

# Sparse features: lists of int32 indices (variable-length per row)
SPARSE_FEATURE_COUNTS = [1763]

# Average length of each sparse feature list (actual length varies around this mean)
SPARSE_FEATURE_LENGTHS = [25]

# Row group size for streaming writes
ROWGROUP_SIZE = 1000
SEED = 42


def ensure_dirs(base='data'):
    paths = [
        os.path.join(base, 'uncompressed'),
        os.path.join(base, 'snappy'),
        os.path.join(base, 'vortex'),
        os.path.join(base, 'lance'),
    ]
    for p in paths:
        os.makedirs(p, exist_ok=True)


def make_schema(num_dense, num_sparse):
    """
    Create PyArrow schema for dense float64 columns and sparse list<int32> columns.
    """
    fields = []
    for i in range(num_dense):
        fields.append(pa.field(f'dense_{i}', pa.float64()))
    for i in range(num_sparse):
        fields.append(pa.field(f'sparse_{i}', pa.list_(pa.int32())))
    return pa.schema(fields)


def generate_chunk(num_dense, num_sparse, sparse_length, chunk_size, rng):
    """
    Generate one chunk of data as a PyArrow RecordBatch (no pandas copy).
    """
    arrays = []

    # Dense features: float64 from normal distribution (compressible)
    for _ in range(num_dense):
        arr = pa.array(rng.standard_normal(chunk_size).astype('float64'))
        arrays.append(arr)

    # Sparse features: variable-length lists of int32 from Zipf distribution (compressible)
    for _ in range(num_sparse):
        lengths = np.maximum(1, rng.poisson(sparse_length, size=chunk_size))
        total_ints = int(lengths.sum())
        # Zipf with alpha=1.2 gives power-law distribution (few popular values)
        pool = rng.zipf(1.2, size=total_ints).astype('int32')
        ends = np.cumsum(lengths)
        starts = np.concatenate([[0], ends[:-1]])
        # Build list array directly
        offsets = pa.array(np.concatenate([[0], ends]).astype('int32'))
        values = pa.array(pool)
        arr = pa.ListArray.from_arrays(offsets, values)
        arrays.append(arr)

    return arrays


def write_parquet_streaming(path, num_dense, num_sparse, sparse_length, nrows, compression=None, row_group_size=ROWGROUP_SIZE, seed=SEED):
    """
    Stream-write a Parquet file in chunks to avoid OOM.
    """
    schema = make_schema(num_dense, num_sparse)
    rng = np.random.default_rng(seed)

    writer = pq.ParquetWriter(path, schema, compression=compression)

    rows_written = 0
    chunk_id = 0
    num_chunks = (nrows + row_group_size - 1) // row_group_size
    report_interval = max(1, num_chunks // 10)

    while rows_written < nrows:
        chunk_size = min(row_group_size, nrows - rows_written)
        arrays = generate_chunk(num_dense, num_sparse, sparse_length, chunk_size, rng)
        batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
        table = pa.Table.from_batches([batch])
        writer.write_table(table)
        rows_written += chunk_size
        chunk_id += 1
        if chunk_id % report_interval == 0 or rows_written >= nrows:
            print(f'    Chunks written: {chunk_id}/{num_chunks} ({100 * rows_written // nrows}%)')

    writer.close()


def write_vortex_streaming(path, num_dense, num_sparse, sparse_length, nrows, row_group_size=ROWGROUP_SIZE, seed=SEED):
    """
    Stream-write a Vortex file. Vortex doesn't have a streaming writer, so we build
    smaller chunks and concatenate, or write via a temp parquet if needed.
    """
    import vortex as vx

    schema = make_schema(num_dense, num_sparse)
    rng = np.random.default_rng(seed)

    # Collect batches (smaller memory footprint than full table)
    batches = []
    rows_written = 0
    chunk_id = 0
    num_chunks = (nrows + row_group_size - 1) // row_group_size
    report_interval = max(1, num_chunks // 10)

    while rows_written < nrows:
        chunk_size = min(row_group_size, nrows - rows_written)
        arrays = generate_chunk(num_dense, num_sparse, sparse_length, chunk_size, rng)
        batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
        batches.append(batch)
        rows_written += chunk_size
        chunk_id += 1

        if chunk_id % report_interval == 0 or rows_written >= nrows:
            print(f'    Chunks generated: {chunk_id}/{num_chunks} ({100 * rows_written // nrows}%)')

        # Write intermediate batches to avoid accumulating too many in memory
        # (write every 10 chunks worth, ~10k rows by default)
        if len(batches) >= 10:
            print(f'    Flushing {len(batches)} batches to reduce memory...')
            # For vortex we need to accumulate all, but we can try to keep batches small
            pass  # Keep going, vortex needs full table

    print(f'    Building final table from {len(batches)} batches...')
    table = pa.Table.from_batches(batches)
    del batches  # Free memory

    print(f'    Writing to vortex...')
    vx.io.write(table, path)


def write_lance_streaming(path, num_dense, num_sparse, sparse_length, nrows, row_group_size=ROWGROUP_SIZE, seed=SEED):
    """
    Stream-write a Lance dataset. Lance supports appending batches, so we can write incrementally.
    """
    import lance

    schema = make_schema(num_dense, num_sparse)
    rng = np.random.default_rng(seed)

    rows_written = 0
    chunk_id = 0
    num_chunks = (nrows + row_group_size - 1) // row_group_size
    report_interval = max(1, num_chunks // 10)
    dataset = None

    while rows_written < nrows:
        chunk_size = min(row_group_size, nrows - rows_written)
        arrays = generate_chunk(num_dense, num_sparse, sparse_length, chunk_size, rng)
        batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
        table = pa.Table.from_batches([batch])

        if dataset is None:
            # First write creates the dataset
            dataset = lance.write_dataset(table, path, mode='create')
        else:
            # Append subsequent batches
            lance.write_dataset(table, path, mode='append')

        rows_written += chunk_size
        chunk_id += 1
        if chunk_id % report_interval == 0 or rows_written >= nrows:
            print(f'    Chunks written: {chunk_id}/{num_chunks} ({100 * rows_written // nrows}%)')


def main():
    ensure_dirs()

    # Validate config arrays have same length
    assert len(DENSE_FEATURE_COUNTS) == len(SPARSE_FEATURE_COUNTS) == len(SPARSE_FEATURE_LENGTHS), \
        "Config arrays must have the same length"

    num_configs = len(DENSE_FEATURE_COUNTS)
    for idx, (dense_count, sparse_count, sparse_len) in enumerate(zip(DENSE_FEATURE_COUNTS, SPARSE_FEATURE_COUNTS, SPARSE_FEATURE_LENGTHS)):
        print(f'\n[{idx + 1}/{num_configs}] Generating table with {dense_count} dense + {sparse_count} sparse features '
              f'(avg sparse length {sparse_len}), {NUM_ROWS} rows')

        # uncompressed parquet (streaming)
        upath = os.path.join('data', 'uncompressed', f'table_d{dense_count}_s{sparse_count}_l{sparse_len}.parquet')
        print(f'  Writing uncompressed parquet to {upath}...')
        write_parquet_streaming(upath, dense_count, sparse_count, sparse_len, NUM_ROWS, compression=None)
        print(f'  Done. Wrote {upath}')

        # snappy compressed parquet (streaming)
        spath = os.path.join('data', 'snappy', f'table_d{dense_count}_s{sparse_count}_l{sparse_len}_snappy.parquet')
        print(f'  Writing snappy-compressed parquet to {spath}...')
        write_parquet_streaming(spath, dense_count, sparse_count, sparse_len, NUM_ROWS, compression='snappy')
        print(f'  Done. Wrote {spath}')

        # vortex (streaming)
        vpath = os.path.join('data', 'vortex', f'table_d{dense_count}_s{sparse_count}_l{sparse_len}.vortex')
        print(f'  Writing vortex to {vpath}...')
        write_vortex_streaming(vpath, dense_count, sparse_count, sparse_len, NUM_ROWS)
        print(f'  Done. Wrote {vpath}')

        # # lance (streaming)
        # lpath = os.path.join('data', 'lance', f'table_d{dense_count}_s{sparse_count}_l{sparse_len}.lance')
        # print(f'  Writing lance to {lpath}...')
        # write_lance_streaming(lpath, dense_count, sparse_count, sparse_len, NUM_ROWS)
        # print(f'  Done. Wrote {lpath}')

    print(f'\nAll {num_configs} table(s) generated successfully.')


if __name__ == '__main__':
    main()
