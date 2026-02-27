#!/usr/bin/env python3
"""Cleanup hyperparameter tuning results by removing runs that crashed.

Scans subdirectories of `hyperparameter_tuning_results` (or a provided base
directory). For each run folder, if `run.log` contains an mpirun termination
message with `Exit code: 255`, the folder is reported and removed when run
with `--yes`. By default the script performs a dry-run and prints candidates.
"""

import argparse
import os
import re
import shutil
from pathlib import Path


def run_log_indicates_mpi_exit_255(path: Path) -> bool:
    if not path.is_file():
        return False
    # Read the last ~16KB of the file to avoid loading huge logs
    try:
        with open(path, 'rb') as f:
            try:
                f.seek(-16384, os.SEEK_END)
            except OSError:
                f.seek(0)
            tail = f.read().decode('utf-8', errors='ignore')
    except Exception:
        return False
    # Look for the mpirun termination snippet indicating exit code 255
    # e.g. a block that contains 'Exit code:' followed by '255'
    if re.search(r"Exit code:\s*255", tail):
        return True
    # Some variants include the textual header; check for typical mpirun lines
    if 'mpirun detected that one or more processes exited with non-zero status' in tail:
        if re.search(r"Exit code:\s*255", tail):
            return True
    return False


def find_failed_runs(base_dir: Path):
    candidates = []
    if not base_dir.is_dir():
        return candidates
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        runlog = entry / 'run.log'
        if run_log_indicates_mpi_exit_255(runlog):
            candidates.append(entry)
    return candidates


def main():
    p = argparse.ArgumentParser(description='Remove tuning runs whose run.log ends with mpirun Exit code 255')
    p.add_argument('--base', default='hyperparameter_tuning_results', help='base results directory')
    p.add_argument('--yes', action='store_true', help='actually delete matching run folders; otherwise dry-run')
    args = p.parse_args()

    base = Path(args.base)
    candidates = find_failed_runs(base)
    if not candidates:
        print('No failed runs (Exit code 255) found in', base)
        return

    print(f'Found {len(candidates)} failed run(s):')
    for c in candidates:
        print('  ', c)

    if not args.yes:
        print('\nDry-run mode: run with --yes to delete these directories')
        return

    # Delete the candidates
    for c in candidates:
        try:
            shutil.rmtree(c)
            print('Deleted', c)
        except Exception as e:
            print('Failed to delete', c, ':', e)


if __name__ == '__main__':
    main()
