#!/usr/bin/env python3
"""Generate graphs from hyperparameter sweep outputs.

For each `nproc` value found under `hyperparameter_tuning_results/`, this script
creates a 4x4 grid of plots. Columns correspond to step-counts (first column = first
step count found, second = second, etc.). Each column contains (top->bottom):
- end-to-end time vs batch size
- accelerator utilization vs batch size
- nj_disktrace: x=elapsed_ms, y=delta_sectors (one line per batch size)
- nj_disktrace: x=elapsed_ms, y=mem_used_kb (one line per batch size)

Outputs are saved to `hyperparameter_tuning_results/graphs/nproc_<N>.png`.
"""

import argparse
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


RUN_DIR = "hyperparameter_tuning_results"


def parse_run_meta(name):
    # Expect names like: run_fP_b16_s2000_p1_i1
    m = re.match(r"run_f(?P<filetype>[PV])_b(?P<batch>\d+)_s(?P<step>\d+)_p(?P<nproc>\d+)_i(?P<idx>\d+)", name)
    if not m:
        return None
    return {k: int(v) if k in ("batch", "step", "nproc", "idx") else v for k, v in m.groupdict().items()}


def parse_run_log(path):
    # Try to extract end-to-end time and accelerator utilization (AU) from run.log
    end2end = None
    au = None
    if not os.path.isfile(path):
        return end2end, au
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            # several heuristics
            for pat in [r"Total time[:=]\s*([0-9]+\.?[0-9]*)", r"End[- ]to[- ]end time[:=]\s*([0-9]+\.?[0-9]*)", r"Elapsed time[:=]\s*([0-9]+\.?[0-9]*)s?", r"Time elapsed[:=]\s*([0-9]+\.?[0-9]*)"]:
                m = re.search(pat, line, re.IGNORECASE)
                if m and end2end is None:
                    try:
                        end2end = float(m.group(1))
                    except ValueError:
                        pass
            for pat in [r"Accelerator utilization[:=]\s*([0-9]+\.?[0-9]*)", r"AU[:=]\s*([0-9]+\.?[0-9]*)", r"accel.*util[:=]\s*([0-9]+\.?[0-9]*)"]:
                m = re.search(pat, line, re.IGNORECASE)
                if m and au is None:
                    try:
                        au = float(m.group(1))
                    except ValueError:
                        pass
            # Parse METRIC block lines (common format in run.log)
            if line.startswith("[METRIC]"):
                # Accelerator Utilization
                m = re.search(r"Training Accelerator Utilization.*:\s*([0-9]+\.?[0-9]*)", line)
                if m and au is None:
                    try:
                        au = float(m.group(1))
                    except ValueError:
                        pass
                # Throughput (samples/sec)
                m = re.search(r"Training Throughput.*:\s*([0-9]+\.?[0-9]*)", line)
                if m:
                    try:
                        throughput = float(m.group(1))
                    except ValueError:
                        throughput = None
                # I/O throughput (MB/sec)
                m = re.search(r"Training I/O Throughput.*:\s*([0-9]+\.?[0-9]*)", line)
                if m:
                    try:
                        io_throughput = float(m.group(1))
                    except ValueError:
                        io_throughput = None
                # continue parsing other lines
                continue
    # return additional metrics where available
    # throughput and io_throughput may be set in METRIC parsing above
    try:
        throughput
    except NameError:
        throughput = None
    try:
        io_throughput
    except NameError:
        io_throughput = None
    return end2end, au, throughput, io_throughput


def load_nj_disktrace(path):
    # nj_disktrace format: elapsed_ms,delta_sectors,mem_used_kb
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path, header=None)
        # support both 2- and 3-column variants
        if df.shape[1] == 2:
            df.columns = ["elapsed_ms", "delta_sectors"]
            df["mem_used_kb"] = np.nan
        else:
            df.columns = ["elapsed_ms", "delta_sectors", "mem_used_kb"]
        return df
    except Exception:
        return None


def gather_runs(base_dir):
    runs = []
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(base_dir)
    for name in os.listdir(base_dir):
        meta = parse_run_meta(name)
        if not meta:
            continue
        outdir = os.path.join(base_dir, name)
        runlog = os.path.join(outdir, "run.log")
        end2end, au, throughput, io_throughput = parse_run_log(runlog)
        nj = load_nj_disktrace(os.path.join(outdir, "nj_disktrace"))
        meta.update({"outdir": outdir, "end2end": end2end, "au": au, "throughput": throughput, "io_throughput": io_throughput, "nj": nj})
        runs.append(meta)
    return runs


def plot_for_nproc(runs, nproc, out_dir):
    # Filter runs
    filtered = [r for r in runs if r["nproc"] == nproc]
    if not filtered:
        return
    # collect unique steps and batches
    steps = sorted({r["step"] for r in filtered})
    batches = sorted({r["batch"] for r in filtered})

    # create output dir
    os.makedirs(out_dir, exist_ok=True)

    # classify runs into Full/Half/Quarter based on formula:
    def compute_full_minus1(batch, nproc):
        try:
            full = int(100000 // (batch * nproc))
            return max(1, full - 1)
        except Exception:
            return None

    def classify_kind(run):
        fm1 = compute_full_minus1(run['batch'], run['nproc'])
        if not fm1:
            return 'other'
        ratio = float(run['step']) / float(fm1)
        if ratio >= 0.75:
            return 'full'
        if ratio >= 0.35:
            return 'half'
        if ratio >= 0.125:
            return 'quarter'
        return 'other'

    kinds = ['full', 'half', 'quarter', 'other']
    cols = 3
    rows = 4
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

    for col_idx in range(cols):
        kind = kinds[col_idx] if col_idx < len(kinds) else 'other'
        axes[0][col_idx].set_title(f"{kind}  (nproc={nproc})")

        # collect metrics per batch and filetype for this kind
        metrics = {ft: {} for ft in set(r["filetype"] for r in filtered)}
        for ft in metrics.keys():
            for b in batches:
                match = next((rr for rr in filtered if classify_kind(rr) == kind and rr["batch"] == b and rr["filetype"] == ft), None)
                if match:
                    metrics[ft][b] = {
                        "end2end": match.get("end2end"),
                        "au": match.get("au"),
                        "throughput": match.get("throughput"),
                        "nj": match.get("nj"),
                        "step_meta": match.get("step")
                    }
                else:
                    metrics[ft][b] = {"end2end": None, "au": None, "throughput": None, "nj": None, "step_meta": None}

        # Row 0: end-to-end vs batch size (skip missing datapoints)
        ax0 = axes[0][col_idx]
        linestyles = {"V": "-", "P": ":"}
        for ft in sorted(metrics.keys()):
            x_vals = [b for b in batches if metrics[ft][b]["end2end"] is not None or metrics[ft][b]["throughput"] is not None]
            y_vals = []
            for b in x_vals:
                e = metrics[ft][b]["end2end"]
                if e is None and metrics[ft][b]["throughput"]:
                    # approximate end-to-end: determine effective steps from Full-1 definition
                    fm1 = compute_full_minus1(b, nproc)
                    if fm1:
                        # choose effective steps based on kind
                        if kind == 'full':
                            eff_steps = fm1
                        elif kind == 'half':
                            eff_steps = max(1, fm1 // 2)
                        elif kind == 'quarter':
                            eff_steps = max(1, fm1 // 4)
                        else:
                            # fallback to meta step if available
                            eff_steps = metrics[ft][b].get('step_meta') or fm1
                        try:
                            e = (eff_steps * b) / float(metrics[ft][b]["throughput"]) if metrics[ft][b]["throughput"] and metrics[ft][b]["throughput"] > 0 else None
                        except Exception:
                            e = None
                y_vals.append(e)
            if x_vals:
                ax0.plot(x_vals, y_vals, marker='o', linestyle=linestyles.get(ft, '-'), label=f'{ft}')
        ax0.set_xlabel('batch size')
        ax0.set_ylabel('end-to-end (s)')
        ax0.grid(True)
        ax0.legend(fontsize='small')
        ax0.set_ylim(bottom=0)

        # Row 1: AU vs batch
        ax1 = axes[1][col_idx]
        for ft in sorted(metrics.keys()):
            x_vals_au = [b for b in batches if metrics[ft][b]["au"] is not None]
            y_vals_au = [metrics[ft][b]["au"] for b in x_vals_au]
            if x_vals_au:
                ax1.plot(x_vals_au, y_vals_au, marker='o', linestyle=linestyles.get(ft, '-'), label=f'{ft}')
        ax1.set_xlabel('batch size')
        ax1.set_ylabel('accelerator utilization')
        ax1.grid(True)
        ax1.legend(fontsize='small')
        ax1.set_ylim(bottom=0)

        # Row 2: nj_disktrace col2 vs elapsed (one line per batch)
        ax2 = axes[2][col_idx]
        # track whether we added a real line for legend
        # for NJ plots, plot one line per (batch,filetype)
        for ft in sorted(metrics.keys()):
            for b in batches:
                df = metrics[ft][b]["nj"]
                label = f'b{b}_{ft}'
                if df is None or df.empty:
                    ax2.plot([], [], linestyle='--', color='gray', label=f'{label} (failed)')
                    continue
                ax2.plot(df['elapsed_ms'], df['delta_sectors'], label=label, linestyle=linestyles.get(ft, '-'))
        ax2.set_xlabel('elapsed ms')
        ax2.set_ylabel('delta sectors')
        ax2.legend(fontsize='small')
        ax2.grid(True)
        ax2.set_ylim(bottom=0)

        # Row 3: nj_disktrace col3 (mem) vs elapsed
        ax3 = axes[3][col_idx]
        for ft in sorted(metrics.keys()):
            for b in batches:
                df = metrics[ft][b]["nj"]
                label = f'b{b}_{ft}'
                if df is None or df.empty or 'mem_used_kb' not in df.columns:
                    ax3.plot([], [], linestyle='--', color='gray', label=f'{label} (failed)')
                    continue
                ax3.plot(df['elapsed_ms'], df['mem_used_kb'], label=label, linestyle=linestyles.get(ft, '-'))
        ax3.set_xlabel('elapsed ms')
        ax3.set_ylabel('mem used (KB)')
        ax3.legend(fontsize='small')
        ax3.grid(True)
        ax3.set_ylim(bottom=0)

    fig.tight_layout()
    outpath = os.path.join(out_dir, f'nproc_{nproc}.png')
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default=RUN_DIR, help='base hyperparameter results dir')
    p.add_argument('--out', default=None, help='output graphs dir (defaults to <base>/graphs)')
    args = p.parse_args()

    runs = gather_runs(args.base)
    if not runs:
        print('No runs found in', args.base)
        return
    nprocs = sorted({r['nproc'] for r in runs})
    out_dir = args.out or os.path.join(args.base, 'graphs')
    os.makedirs(out_dir, exist_ok=True)
    for n in nprocs:
        print('Plotting nproc=', n)
        plot_for_nproc(runs, n, out_dir)


if __name__ == '__main__':
    main()
