#!/usr/bin/env python3
"""
dlssnr_bench.py - turn the mod's own log into defensible numbers.

The shim already prints everything a benchmark needs; nobody was reading it.
This parses `<exe>_dlssnr_on_amd.log` and reports median/p99 job time per
configuration, with the confounded intervals thrown out rather than averaged in.

What it reads
-------------
  network job %d done in %llu ms (history %s, %s)   <- the primary metric
  FSR active   job %d: %llu ms   %s   %s
  frames %d dispatches %d submitted %d ready %d route %s
  staging ready: colour %ux%u ... motion ... depth ...
  env: using HIP device %d (%s): %s, arch %s, %d CUs, %.0f MB
  ini changed Inline=%d Interop=%d; re-creating staging   <- config boundary

Why the rejections matter
-------------------------
`job` values of -1 mean the FFX hook never fired: the mod is idle and any
"fast" number is meaningless. Frame-replacement fallbacks, in-flight skips and
GPU wait timeouts all change what is being measured, so samples in those
windows are dropped instead of being averaged into the result.

Job time is logged in whole milliseconds, so quantisation is ~2.6% on a 39 ms
job. That is fine for a resolution sweep and useless for anything sub-ms; the
tool refuses to report a shift smaller than one quantum.

Usage
-----
  python dlssnr_bench.py game_dlssnr_on_amd.log
  python dlssnr_bench.py a.log b.log --compare
"""
import argparse
import collections
import re
import statistics
import sys

RE_JOB = re.compile(r"network job (-?\d+) done in (\d+) ms(?:.*?\(history (\S+?),\s*(.*?)\))?")
RE_FSR = re.compile(r"FSR active\s+job (-?\d+):\s+(\d+) ms")
RE_ROUTE = re.compile(r"frames (\d+) dispatches (\d+) submitted (\d+) ready (\d+) route (.+)")
RE_INI = re.compile(r"ini changed Inline=(\d+) Interop=(\d+)")
RE_DEV = re.compile(r"env: using HIP device (\d+) \(([^)]*)\): ([^,]+), arch (\S+), (\d+) CUs")
RE_STAGING = re.compile(r"staging ready: colour (\d+)x(\d+)")
RE_WEIGHTS = re.compile(r"wrote (\d+) blobs")

BAD = [
    ("too many frames in flight", "frame skipped"),
    ("GPU wait timeouts", "inline wait timeout"),
    ("residual textures failed", "fell back to frame replacement"),
    ("output has no UAV flag", "residual apply unavailable"),
    ("FSR is not active", "FSR was off"),
    ("UNSUPPORTED GPU ARCHITECTURE", "unsupported GPU"),
]


def parse(path):
    samples = []          # (line_no, ms)
    warnings = collections.Counter()
    meta = {}
    routes = collections.Counter()
    first_run = False
    bad_windows = []      # line numbers where something invalidating happened
    neg_jobs = 0

    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            for needle, label in BAD:
                if needle in line:
                    warnings[label] += 1
                    bad_windows.append(n)
            if RE_WEIGHTS.search(line):
                first_run = True
            m = RE_DEV.search(line)
            if m:
                meta["hip_device"] = f"{m.group(3).strip()} (arch {m.group(4)}, {m.group(5)} CUs)"
            m = RE_STAGING.search(line)
            if m:
                meta["colour"] = f"{m.group(1)}x{m.group(2)}"
            m = RE_INI.search(line)
            if m:
                meta.setdefault("ini_changes", []).append((n, f"Inline={m.group(1)} Interop={m.group(2)}"))
            m = RE_ROUTE.search(line)
            if m:
                routes[m.group(5).strip()] += 1
            m = RE_JOB.search(line) or RE_FSR.search(line)
            if m:
                job = int(m.group(1))
                if job < 0:
                    neg_jobs += 1
                    continue
                samples.append((n, int(m.group(2))))
    return samples, warnings, meta, routes, first_run, bad_windows, neg_jobs


def clean(samples, bad_windows, guard=120):
    """Drop samples within `guard` lines of anything invalidating."""
    if not bad_windows:
        return samples, 0
    bad = sorted(bad_windows)
    keep, dropped = [], 0
    for n, ms in samples:
        near = any(abs(n - b) <= guard for b in bad)
        if near:
            dropped += 1
        else:
            keep.append((n, ms))
    return keep, dropped


def describe(name, vals):
    if not vals:
        print(f"  {name}: no usable samples")
        return None
    vals = sorted(vals)
    med = statistics.median(vals)
    p99 = vals[min(len(vals) - 1, int(0.99 * len(vals)))]
    print(f"  {name}: n={len(vals)}  median={med:.0f} ms  p99={p99} ms  "
          f"min={vals[0]} max={vals[-1]}")
    hist = collections.Counter(vals)
    span = sorted(hist)
    if len(span) <= 14:
        print("      buckets: " + "  ".join(f"{k}ms:{hist[k]}" for k in span))
    return med


def report(path):
    samples, warnings, meta, routes, first_run, bad, neg = parse(path)
    print(f"=== {path} ===")
    for k in ("hip_device", "colour"):
        if k in meta:
            print(f"  {k}: {meta[k]}")
    if routes:
        print(f"  routes seen: {dict(routes)}")
    if meta.get("ini_changes"):
        print(f"  ini reloads: {len(meta['ini_changes'])} "
              f"-> {[c[1] for c in meta['ini_changes']]}")

    if neg:
        print(f"\n  !! {neg} sample(s) had job = -1: the FFX hook never fired.")
        print("     The mod was idle. Nothing here measures the network.")
    if first_run:
        print("\n  !! this log contains 'wrote N blobs' - it is a FIRST RUN.")
        print("     It paid for the weight-cache build; discard it entirely.")
    if warnings:
        print("\n  invalidating events:")
        for label, count in warnings.most_common():
            print(f"    {label}: {count}")

    kept, dropped = clean(samples, bad)
    print(f"\n  samples: {len(samples)} parsed, {dropped} dropped near an "
          f"invalidating event, {len(kept)} usable")
    med = describe("job time", [v for _, v in kept])
    if med is not None:
        print(f"\n  quantisation: 1 ms on a {med:.0f} ms median = "
              f"{100.0 / med:.1f}% - ignore any shift smaller than that.")
    return [v for _, v in kept]


def mannwhitney_u(x, y):
    """Two-sided Mann-Whitney U with a normal approximation. No SciPy needed."""
    import math
    merged = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    ranks = [0.0] * len(merged)
    i = 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, merged) if g == 0)
    n1, n2 = len(x), len(y)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma == 0:
        return 1.0
    z = (u1 - mu) / sigma
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--compare", action="store_true",
                    help="compare the first two logs (Mann-Whitney + median shift)")
    a = ap.parse_args()

    series = []
    for p in a.logs:
        series.append(report(p))
        print()

    if a.compare:
        if len(series) < 2 or not series[0] or not series[1]:
            sys.exit("need two logs with usable samples to compare")
        x, y = series[0], series[1]
        mx, my = statistics.median(x), statistics.median(y)
        pv = mannwhitney_u(x, y)
        print("=== comparison ===")
        print(f"  A median {mx:.0f} ms (n={len(x)})   B median {my:.0f} ms (n={len(y)})")
        print(f"  shift: {my - mx:+.0f} ms ({100 * (my - mx) / mx:+.1f}%)")
        print(f"  Mann-Whitney two-sided p = {pv:.4g}")
        if abs(my - mx) < 1.0:
            print("  -> shift is below the 1 ms logging quantum. Not a result.")
        elif pv >= 0.05:
            print("  -> not significant at p<0.05.")
        else:
            print("  -> significant.")


if __name__ == "__main__":
    main()
