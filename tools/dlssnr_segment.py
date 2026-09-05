#!/usr/bin/env python3
"""
dlssnr_segment.py - find sub-tensor boundaries *inside* a weight blob.

Why this exists
---------------
The container names 153 blobs, but a blob is not a tensor. Trying to factor a
blob's byte count into `Cout x Cin x k x k (+ bias)` fails for 16 of the 24
distinct sizes, and the "solutions" it does find are meaningless divisors like
48 x 14359. The reason is that each named blob is a *bundle* of several
sub-tensors, and the container stores only the bundle's total length - the
internal partition lives in code we do not have.

The bundles are recoverable anyway, because the sub-tensors have different
dtypes. Sliding the parity-entropy discriminator along a blob shows the dtype
change partway through. That is what this tool does.

Worked example: `block23.layer2.layer` (917,568 B) resolves into a long FP8
region followed by an FP16 region, with the boundary at ~785.8 KB. 512 x 1536 =
786,432 - the FFN up-projection of a dim-512 transformer block - sits inside the
method's resolution band. Consistent, not proven.

Limits, stated plainly
----------------------
Resolution is about +/- one window (default 2 KiB). Two adjacent sub-tensors of
the SAME dtype are invisible to this method - it finds dtype changes, not tensor
changes. So a clean two-segment result is a lower bound on the part count, never
an upper one.

Usage
-----
  python dlssnr_segment.py bundle  weights_ht.bin              # group-averaged, the good one
  python dlssnr_segment.py map     weights_ht.bin              # single-blob coarse map
  python dlssnr_segment.py show    weights_ht.bin -n block23.layer2.layer
  python dlssnr_segment.py refine  weights_ht.bin -n block23.layer2.layer
"""
import argparse
import collections
import math
import struct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dlssnr_weights import walk, entropy   # noqa: E402

GAP = 0.35          # parity-entropy threshold separating FP16 from byte-granular


def gap_at(blob, base, off, win):
    b = blob[base + off:base + off + win]
    if len(b) < 64:
        return 0.0
    return entropy(collections.Counter(b[0::2])) - entropy(collections.Counter(b[1::2]))


def segments(blob, ent, win=4096):
    """Coarse dtype map: list of (kind, start, end)."""
    base, n = ent["data"], ent["size"]
    marks = []
    for off in range(0, max(0, n - win) + 1, win):
        marks.append(("f16" if gap_at(blob, base, off, win) > GAP else "f8", off))
    if not marks:
        return []
    out, kind, start = [], marks[0][0], 0
    for k, off in marks[1:]:
        if k != kind:
            out.append((kind, start, off))
            kind, start = k, off
    out.append((kind, start, n))
    return out


def refine(blob, ent, lo, hi, win=2048):
    """Binary-search one FP8 -> FP16 transition. Returns (lo, hi) bracket."""
    base = ent["data"]
    while hi - lo > 64:
        mid = (lo + hi) // 2
        if gap_at(blob, base, max(0, mid - win // 2), win) <= GAP:
            lo = mid
        else:
            hi = mid
    return lo, hi


def nice(n):
    """Round shapes a boundary near n might correspond to."""
    out = []
    for a in (64, 128, 256, 320, 384, 512, 768, 1024, 1536, 2048, 3072, 4096):
        if n % a == 0 and 8 <= n // a <= 16384:
            out.append((a, n // a))
    return out


# ---------------------------------------------------------------- bundle
# Blobs of the same byte size are the same layer type repeated across blocks,
# so their internal layout is identical. Averaging the SIGNED parity difference
# across all copies makes the noise fall as sqrt(n) while the FP16 signal stays,
# which buys a much smaller window -- and the window is the boundary resolution.
#
# Statistic: mean(even byte) - mean(odd byte) over a sliding window.
#   FP8   : both parities are the same distribution  -> 0
#   FP16LE: low byte near-uniform, high byte small    -> ~100
# Taking |.| BEFORE averaging would keep the noise floor; take it after.

def run_mean(x, w):
    import numpy as np
    c = np.concatenate(([0.0], np.cumsum(x)))
    return (c[w:] - c[:-w]) / w


def bundle_profile(arr, members, win_hw):
    """Group-averaged signed parity difference. win_hw is in halfwords."""
    import numpy as np
    acc = None
    for e in members:
        b = arr[e["data"]:e["data"] + e["size"]]
        n = (len(b) // 2) * 2
        v = run_mean(b[0:n:2], win_hw) - run_mean(b[1:n:2], win_hw)
        acc = v if acc is None else acc + v
    return np.abs(acc / len(members))


def cmd_bundle(blob, ents):
    import math
    import numpy as np
    arr = np.frombuffer(blob, dtype=np.uint8).astype(np.float64)
    groups = collections.defaultdict(list)
    for e in ents:
        groups[e["size"]].append(e)
    print("Group-averaged bundle segmentation.")
    print("Window adapts to copy count: more copies -> finer boundary resolution.\n")
    print(f"{'blob size':>11} {'n':>4} {'window':>8} {'SE':>6}  segments (bytes)")
    for sz, mem in sorted(groups.items(), key=lambda x: -x[0]):
        if sz < 4096:
            continue
        n = len(mem)
        win_hw = max(64, int(2 ** math.ceil(math.log2(4096 / max(1, n)))))
        p = bundle_profile(arr, mem, win_hw)
        se = 60.0 * math.sqrt(2.0 / (win_hw * n))
        thr = max(8.0, 6 * se)
        k = (p > thr).astype(int)
        segs, cur, st = [], k[0], 0
        for i, v in enumerate(k[1:], 1):
            if v != cur:
                segs.append(("f16" if cur else "f8", st * 2, (i + win_hw // 2) * 2))
                cur, st = v, i + win_hw // 2
        segs.append(("f16" if cur else "f8", st * 2, sz))
        segs = [x for x in segs if x[2] - x[1] >= win_hw]
        lay = " | ".join(f"{kk}:{b - a:,}" for kk, a, b in segs)
        note = ""
        if n < 6:
            note = "   (few copies -> window too coarse to place a boundary)"
        elif len(segs) > 3:
            note = "   (conv bundle; boundaries not yet on round values)"
        print(f"{sz:>11,} {n:>4} {win_hw * 2:>8} {se:>6.2f}  {lay[:88]}{note}")
    print("""
Reading the result: a segment's BYTE COUNT is what the data determines. Splitting
it into a x b is not - 1024x4096 and 512x8192 have the same byte count. Use the
tier dimension (512 for the dim-512 transformer blocks, 1024 for the bottleneck)
to fix the factorisation, then the byte count confirms it.""")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["map", "show", "refine", "bundle"])
    ap.add_argument("blob")
    ap.add_argument("-n", "--name", help="blob name for show/refine")
    ap.add_argument("-w", "--window", type=int, default=4096)
    a = ap.parse_args()

    with open(a.blob, "rb") as fh:
        blob = fh.read()
    ents, _ = walk(blob)
    by = {e["name"]: e for e in ents}

    if a.cmd == "bundle":
        cmd_bundle(blob, ents)

    elif a.cmd == "map":
        multi = 0
        print(f"{'blob':30}{'bytes':>12}{'segments':>10}  layout")
        for e in sorted(ents, key=lambda x: -x["size"]):
            if e["size"] < a.window * 2:
                continue
            segs = segments(blob, e, a.window)
            if len(segs) > 1:
                multi += 1
            layout = " ".join(f"{k}:{(b - s) // 1024}K" for k, s, b in segs[:6])
            print(f"{e['name'][:30]:30}{e['size']:>12,}{len(segs):>10}  {layout}")
        print(f"\n{multi} blobs are internally multi-segment -> a named blob is a "
              f"BUNDLE, not a tensor.")
        print("Same-dtype neighbours are invisible here, so these counts are LOWER BOUNDS.")

    elif a.cmd in ("show", "refine"):
        if not a.name or a.name not in by:
            raise SystemExit(f"give -n with one of: {', '.join(list(by)[:4])} ...")
        e = by[a.name]
        segs = segments(blob, e, a.window)
        print(f"{a.name}  {e['size']:,} B  ->  {len(segs)} coarse segment(s)")
        for k, s, b in segs:
            print(f"    {k:4} {s:>10,} .. {b:>10,}   ({b - s:>10,} B)")
        if a.cmd == "refine":
            for i in range(len(segs) - 1):
                k0, _, b0 = segs[i]
                lo, hi = refine(blob, e, max(0, b0 - 2 * a.window), b0 + 2 * a.window)
                mid = (lo + hi) // 2
                print(f"\n  transition {segs[i][0]} -> {segs[i + 1][0]}"
                      f" bracketed at [{lo:,} .. {hi:,}]")
                for cand in sorted({786432, 524288, 262144, 1048576,
                                    (mid // 1024) * 1024, ((mid // 1024) + 1) * 1024}):
                    if abs(cand - mid) <= 4 * a.window:
                        shapes = nice(cand)
                        stxt = ", ".join(f"{x}x{y}" for x, y in shapes[:4]) or "no round shape"
                        print(f"    candidate split {cand:>10,} B  (+/-{abs(cand - mid):>6,})"
                              f"  -> {stxt}")
                print("    resolution is about +/- one window; a candidate inside the")
                print("    bracket is CONSISTENT with the data, not proven by it.")


if __name__ == "__main__":
    main()
