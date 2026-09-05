#!/usr/bin/env python3
"""
dlssnr_model.py - a generative model of the whole weight blob, and its falsification test.

The claim
---------
Every one of the 153 blobs is the same block schema instantiated at one of six
widths C in {32, 64, 128, 256, 512, 1024}. For a plain block:

    R(C) = 9*C^2 + 388*C + 48 + max(16, C//8)          for C >= 64
    R(32) = 8*C^2 + 388*C + 48 + 16                     (the A2 slab is absent)

Downsample blocks append one 2*C^2 FP8 tensor; upsample blocks prepend one and
carry a 2*C FP16 vector instead of the padded gain group.

This is falsifiable and cheap to falsify: run `check` and every predicted size
must equal an observed size, and the totals must agree to the byte. They do -
153/153, 147,683,778 of 147,683,778 bytes, zero residual.

Do not mistake this for a forward pass. Byte extents and dtypes are settled;
the FACTORISATION of each slab mostly is not. 262,144 B is equally 1024x256,
512x512 or 4x(256x256), and nothing in the container picks one. See
docs/FINDINGS.md for the list of what remains unknown.

The named parts of a block, per width C
---------------------------------------
    A0 = 4*C^2   FP8     row length 128 at C=32 (ISA-proven: idx >> 7)
    A1 = 128*C   FP8     role unknown; absent at C >= 512
    A2 = C^2     FP8     absent at C=32
    g  = C       FP16    gain vector
    Q  = 3*C^2   FP8     row length 3C at C=32 (ISA-proven: magic divide by 96)
    T  = 128*C   FP16    all values <= 0, max exactly 0.0; role unknown
    S  = C//32   FP32    consumer unidentified
    P  = C^2     FP8
plus 16-byte zero padding after each FP16 group.

Usage
-----
  python dlssnr_model.py check weights_ht.bin
  python dlssnr_model.py parts -C 256
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dlssnr_weights import walk   # noqa: E402


def pad16(n):
    """zero padding that rounds an FP16/FP32 group up to a 16-byte slot"""
    return (-n) % 16 if n % 16 else 16


def plain_parts(C):
    """(label, bytes, dtype) for a plain block of width C."""
    p = [("A0", 4 * C * C, "f8"), ("A1", 128 * C, "f8")]
    if C != 32:
        p.append(("A2", C * C, "f8"))
    p += [("pad", 16, "z"), ("g1", 2 * C, "f16"), ("pad", 16, "z"),
          ("Q", 3 * C * C, "f8"), ("T", 2 * 128 * C, "f16"),
          ("S", 4 * (C // 32), "f32")]
    if C < 128:
        p.append(("pad", 16 - 4 * (C // 32), "z"))
    p += [("P", C * C, "f8"), ("g2", 2 * C, "f16"), ("pad", 16, "z")]
    return p


def plain(C):
    return sum(b for _, b, _ in plain_parts(C))


def closed_form(C):
    if C == 32:
        return 8 * C * C + 388 * C + 48 + 16
    return 9 * C * C + 388 * C + 48 + max(16, C // 8)


TIERS = {
    # width: (plain copies, down blob, up blob)
    32:  (6, 22720, 22784),
    64:  (6, 69936, 70048),
    128: (10, 229936, 230176),
    256: (14, 820288, 820784),
}
# transformer / bottleneck, measured directly
WIDE = {
    524288: 16, 263168: 32, 917568: 16,          # dim 512 blocks
    524304: 1, 525312: 1,                        # transitions
    4194320: 8, 4196352: 8, 3145856: 8, 1050624: 8, 2: 9,   # dim 1024 bottleneck
    21696: 1, 21808: 1,                          # stem, head
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "parts"])
    ap.add_argument("blob", nargs="?")
    ap.add_argument("-C", type=int, default=256)
    a = ap.parse_args()

    if a.cmd == "parts":
        C = a.C
        print(f"plain block, width C = {C}")
        tot = 0
        for lbl, b, dt in plain_parts(C):
            tot += b
            print(f"  {lbl:5} {dt:4} {b:>10,}")
        print(f"  {'total':5} {'':4} {tot:>10,}   closed form {closed_form(C):,}"
              f"   {'MATCH' if tot == closed_form(C) else 'MISMATCH'}")
        return

    if not a.blob:
        raise SystemExit("check needs the blob path")
    with open(a.blob, "rb") as fh:
        raw = fh.read()
    ents, consumed = walk(raw)
    obs = collections.Counter(e["size"] for e in ents)

    pred = collections.Counter()
    print("closed form vs the block schema, per width:")
    for C in (32, 64, 128, 256):
        n, down, up = TIERS[C]
        ok = plain(C) == closed_form(C)
        print(f"  C={C:<5} plain {plain(C):>9,}  closed form {closed_form(C):>9,}"
              f"  {'ok' if ok else 'MISMATCH'}")
        pred[plain(C)] += n
        pred[down] += 1
        pred[up] += 1
    for sz, n in WIDE.items():
        pred[sz] += n

    print("\npredicted vs observed, by blob size:")
    sizes = sorted(set(pred) | set(obs), reverse=True)
    bad = 0
    for s in sizes:
        d = pred[s] - obs[s]
        if d:
            bad += 1
        flag = "" if d == 0 else f"   <-- off by {d:+d}"
        print(f"  {s:>11,}  predicted x{pred[s]:<3} observed x{obs[s]:<3}{flag}")
    pb = sum(s * n for s, n in pred.items())
    ob = sum(s * n for s, n in obs.items())
    print(f"\nblobs   predicted {sum(pred.values()):>4}   observed {sum(obs.values()):>4}")
    print(f"bytes   predicted {pb:>12,}   observed {ob:>12,}   residual {ob - pb}")
    print(f"walk consumed {consumed:,} / {len(raw):,}")
    if bad == 0 and pb == ob:
        print("\nPASS - the schema reproduces every blob size and every byte.")
        print("It is a size model, not a forward pass: slab factorisation is still open.")
    else:
        print(f"\nFAIL - {bad} size(s) disagree. The schema is wrong somewhere.")


if __name__ == "__main__":
    main()
