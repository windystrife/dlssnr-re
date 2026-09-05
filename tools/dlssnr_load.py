#!/usr/bin/env python3
"""
dlssnr_load.py - load the DLSS-NR weights as real, typed, shaped numpy arrays.

This is the loader a reimplementation would sit on top of. It takes the weight
blob and hands back, for every block, the named sub-tensors decoded to float32:

    A0  4C^2  FP8 E4M3     row length 128 at C=32 (ISA-proven from `idx >> 7`)
    A1  128C  FP8 E4M3
    A2  C^2   FP8 E4M3     absent at C=32
    g1  2C    FP16         gain vector
    Q   3C^2  FP8 E4M3     row length 3C at C=32 (ISA-proven, magic divide /96)
    T   256C  FP16         all values <= 0, max exactly 0.0
    S   4*(C/32) FP32
    P   C^2   FP8 E4M3
    g2  2C    FP16

Shapes are given only where they are established. Where the byte count admits
several factorisations the array is returned FLAT, with `.shape_known = False`,
rather than guessing - 262,144 bytes is equally 1024x256, 512x512 or 4x(256x256)
and nothing in the container picks one.

Usage
-----
  python dlssnr_load.py summary weights_ht.bin
  python dlssnr_load.py block   weights_ht.bin -b 1
  python dlssnr_load.py stats   weights_ht.bin
  python dlssnr_load.py save    weights_ht.bin -o out/          # .npz per block
"""
import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dlssnr_weights import walk   # noqa: E402

# block -> tier width, from the mirrored tier structure
TIER = {}
for b in list(range(0, 5)) + list(range(66, 71)):
    TIER[b] = 32
for b in list(range(5, 9)) + list(range(62, 66)):
    TIER[b] = 64
for b in list(range(9, 15)) + list(range(56, 62)):
    TIER[b] = 128
for b in list(range(15, 23)) + list(range(48, 56)):
    TIER[b] = 256
for b in list(range(23, 31)) + list(range(40, 48)):
    TIER[b] = 512
for b in range(31, 39):
    TIER[b] = 1024
TIER[39] = 512

# blocks whose layout departs from the plain schema
STEM, HEAD = 0, 70
DOWN = {4: 32, 8: 64, 14: 128, 22: 256}          # append 2C^2
UP = {66: 32, 62: 64, 56: 128, 48: 256}          # prepend 2C^2


def e4m3_lut():
    """256-entry FP8 E4M3 -> float32 lookup."""
    out = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        ex, ma = (b >> 3) & 0xF, b & 7
        if ex == 0:
            out[b] = s * (ma / 8.0) * 2.0 ** -6
        elif ex == 15 and ma == 7:
            out[b] = np.nan
        else:
            out[b] = s * (1.0 + ma / 8.0) * 2.0 ** (ex - 7)
    return out


LUT = e4m3_lut()


def fp8(buf):
    return LUT[np.frombuffer(buf, dtype=np.uint8)]


def fp16(buf):
    return np.frombuffer(buf, dtype=np.float16).astype(np.float32)


def fp32(buf):
    return np.frombuffer(buf, dtype=np.float32).copy()


def _core(C):
    """A0/A1/A2 in file order."""
    L = [("A0", 4 * C * C, "f8"), ("A1", 128 * C, "f8")]
    if C != 32:
        L.append(("A2", C * C, "f8"))
    return L


def _tail(C, trailing_pad=True):
    L = [("Q", 3 * C * C, "f8"), ("T", 256 * C, "f16"), ("S", 4 * (C // 32), "f32")]
    if C < 128:
        L.append(("_p2", 16 - 4 * (C // 32), "z"))
    L += [("P", C * C, "f8"), ("g2", 2 * C, "f16")]
    if trailing_pad:
        L.append(("_p3", 16, "z"))
    return L


def variants(C, block):
    """Candidate layouts for this block, most specific first.

    The down/up blocks are not the plain schema plus a slab: the padding around
    the gain group behaves differently, and it behaves differently again at
    C=32. Rather than hard-code one guess, offer the candidates and let the
    exact byte count choose - a layout that does not total the blob size is
    simply wrong."""
    plain = _core(C) + [("_p0", 16, "z"), ("g1", 2 * C, "f16"), ("_p1", 16, "z")] + _tail(C)
    out = [("plain", plain)]

    # downsample: one 2C^2 slab appended, trailing pad kept at C=32, absorbed above
    body = _core(C) + [("_p0", 16, "z"), ("g1", 2 * C, "f16"), ("_p1", 16, "z")] + _tail(C, False)
    out.append(("down/pad-kept", body + [("post", 2 * C * C, "f8"), ("_p3", 16, "z")]))
    out.append(("down/pad-absorbed", body + [("post", 2 * C * C, "f8")]))

    # upsample: 2C^2 prepended; the pad|g1|pad group gains a second 2C vector,
    # and at C>=64 the two pads disappear
    pre = [("pre", 2 * C * C, "f8")]
    out.append(("up/padded", pre + _core(C) +
                [("_p0", 16, "z"), ("g1", 2 * C, "f16"), ("_p1", 16, "z"),
                 ("g1b", 2 * C, "f16")] + _tail(C)))
    out.append(("up/unpadded", pre + _core(C) +
                [("g1", 2 * C, "f16"), ("g1b", 2 * C, "f16")] + _tail(C)))
    return out


# row lengths that are ISA-proven, keyed by (part, C)
KNOWN_ROWS = {("A0", 32): 128, ("Q", 32): 96}


class Tensor:
    def __init__(self, name, arr, dtype, shape_known):
        self.name, self.data, self.dtype = name, arr, dtype
        self.shape_known = shape_known

    def __repr__(self):
        sh = "x".join(str(d) for d in self.data.shape)
        return f"<{self.name} {self.dtype} [{sh}]{'' if self.shape_known else ' flat/unknown'}>"


def load_block(raw, ent, C):
    """Split one blob into its schema parts, decoded to float32."""
    m = ent["name"]
    b = int(m.split(".")[0][5:])
    layout = kind = None
    for k, cand in variants(C, b):
        if sum(n for _, n, _ in cand) == ent["size"]:
            layout, kind = cand, k
            break
    if layout is None:
        return None, [(k, sum(n for _, n, _ in c)) for k, c in variants(C, b)]
    out, off = {"_layout": kind}, ent["data"]
    for nm, n, dt in layout:
        buf = raw[off:off + n]
        off += n
        if dt == "z":
            continue
        arr = {"f8": fp8, "f16": fp16, "f32": fp32}[dt](buf)
        rows = KNOWN_ROWS.get((nm, C))
        known = False
        if rows and arr.size % rows == 0:
            arr = arr.reshape(-1, rows)
            known = True
        out[nm] = Tensor(nm, arr, {"f8": "float8_e4m3", "f16": "float16",
                                   "f32": "float32"}[dt], known)
    return out, kind


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["summary", "block", "stats", "save"])
    ap.add_argument("blob")
    ap.add_argument("-b", "--block", type=int, default=1)
    ap.add_argument("-o", "--out")
    a = ap.parse_args()

    with open(a.blob, "rb") as fh:
        raw = fh.read()
    ents, _ = walk(raw)
    byname = {e["name"]: e for e in ents}

    def blocks_with_layer0():
        for e in ents:
            p = e["name"].split(".")
            if len(p) == 3 and p[1] == "layer0" and p[2] == "layer":
                yield int(p[0][5:]), e

    if a.cmd in ("summary", "stats", "save"):
        ok = fail = 0
        allparts = collections.Counter()
        rows = []
        for b, e in sorted(blocks_with_layer0()):
            C = TIER.get(b)
            if C is None or C >= 512 or b in (STEM, HEAD):
                continue                      # transformer tiers use a different layout
            parts, need = load_block(raw, e, C)
            if parts is None:
                fail += 1
                rows.append((b, C, e["size"], None, need))
                continue
            ok += 1
            rows.append((b, C, e["size"], parts, need))
        print(f"blocks decoded {ok}, layout mismatch {fail}")
        if a.cmd == "summary":
            print(f"\n{'blk':>4}{'C':>6}{'bytes':>10}  parts")
            for b, C, sz, parts, need in rows:
                if parts is None:
                    print(f"{b:>4}{C:>6}{sz:>10}  NO LAYOUT FITS; candidates {need}")
                else:
                    print(f"{b:>4}{C:>6}{sz:>10}  [{need}] " +
                          " ".join(f"{k}:{v.data.size:,}" for k, v in parts.items()
                                   if k != "_layout"))
        elif a.cmd == "stats":
            print(f"\n{'part':>5}{'dtype':>14}{'count':>8}{'elements':>14}"
                  f"{'min':>10}{'max':>10}{'NaN':>6}")
            agg = collections.defaultdict(list)
            for b, C, sz, parts, _ in rows:
                if parts:
                    for k, v in parts.items():
                        if k != "_layout":
                            agg[(k, v.dtype)].append(v.data)
            for (k, dt), arrs in sorted(agg.items()):
                cat = np.concatenate([x.ravel() for x in arrs])
                nan = int(np.isnan(cat).sum())
                fin = cat[~np.isnan(cat)]
                print(f"{k:>5}{dt:>14}{len(arrs):>8}{cat.size:>14,}"
                      f"{fin.min():>10.4f}{fin.max():>10.4f}{nan:>6}")
            t = agg.get(("T", "float16"))
            if t:
                cat = np.concatenate([x.ravel() for x in t])
                print(f"\n  T check: max = {cat.max():.6f}, "
                      f"all <= 0 -> {bool((cat <= 0).all())}, "
                      f"exact zeros {int((cat == 0).sum()):,}")
        elif a.cmd == "save":
            if not a.out:
                raise SystemExit("save needs -o")
            os.makedirs(a.out, exist_ok=True)
            n = 0
            for b, C, sz, parts, _ in rows:
                if parts:
                    np.savez(os.path.join(a.out, f"block{b}.npz"),
                             **{k: v.data for k, v in parts.items() if k != "_layout"})
                    n += 1
            print(f"wrote {n} .npz files to {a.out}")
        return

    b = a.block
    C = TIER.get(b)
    e = byname.get(f"block{b}.layer0.layer")
    if e is None or C is None:
        raise SystemExit(f"no block{b}")
    parts, need = load_block(raw, e, C)
    print(f"block{b}  C={C}  blob {e['size']:,} B  layout={need}")
    if parts is None:
        raise SystemExit(f"no layout fits; candidates {need}")
    for k, v in parts.items():
        if k == "_layout":
            continue
        d = v.data
        fin = d[~np.isnan(d)]
        print(f"  {v!r:44} n={d.size:>9,} min={fin.min():+.4f} "
              f"max={fin.max():+.4f} mean={fin.mean():+.5f} std={fin.std():.4f}")


if __name__ == "__main__":
    main()
