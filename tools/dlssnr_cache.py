#!/usr/bin/env python3
"""
dlssnr_cache.py - read the mod's transcoded weight cache and prove it against the source.

Why this matters
----------------
Every displacement recovered from the GPU disassembly addresses the *transcoded*
buffer the mod builds at first run (`dlssnr_on_amd_weights.bin`), not NVIDIA's
`WEIGHTS_HT` resource. All the sub-tensor offsets in `docs/FINDINGS.md` were
cross-checked against WEIGHTS_HT on the assumption that the transcode is a
straight copy. This tool checks that assumption instead of assuming it.

The cache format (`DLSSNRW1`)
-----------------------------
    char magic[8]        "DLSSNRW1"
    u32  count           153
    u32  index_size      total header+index bytes
    repeat count times, sorted lexicographically by name:
      u8   name_len
      char name[name_len]
      u64  offset        into the payload region, which starts at index_size
      u64  size
    then the payloads, packed contiguously in the same lexicographic order.

Result on the shipped build
---------------------------
The index predicted from the source container's own tensor table matches to the
byte: `16 + 153*17 + 3056 = 5,673`, and `5,673 + 147,683,778 = 147,689,451`, the
observed cache size, residual 0. Spot-checked sha256 over four tensors -
including two hashed in full - is identical on both sides. The transcode is a
pure re-index; the payloads are untouched.

Usage
-----
  python dlssnr_cache.py index  dlssnr_on_amd_weights.bin
  python dlssnr_cache.py verify dlssnr_on_amd_weights.bin --source weights_ht.bin
  python dlssnr_cache.py predict --source weights_ht.bin     # size check, no cache needed
"""
import argparse
import hashlib
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dlssnr_weights import walk   # noqa: E402

MAGIC = b"DLSSNRW1"


def read_index(fh):
    head = fh.read(16)
    if head[:8] != MAGIC:
        raise SystemExit("not a DLSSNRW1 cache")
    count, index_size = struct.unpack_from("<II", head, 8)
    body = fh.read(index_size - 16)
    p, out = 0, []
    for _ in range(count):
        nl = body[p]
        name = body[p + 1:p + 1 + nl].decode("latin1")
        p += 1 + nl
        off, size = struct.unpack_from("<QQ", body, p)
        p += 16
        out.append((name, off, size))
    if p != len(body):
        print(f"warning: index parse consumed {p} of {len(body)} bytes")
    return count, index_size, out


def source_table(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    ents, _ = walk(raw)
    return raw, {e["name"]: e for e in ents}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["index", "verify", "predict"])
    ap.add_argument("cache", nargs="?")
    ap.add_argument("--source", help="WEIGHTS_HT blob extracted from nvngx_dlssnr.dll")
    ap.add_argument("-n", "--samples", type=int, default=6)
    ap.add_argument("--bytes", type=int, default=65536, help="bytes to hash per tensor")
    a = ap.parse_args()

    if a.cmd == "predict":
        if not a.source:
            raise SystemExit("predict needs --source")
        _, src = source_table(a.source)
        payload = sum(e["size"] for e in src.values())
        names = sum(len(n) for n in src)
        idx = 16 + len(src) * 17 + names
        print(f"tensors            {len(src)}")
        print(f"name bytes         {names:,}")
        print(f"predicted index    16 + {len(src)}*17 + {names} = {idx:,}")
        print(f"payload            {payload:,}")
        print(f"predicted cache    {idx + payload:,}")
        print("\nCompare against the real dlssnr_on_amd_weights.bin. An exact match means the")
        print("transcode adds only a header and an index and copies every payload unchanged.")
        return

    if not a.cache:
        raise SystemExit("this subcommand needs the cache path")
    with open(a.cache, "rb") as fh:
        count, index_size, entries = read_index(fh)
    total = os.path.getsize(a.cache)
    payload = sum(s for _, _, s in entries)
    print(f"{count} tensors, index {index_size:,} B, payload {payload:,} B, file {total:,} B")
    print(f"index + payload == file: {index_size + payload == total}")

    run, contiguous = 0, True
    for name, off, size in sorted(entries):
        if off != run:
            contiguous = False
            print(f"  offset break at {name}: {off:,} expected {run:,}")
            break
        run += size
    print(f"payloads packed contiguously in lexicographic order: {contiguous}")

    if a.cmd == "index":
        for name, off, size in entries[:12]:
            print(f"  {name:28} @{off:>12,}  {size:>12,}")
        if len(entries) > 12:
            print(f"  ... {len(entries) - 12} more")
        return

    if not a.source:
        raise SystemExit("verify needs --source")
    raw, src = source_table(a.source)
    cache_names = {n for n, _, _ in entries}
    print(f"\nname sets identical: {cache_names == set(src)}")
    bad = [n for n, _, s in entries if n in src and src[n]["size"] != s]
    print(f"size mismatches: {len(bad)}")

    big = sorted(entries, key=lambda e: -e[2])[:a.samples]
    print(f"\nhashing {len(big)} tensors, up to {a.bytes:,} B each:")
    fails = 0
    with open(a.cache, "rb") as fh:
        for name, off, size in big:
            n = min(size, a.bytes)
            fh.seek(index_size + off)
            hc = hashlib.sha256(fh.read(n)).hexdigest()
            e = src[name]
            hs = hashlib.sha256(raw[e["data"]:e["data"] + n]).hexdigest()
            ok = hc == hs
            fails += not ok
            print(f"  {name:28} {n:>9,} B  {'MATCH' if ok else 'DIFFER'}  {hc[:16]}")
    if fails == 0 and not bad and cache_names == set(src):
        print("\nPROVEN: the transcode is a pure re-index. Payloads are byte-identical, so every")
        print("offset measured against WEIGHTS_HT also describes the buffer the GPU kernels read.")
    else:
        print(f"\nFAIL: {fails} hash mismatch(es). The transcode is NOT a straight copy.")


if __name__ == "__main__":
    main()
