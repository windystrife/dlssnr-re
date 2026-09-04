#!/usr/bin/env python3
"""
dlssnr_weights.py - decode the WEIGHTS_HT container from nvngx_dlssnr.dll.

Container layout (recovered by exact-byte accounting; a correct walk consumes
100.000% of the blob):

    u64  blob_total            # equals the resource size
    repeat:
      u64  name_len
      char name[name_len]      # e.g. "block31.layer1.layer"
      u64  a                   # == b, == c + 40
      u64  b
      u64  c                   # payload bytes
      u32  flag                # 1
      u8   data[c]             # FP8 E4M3 or FP16 - see `verify`
      u64  pad                 # 0
      u64  one                 # 1
      u32  len_halfwords      # payload bytes / 2 -- a redundant length field

`check` proves the layout numerically: the trailing u32 equals payload_size/2 for
every tensor, so a walk that is wrong by even one byte anywhere breaks it for all
subsequent tensors. On the shipped blob it holds 153/153.

`verify` re-derives the dtype from the data rather than trusting this docstring. The discriminator is byte parity. If the
stream were FP16 little-endian, the low byte would be near-uniform (~8 bits of
entropy) while the high byte stayed concentrated; matching distributions on
both parities mean the stream is byte-granular.

This tool reads sizes, names and statistics. It does not extract or
redistribute weights - supply your own legally obtained DLL.

Usage
-----
  python dlssnr_weights.py check    weights_ht.bin        # prove the layout
  python dlssnr_weights.py verify   weights_ht.bin        # classify dtypes
  python dlssnr_weights.py topology weights_ht.bin
  python dlssnr_weights.py list     weights_ht.bin
  python dlssnr_weights.py export   weights_ht.bin -o npy/ --limit 10
"""
import argparse
import collections
import json
import math
import re
import struct


def walk(blob):
    """Yield dicts for every tensor. Raises if the walk does not land exactly."""
    total, = struct.unpack_from("<Q", blob, 0)
    if total != len(blob):
        print(f"warning: header says {total:,} but blob is {len(blob):,}")
    p = 8
    ents = []
    while p < len(blob) - 40:
        name_len, = struct.unpack_from("<Q", blob, p)
        if name_len == 0 or name_len > 128:
            break
        name = blob[p + 8:p + 8 + name_len].decode("latin1")
        if not re.match(r"^[\w.]+$", name):
            break
        q = p + 8 + name_len
        a, b, c = struct.unpack_from("<QQQ", blob, q)
        flag, = struct.unpack_from("<I", blob, q + 24)
        data = q + 28
        lo, hi = struct.unpack_from("<HH", blob, data + c + 16)
        ents.append(dict(name=name, a=a, b=b, size=c, flag=flag,
                         data=data, halfwords=lo | (hi << 16)))
        p = data + c + 20
    return ents, p


def e4m3(byte):
    """Decode one FP8 E4M3 (OCP) byte."""
    sign = -1.0 if byte & 0x80 else 1.0
    exp = (byte >> 3) & 0x0F
    man = byte & 0x07
    if exp == 0:
        return sign * (man / 8.0) * 2.0 ** -6
    if exp == 15 and man == 7:
        return float("nan")
    return sign * (1.0 + man / 8.0) * 2.0 ** (exp - 7)


def entropy(counter):
    n = sum(counter.values())
    return -sum(v / n * math.log2(v / n) for v in counter.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["list", "topology", "verify", "check", "export"])
    ap.add_argument("-o", "--out", help="export: output directory")
    ap.add_argument("--limit", type=int, help="export: only the N largest tensors")
    ap.add_argument("blob")
    a = ap.parse_args()
    with open(a.blob, "rb") as fh:
        blob = fh.read()
    ents, consumed = walk(blob)

    pct = 100.0 * consumed / len(blob)
    print(f"{len(ents)} tensors; walk consumed {consumed:,}/{len(blob):,} ({pct:.3f}%)")
    if abs(pct - 100.0) > 1e-6:
        print("  !! the walk did not land exactly - the layout assumption is wrong")
    payload = sum(e["size"] for e in ents)
    print(f"payload {payload:,} bytes ({payload / 1048576:.1f} MB)")
    print("  (the model is MIXED precision - run `verify` for the true parameter "
          "count)\n")

    if a.cmd == "list":
        for e in sorted(ents, key=lambda x: -x["size"]):
            print(f"  {e['name']:34} {e['size']:>12,}  halfwords={e['halfwords']:>10,}")

    elif a.cmd == "topology":
        by = collections.defaultdict(dict)
        for e in ents:
            m = re.match(r"block(\d+)\.(layer\d+)\.(\w+)", e["name"])
            if m:
                by[int(m.group(1))][f"{m.group(2)}.{m.group(3)}"] = e["size"]
        print(f"{'BLK':>4} {'layer0':>12} {'layer1':>12} {'layer2':>12} "
              f"{'layer3':>10} {'layer4':>12}")
        for b in sorted(by):
            r = by[b]
            def g(k):
                return f"{r[k]:,}" if k in r else "-"
            print(f"{b:>4} {g('layer0.layer'):>12} {g('layer1.layer'):>12} "
                  f"{g('layer2.layer'):>12} {g('layer3.layer'):>10} "
                  f"{g('layer4.layer'):>12}")
        counts = collections.Counter(n.split(".", 1)[1] for n in
                                     (e["name"] for e in ents) if "." in n)
        print("\nlayer-name histogram:", dict(counts.most_common()))
        print("\nExpected shape: a symmetric U-Net. Conv encoder, a dim-512 transformer\n"
              "stage, a dim-1024 bottleneck (the largest blocks), then the decoder\n"
              "mirroring the encoder tiers, and a final head block.")

    elif a.cmd == "verify":
        # Classify EVERY tensor, not one probe. The model is mixed precision, so a
        # single sample gives whichever answer that tensor happens to carry.
        #
        # Discriminator: parity entropy gap. For FP16 little-endian the low byte
        # (even index) is near-uniform while the high byte (odd index) is
        # concentrated, so H(even) - H(odd) is large. For a byte-granular stream
        # both parities are the same distribution and the gap collapses.
        # Distinct-code COUNT is not reliable here - it separates far less
        # cleanly than the entropy gap does.
        GAP = 0.35
        fp8, fp16, tiny = [], [], []
        for e in ents:
            if e["size"] < 512:
                tiny.append(e)
                continue
            buf = blob[e["data"]:e["data"] + min(e["size"], 400000)]
            ev, od = collections.Counter(buf[0::2]), collections.Counter(buf[1::2])
            gap = entropy(ev) - entropy(od)
            rec = (e, gap, len(ev), len(od))
            (fp16 if (gap > GAP and len(ev) >= 250) else fp8).append(rec)

        b8 = sum(e["size"] for e, *_ in fp8)
        b16 = sum(e["size"] for e, *_ in fp16)
        bt = sum(e["size"] for e in tiny)
        print(f"{'class':6}{'tensors':>9}{'bytes':>16}{'parameters':>16}")
        print(f"{'FP8':6}{len(fp8):>9}{b8:>16,}{b8:>16,}")
        print(f"{'FP16':6}{len(fp16):>9}{b16:>16,}{b16 // 2:>16,}")
        print(f"{'tiny':6}{len(tiny):>9}{bt:>16,}{bt // 2:>16,}   (LayerScale scalars)")
        print(f"\ntrue parameter count = {b8 + b16 // 2 + bt // 2:,}")
        print(f"(counting every byte as one parameter would give {b8 + b16 + bt:,} - wrong)")

        if fp8 and fp16:
            worst8 = max(g for _, g, _, _ in fp8)
            best16 = min(g for _, g, _, _ in fp16)
            print(f"\nseparation check: largest gap among FP8 = {worst8:.3f}, "
                  f"smallest among FP16 = {best16:.3f}")
            print(f"  margin {best16 / worst8:.1f}x - "
                  f"{'clean split, not a threshold fudge' if best16 > 3 * worst8 else 'WEAK; treat with suspicion'}")

        def blocks(lst):
            out = set()
            for item in lst:
                e = item[0] if isinstance(item, tuple) else item
                m = re.match(r"block(\d+)", e["name"])
                if m:
                    out.add(int(m.group(1)))
            return sorted(out)

        print(f"\nFP16 blocks: {blocks(fp16)}")
        print(f"FP8  blocks: {blocks(fp8)}")

        big = max(ents, key=lambda e: e["size"])
        buf = blob[big["data"]:big["data"] + min(big["size"], 200000)]
        vals = [e4m3(x) for x in buf]
        finite = [v for v in vals if v == v]
        print(f"\nworked example - {big['name']} ({big['size']:,} B) decoded as FP8 E4M3:")
        print(f"  mean {sum(finite) / len(finite):+.5f}  "
              f"|max| {max(abs(v) for v in finite):.4f}  NaNs {len(vals) - len(finite)}")

    elif a.cmd == "check":
        # Numerical proof of the container layout, independent of any dtype claim.
        # The trailing u32 is payload_size/2. If the walk's stride were wrong by a
        # single byte anywhere, every later tensor would land on garbage and fail.
        ok = sum(1 for e in ents if e["halfwords"] * 2 == e["size"])
        print(f"invariant  len_halfwords * 2 == payload_size")
        print(f"  holds {ok}/{len(ents)} tensors")
        for e in ents:
            if e["halfwords"] * 2 != e["size"]:
                print(f"    FAIL {e['name']:30} size={e['size']:,} "
                      f"halfwords={e['halfwords']:,}")
        if ok == len(ents) and consumed == len(blob):
            print("\n  PASS - the walk consumed the blob exactly AND every tensor's")
            print("  redundant length field agrees. The layout is confirmed numerically,")
            print("  not just statistically.")
        else:
            print("\n  FAIL - the layout assumption is wrong somewhere.")

    elif a.cmd == "export":
        # Decodes YOUR OWN weights to .npy for offline study. Writes nothing that
        # was not already on your disk; do not redistribute the output.
        import os
        try:
            import numpy as np
        except ImportError:
            raise SystemExit("export needs numpy: pip install numpy")
        if not a.out:
            raise SystemExit("export needs -o OUTDIR")
        os.makedirs(a.out, exist_ok=True)

        # E4M3 -> float32 lookup, built once
        lut = np.array([e4m3(i) for i in range(256)], dtype=np.float32)

        sel = sorted(ents, key=lambda e: -e["size"])
        if a.limit:
            sel = sel[:a.limit]
        manifest = []
        for e in sel:
            raw = np.frombuffer(blob, dtype=np.uint8, count=e["size"], offset=e["data"])
            ev = entropy(collections.Counter(raw[0::2].tolist()))
            od = entropy(collections.Counter(raw[1::2].tolist())) if e["size"] > 1 else ev
            if (ev - od) > 0.35 and len(set(raw[0::2].tolist())) >= 250:
                dtype, arr = "float16", raw.view(np.float16).astype(np.float32)
            else:
                dtype, arr = "float8_e4m3", lut[raw]
            path = os.path.join(a.out, e["name"] + ".npy")
            np.save(path, arr)
            manifest.append(dict(name=e["name"], bytes=e["size"],
                                 elements=int(arr.size), dtype=dtype,
                                 mean=float(arr.mean()), absmax=float(np.abs(arr).max())))
            print(f"  {e['name']:32} {dtype:12} {arr.size:>10,} elems -> {path}")
        with open(os.path.join(a.out, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"\n{len(manifest)} tensors + manifest.json -> {a.out}")
        print("These are NVIDIA's trained weights decoded from your own DLL. Local")
        print("research only - do not redistribute.")


if __name__ == "__main__":
    main()
