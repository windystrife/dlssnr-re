#!/usr/bin/env python3
"""
dlssnr_bundle.py - split a clang offload bundle into per-GPU AMDGPU ELFs and
report each kernel's register / LDS budget.

The shim keeps its GPU code in the `.hip_fat` PE section as a
`__CLANG_OFFLOAD_BUNDLE__`. This tool splits it, then reads the AMDGPU
msgpack note (`amdhsa.kernels`) to recover, per kernel:
    .vgpr_count  .sgpr_count  .agpr_count
    .group_segment_fixed_size   (LDS bytes per workgroup)
    .private_segment_fixed_size (scratch; non-zero implies a real call)
    .vgpr_spill_count  .max_flat_workgroup_size  .wavefront_size
    .args[0].size               (the kernel's parameter struct size)

msgpack is optional; without it you still get sections, sizes and symbols.

Usage
-----
  python dlssnr_bundle.py split  hip_fat.bin -o out/
  python dlssnr_bundle.py kernels out/bundle-gfx1201.bin
  python dlssnr_bundle.py compare out/bundle-gfx1100.bin out/bundle-gfx1201.bin
  python dlssnr_bundle.py text    out/bundle-gfx1201.bin -o text.bin --syms syms.txt
"""
import argparse
import os
import struct
import sys

MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
STT_AMDGPU_HSA_KERNEL = 10


# ---------------------------------------------------------------- bundle
def split_bundle(blob):
    if not blob.startswith(MAGIC):
        raise ValueError("not a clang offload bundle")
    off = len(MAGIC)
    n, = struct.unpack_from("<Q", blob, off)
    off += 8
    out = []
    for _ in range(n):
        boff, bsize, tlen = struct.unpack_from("<QQQ", blob, off)
        off += 24
        triple = blob[off:off + tlen].decode("latin1")
        off += tlen
        out.append((triple, blob[boff:boff + bsize]))
    return out


# ------------------------------------------------------------------- ELF
class Elf:
    def __init__(self, data):
        if data[:4] != b"\x7fELF":
            raise ValueError("not an ELF")
        self.d = data
        self.e_flags, = struct.unpack_from("<I", data, 0x30)
        shoff, = struct.unpack_from("<Q", data, 0x28)
        shent, = struct.unpack_from("<H", data, 0x3A)
        shnum, = struct.unpack_from("<H", data, 0x3C)
        shstr, = struct.unpack_from("<H", data, 0x3E)
        self.secs = []
        for i in range(shnum):
            o = shoff + i * shent
            v = struct.unpack_from("<IIQQQQIIQQ", data, o)
            self.secs.append(dict(nameoff=v[0], type=v[1], addr=v[3], off=v[4],
                                  size=v[5], link=v[6], entsize=v[9]))
        st = self.secs[shstr]
        for s in self.secs:
            s["name"] = self._str(st, s["nameoff"])

    def _str(self, strtab, x):
        base = strtab["off"] + x
        return self.d[base:self.d.index(b"\0", base)].decode("latin1")

    def sec(self, name):
        for s in self.secs:
            if s["name"] == name:
                return s
        return None

    def symbols(self):
        """(name, value, size, type) from .symtab, falling back to .dynsym."""
        for want in (".symtab", ".dynsym"):
            s = self.sec(want)
            if not s or not s["entsize"]:
                continue
            strt = self.secs[s["link"]]
            out = []
            for i in range(s["size"] // s["entsize"]):
                o = s["off"] + i * s["entsize"]
                nameoff, info, _, _, value, size = struct.unpack_from("<IBBHQQ", self.d, o)
                name = self._str(strt, nameoff)
                if name:
                    out.append((name, value, size, info & 0xF))
            if out:
                return out
        return []

    def kernel_metadata(self):
        """amdhsa.kernels from the AMDGPU msgpack note, or None."""
        try:
            import msgpack
        except ImportError:
            return None
        s = self.sec(".note")
        if not s:
            return None
        buf = self.d[s["off"]:s["off"] + s["size"]]
        off = 0
        while off + 12 <= len(buf):
            nsz, dsz, typ = struct.unpack_from("<III", buf, off)
            do = off + 12 + ((nsz + 3) // 4) * 4
            payload = buf[do:do + dsz]
            if b"amdhsa.kernels" in payload[:256]:
                md = msgpack.unpackb(payload, strict_map_key=False, raw=False)
                return md.get("amdhsa.kernels", [])
            off = do + ((dsz + 3) // 4) * 4
        return None


def kernel_rows(elf):
    """Per-kernel rows, from msgpack when available, else from symbols."""
    md = elf.kernel_metadata()
    sizes = {n: sz for n, _, sz, t in elf.symbols()}
    if md is not None:
        rows = []
        for k in md:
            name = k.get(".name", "?")
            args = k.get(".args") or []
            rows.append(dict(
                name=name,
                code=sizes.get(name, 0),
                vgpr=k.get(".vgpr_count", 0),
                sgpr=k.get(".sgpr_count", 0),
                agpr=k.get(".agpr_count", 0),
                lds=k.get(".group_segment_fixed_size", 0),
                scratch=k.get(".private_segment_fixed_size", 0),
                spill=k.get(".vgpr_spill_count", 0),
                wg=k.get(".max_flat_workgroup_size", 0),
                wave=k.get(".wavefront_size", 0),
                argsz=(args[0].get(".size") if args else None),
            ))
        return rows
    rows = []
    for name, _, size, typ in elf.symbols():
        if name.endswith(".kd"):
            base = name[:-3]
            rows.append(dict(name=base, code=sizes.get(base, 0), vgpr=0, sgpr=0,
                             agpr=0, lds=0, scratch=0, spill=0, wg=0, wave=0,
                             argsz=None))
    return rows


LDS_CEILING = 65536


def print_kernels(rows, title):
    rows = sorted(rows, key=lambda r: -r["lds"])
    print(f"\n=== {title} : {len(rows)} kernels ===")
    print(f"{'KERNEL':52}{'CODE':>9}{'VGPR':>6}{'SGPR':>6}{'LDS':>9}{'LDS%':>7}"
          f"{'SCRATCH':>9}{'SPILL':>7}{'WG':>6}{'ARGSZ':>7}")
    for r in rows:
        pct = 100.0 * r["lds"] / LDS_CEILING if r["lds"] else 0.0
        flag = " <" if pct >= 85 else ""
        print(f"{r['name'][:52]:52}{r['code']:>9}{r['vgpr']:>6}{r['sgpr']:>6}"
              f"{r['lds']:>9}{pct:>6.1f}%{r['scratch']:>9}{r['spill']:>7}"
              f"{r['wg']:>6}{str(r['argsz'] if r['argsz'] is not None else '-'):>7}{flag}")
    tot = sum(r["code"] for r in rows)
    print(f"{'TOTAL code':52}{tot:>9}")
    hot = [r for r in rows if r["lds"] >= 0.85 * LDS_CEILING]
    if hot:
        print(f"\n  {len(hot)} kernel(s) at >=85% of the {LDS_CEILING} B LDS ceiling "
              f"-> pinned to 2 workgroups/WGP = 4 of 16 waves/SIMD (25%):")
        for r in hot:
            print(f"    {r['name']}  LDS={r['lds']}  VGPR={r['vgpr']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("split"); p.add_argument("blob"); p.add_argument("-o", "--out", default=".")
    p = sub.add_parser("kernels"); p.add_argument("elf")
    p = sub.add_parser("compare"); p.add_argument("a"); p.add_argument("b")
    p = sub.add_parser("text"); p.add_argument("elf")
    p.add_argument("-o", "--out", required=True); p.add_argument("--syms", required=True)
    a = ap.parse_args()

    if a.cmd == "split":
        with open(a.blob, "rb") as fh:
            blob = fh.read()
        os.makedirs(a.out, exist_ok=True)
        for triple, data in split_bundle(blob):
            gfx = triple.rsplit("-", 1)[-1] or "host"
            path = os.path.join(a.out, f"bundle-{gfx}.bin")
            with open(path, "wb") as fh:
                fh.write(data)
            print(f"{triple:52} {len(data):>10,} B -> {path}")

    elif a.cmd == "kernels":
        with open(a.elf, "rb") as fh:
            elf = Elf(fh.read())
        rows = kernel_rows(elf)
        if elf.kernel_metadata() is None:
            print("note: install `msgpack` for register/LDS detail", file=sys.stderr)
        print_kernels(rows, os.path.basename(a.elf))

    elif a.cmd == "compare":
        with open(a.a, "rb") as fh:
            ea = Elf(fh.read())
        with open(a.b, "rb") as fh:
            eb = Elf(fh.read())

        def text_syms(elf):
            """Every sized .text symbol - kernels AND out-of-line helpers.

            Using only `amdhsa.kernels` here would hide exactly the thing this
            subcommand exists to catch: a shared helper that one target inlines
            and the other calls."""
            t = elf.sec(".text")
            out = {}
            for n, v, sz, ty in elf.symbols():
                if sz > 0 and not n.endswith(".kd") \
                        and t["addr"] <= v < t["addr"] + t["size"]:
                    out[n] = sz
            return out

        ra, rb = text_syms(ea), text_syms(eb)
        na, nb = os.path.basename(a.a), os.path.basename(a.b)
        print(f"{'SYMBOL':50}{na[:14]:>15}{nb[:14]:>15}{'ratio':>9}")
        ta = tb = 0
        only_a, only_b = [], []
        for name in sorted(set(ra) | set(rb), key=lambda n: -ra.get(n, 0)):
            ca, cb = ra.get(name, 0), rb.get(name, 0)
            ta += ca; tb += cb
            ratio = f"{cb / ca:.2f}x" if ca else "n/a"
            mark = ""
            if name not in rb:
                mark = "  <-- only in A"; only_a.append((name, ca))
            elif name not in ra:
                mark = "  <-- only in B"; only_b.append((name, cb))
            print(f"{name[:50]:50}{ca:>15,}{cb:>15,}{ratio:>9}{mark}")
        print("-" * 89)
        print(f"{'TOTAL .text symbols':50}{ta:>15,}{tb:>15,}"
              f"{(f'{tb / ta:.3f}x' if ta else 'n/a'):>9}")

        if only_a or only_b:
            print("\n!! ASYMMETRIC SYMBOLS - read this before quoting any size ratio.")
            for name, sz in only_a:
                print(f"   only in A: {name}  ({sz:,} B)")
            for name, sz in only_b:
                print(f"   only in B: {name}  ({sz:,} B)")
            print("   A symbol present in only one target is usually an out-of-line helper")
            print("   that the other target inlined. Comparing the callers alone makes the")
            print("   inlining target look bloated; attribute the helper to its callers, or")
            print("   compare whole-module .text instead.")

    elif a.cmd == "text":
        with open(a.elf, "rb") as fh:
            elf = Elf(fh.read())
        t = elf.sec(".text")
        with open(a.out, "wb") as fh:
            fh.write(elf.d[t["off"]:t["off"] + t["size"]])
        syms = [(v - t["addr"], sz, n) for n, v, sz, ty in elf.symbols()
                if sz > 0 and not n.endswith(".kd")
                and t["addr"] <= v < t["addr"] + t["size"]]
        syms.sort()
        with open(a.syms, "w") as fh:
            for off, sz, n in syms:
                fh.write(f"{off} {sz} {n}\n")
        print(f".text vaddr=0x{t['addr']:x} size={t['size']:,} -> {a.out}")
        print(f"{len(syms)} symbols -> {a.syms}")


if __name__ == "__main__":
    main()
