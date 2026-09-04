#!/usr/bin/env python3
"""
dlssnr_pe.py - PE inspection for the DLSS-NR-on-AMD shim and NVIDIA's nvngx_dlssnr.dll.

No dependencies. Pure-stdlib PE64 parser.

Subcommands
-----------
  sections <pe>                  list sections
  strings  <pe> [-s SECTION]     printable strings with section+offset
  imports  <pe>                  imported DLLs and symbols
  exports  <pe>                  exported symbols
  resources <pe>                 walk the resource tree
  extract  <pe> --section NAME --out FILE
  extract  <pe> --resource NAME  --out FILE

Typical use
-----------
  python dlssnr_pe.py extract version.dll --section .hip_fat --out hip_fat.bin
  python dlssnr_pe.py extract nvngx_dlssnr.dll --resource WEIGHTS_HT --out weights_ht.bin
  python dlssnr_pe.py imports version.dll | grep -i amdhip
"""
import argparse
import re
import struct
import sys


class PE:
    def __init__(self, path):
        with open(path, "rb") as fh:
            self.d = fh.read()
        if self.d[:2] != b"MZ":
            raise ValueError(f"{path}: not a PE (no MZ)")
        self.lfanew = struct.unpack_from("<I", self.d, 0x3C)[0]
        if self.d[self.lfanew:self.lfanew + 4] != b"PE\0\0":
            raise ValueError(f"{path}: no PE signature")
        self.coff = self.lfanew + 4
        self.nsec, = struct.unpack_from("<H", self.d, self.coff + 2)
        self.optsz, = struct.unpack_from("<H", self.d, self.coff + 16)
        magic, = struct.unpack_from("<H", self.d, self.coff + 20)
        if magic != 0x20B:
            raise ValueError("only PE32+ (x64) is supported")
        self.datadir = self.coff + 20 + 112
        self.sections = []
        sh = self.coff + 20 + self.optsz
        for i in range(self.nsec):
            o = sh + i * 40
            name = self.d[o:o + 8].rstrip(b"\0").decode("latin1")
            vsz, va, rsz, ptr = struct.unpack_from("<IIII", self.d, o + 8)
            chars, = struct.unpack_from("<I", self.d, o + 36)
            self.sections.append(dict(name=name, vsize=vsz, rva=va,
                                      rawsize=rsz, rawptr=ptr, flags=chars))

    def dir_entry(self, index):
        return struct.unpack_from("<II", self.d, self.datadir + index * 8)

    def rva_to_off(self, rva):
        for s in self.sections:
            span = max(s["vsize"], s["rawsize"])
            if s["rva"] <= rva < s["rva"] + span:
                return s["rawptr"] + (rva - s["rva"])
        return None

    def cstr(self, off):
        end = self.d.index(b"\0", off)
        return self.d[off:end].decode("latin1")

    def section(self, name):
        for s in self.sections:
            if s["name"] == name:
                return s
        return None

    def section_bytes(self, name):
        s = self.section(name)
        if not s:
            raise KeyError(f"no section named {name!r}")
        return self.d[s["rawptr"]:s["rawptr"] + s["rawsize"]]

    # ---- imports -------------------------------------------------------
    def imports(self):
        rva, size = self.dir_entry(1)
        if not rva:
            return {}
        base = self.rva_to_off(rva)
        out = {}
        i = 0
        while True:
            o = base + i * 20
            ilt, ts, fwd, name_rva, iat = struct.unpack_from("<IIIII", self.d, o)
            if not name_rva:
                break
            dll = self.cstr(self.rva_to_off(name_rva))
            syms = []
            thunk_rva = ilt or iat
            t = self.rva_to_off(thunk_rva)
            while True:
                val, = struct.unpack_from("<Q", self.d, t)
                if val == 0:
                    break
                if val & (1 << 63):
                    syms.append(f"#{val & 0xFFFF}")
                else:
                    ho = self.rva_to_off(val & 0x7FFFFFFF)
                    syms.append(self.cstr(ho + 2))
                t += 8
            out[dll] = syms
            i += 1
        return out

    # ---- exports -------------------------------------------------------
    def exports(self):
        rva, size = self.dir_entry(0)
        if not rva:
            return []
        b = self.rva_to_off(rva)
        (_, _, _, _, ordbase, naddr, nnames,
         addr_rva, names_rva, ord_rva) = struct.unpack_from("<IIHHIIIIII", self.d, b)
        names_off = self.rva_to_off(names_rva)
        ord_off = self.rva_to_off(ord_rva)
        addr_off = self.rva_to_off(addr_rva)
        out = []
        for i in range(nnames):
            nrva, = struct.unpack_from("<I", self.d, names_off + i * 4)
            name = self.cstr(self.rva_to_off(nrva))
            idx, = struct.unpack_from("<H", self.d, ord_off + i * 2)
            fn, = struct.unpack_from("<I", self.d, addr_off + idx * 4)
            out.append((idx + ordbase, fn, name))
        return out

    # ---- resources -----------------------------------------------------
    def resources(self):
        rva, size = self.dir_entry(2)
        if not rva:
            return []
        base = self.rva_to_off(rva)
        found = []

        def walk(off, path):
            _, _, _, _, nnamed, nid = struct.unpack_from("<IIHHHH", self.d, off)
            for i in range(nnamed + nid):
                eo = off + 16 + i * 8
                nid_, off2 = struct.unpack_from("<II", self.d, eo)
                if nid_ & 0x80000000:
                    so = base + (nid_ & 0x7FFFFFFF)
                    ln, = struct.unpack_from("<H", self.d, so)
                    nm = self.d[so + 2:so + 2 + ln * 2].decode("utf-16le")
                else:
                    nm = str(nid_)
                if off2 & 0x80000000:
                    walk(base + (off2 & 0x7FFFFFFF), path + [nm])
                else:
                    drva, dsz, _, _ = struct.unpack_from("<IIII", self.d, base + off2)
                    found.append((path + [nm], drva, dsz))

        walk(base, [])
        return found


def strings_in(buf, minlen=5):
    for m in re.finditer(rb"[\x20-\x7e]{%d,}" % minlen, buf):
        yield m.start(), m.group().decode("latin1")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sections", "strings", "imports",
                                    "exports", "resources", "extract"])
    ap.add_argument("pe")
    ap.add_argument("-s", "--section")
    ap.add_argument("-r", "--resource")
    ap.add_argument("-o", "--out")
    ap.add_argument("-n", "--minlen", type=int, default=6)
    a = ap.parse_args()
    pe = PE(a.pe)

    if a.cmd == "sections":
        print(f"{'NAME':10} {'VSIZE':>10} {'RVA':>10} {'RAWSIZE':>12} {'RAWPTR':>12} FLAGS")
        for s in pe.sections:
            print(f"{s['name']:10} {s['vsize']:10x} {s['rva']:10x} "
                  f"{s['rawsize']:12x} {s['rawptr']:12x} {s['flags']:08x}")

    elif a.cmd == "strings":
        for s in pe.sections:
            if a.section and s["name"] != a.section:
                continue
            buf = pe.d[s["rawptr"]:s["rawptr"] + s["rawsize"]]
            for off, text in strings_in(buf, a.minlen):
                print(f"{s['name']}+{off:08x} {text}")

    elif a.cmd == "imports":
        for dll, syms in pe.imports().items():
            print(f"{dll}  ({len(syms)} symbols)")
            for s in syms:
                print(f"    {s}")

    elif a.cmd == "exports":
        for ordinal, fn, name in pe.exports():
            print(f"  {ordinal:5} {fn:08x} {name}")

    elif a.cmd == "resources":
        ents = pe.resources()
        print(f"{len(ents)} resource entries")
        for path, rva, sz in sorted(ents, key=lambda e: -e[2]):
            print(f"  {'/'.join(path):40} rva={rva:08x} size={sz:>12,} "
                  f"({sz / 1048576:.2f} MB)")

    elif a.cmd == "extract":
        if not a.out:
            sys.exit("--out is required for extract")
        if a.section:
            data = pe.section_bytes(a.section)
            s = pe.section(a.section)
            data = data[:s["vsize"]] if s["vsize"] < len(data) else data
        elif a.resource:
            hit = [e for e in pe.resources() if a.resource in e[0]]
            if not hit:
                sys.exit(f"no resource matching {a.resource!r}")
            _, rva, sz = hit[0]
            off = pe.rva_to_off(rva)
            data = pe.d[off:off + sz]
        else:
            sys.exit("give --section or --resource")
        with open(a.out, "wb") as fh:
            fh.write(data)
        print(f"wrote {len(data):,} bytes -> {a.out}")


if __name__ == "__main__":
    main()
