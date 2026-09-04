#!/usr/bin/env python3
"""
dlssnr_isa.py - statistics over AMDGPU disassembly produced by disasm_comgr.ps1.

Input is a text file of the form:

    === KERNEL <mangled> off=<n> size=<n> ===
    <tab>s_load_b32 s12, s[2:3], 0x34
    ...

Subcommands
-----------
  wmma   <dis>            matrix-instruction census, per kernel and total
  lds    <dis>            LDS access-width breakdown (the sub-dword problem)
  ops    <dis> [-k K]     opcode histogram, whole file or one kernel
  loops  <dis>            branch / backward-branch counts (loop proxy)
  diff   <disA> <disB>    per-kernel instruction-count comparison

Caveat carried by every subcommand: these are STATIC counts. A WMMA inside a
hot loop can dominate runtime while contributing 1 here. Static analysis proves
which instructions exist and what the resource budget is; it does not show
where time goes. Use a profiler for that.
"""
import argparse
import collections
import re

KERNEL_RE = re.compile(r"=== KERNEL (\S+) off=(\d+) size=(\d+)")
MATRIX_RE = re.compile(r"^(v_wmma\w*|v_swmmac\w*|v_dot\w*)$")
WIDTH_RE = re.compile(r"_(u8|i8|b8|u16|i16|b16|b32|b64|b96|b128)(?:_|$)")
WIDTH_BYTES = {"u8": 1, "i8": 1, "b8": 1, "u16": 2, "i16": 2, "b16": 2,
               "b32": 4, "b64": 8, "b96": 12, "b128": 16}
NARROW = {"u8", "i8", "b8", "u16", "i16", "b16"}


def load(path):
    """OrderedDict: mangled kernel name -> list of instruction texts."""
    per = collections.OrderedDict()
    cur = None
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            m = KERNEL_RE.match(s)
            if m:
                cur = m.group(1)
                per[cur] = []
                continue
            if s and cur is not None and re.match(r"^[a-z]", s):
                per[cur].append(s)
    return per


def short(name):
    return re.sub(r"^_Z\d+", "", name).replace("Params", "P")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("wmma", "lds", "loops"):
        p = sub.add_parser(c); p.add_argument("dis")
    p = sub.add_parser("ops"); p.add_argument("dis")
    p.add_argument("-k", "--kernel"); p.add_argument("-n", "--top", type=int, default=32)
    p = sub.add_parser("diff"); p.add_argument("a"); p.add_argument("b")
    a = ap.parse_args()

    if a.cmd == "diff":
        A, B = load(a.a), load(a.b)
        print(f"{'KERNEL':46}{'A insn':>10}{'B insn':>10}{'ratio':>9}")
        for k in sorted(set(A) | set(B), key=lambda n: -len(A.get(n, []))):
            na, nb = len(A.get(k, [])), len(B.get(k, []))
            ratio = f"{nb / na:.2f}x" if na else "n/a"
            mark = "  (A only)" if k not in B else ("  (B only)" if k not in A else "")
            print(f"{short(k)[:46]:46}{na:>10,}{nb:>10,}{ratio:>9}{mark}")
        ta = sum(len(v) for v in A.values()); tb = sum(len(v) for v in B.values())
        print("-" * 75)
        print(f"{'TOTAL':46}{ta:>10,}{tb:>10,}"
              f"{(f'{tb / ta:.3f}x' if ta else 'n/a'):>9}")
        return

    per = load(a.dis)

    if a.cmd == "wmma":
        total = collections.Counter()
        print(f"{'KERNEL':46}{'insn':>8}{'fp8 WMMA':>10}{'f16 WMMA':>10}{'other mtx':>11}")
        for k, ins in per.items():
            c = collections.Counter(i.split()[0] for i in ins)
            fp8 = sum(v for o, v in c.items() if MATRIX_RE.match(o) and "fp8" in o)
            f16 = sum(v for o, v in c.items()
                      if MATRIX_RE.match(o) and "f16" in o and "fp8" not in o)
            oth = sum(v for o, v in c.items() if MATRIX_RE.match(o)) - fp8 - f16
            for o, v in c.items():
                if MATRIX_RE.match(o):
                    total[o] += v
            print(f"{short(k)[:46]:46}{len(ins):>8,}{fp8:>10}{f16:>10}{oth:>11}")
        print("\n=== matrix instruction totals ===")
        for o, v in total.most_common():
            print(f"  {o:44}{v:>8}")
        if not total:
            print("  (none - this target has no matrix instructions in this build)")

    elif a.cmd == "lds":
        print(f"{'KERNEL':40}{'ds ops':>8}{'narrow':>8}{'b32':>7}{'b64':>7}"
              f"{'b128':>7}{'B/op':>7}{'narrow%':>9}")
        for k, ins in per.items():
            ds = [i for i in ins if i.startswith("ds_")]
            if not ds:
                continue
            w = collections.Counter()
            total_bytes = 0
            for i in ds:
                m = WIDTH_RE.search(i.split()[0])
                key = m.group(1) if m else "?"
                w[key] += 1
                total_bytes += WIDTH_BYTES.get(key, 4)
            narrow = sum(v for kk, v in w.items() if kk in NARROW)
            print(f"{short(k)[:40]:40}{len(ds):>8}{narrow:>8}{w.get('b32', 0):>7}"
                  f"{w.get('b64', 0):>7}{w.get('b128', 0):>7}"
                  f"{total_bytes / len(ds):>7.1f}{100 * narrow / len(ds):>8.0f}%")
        print("\nRDNA banks LDS in 32-bit words: a 16-bit access still costs a full bank\n"
              "cycle, so sub-dword traffic caps near 50% of peak LDS bandwidth (25% for\n"
              "bytes). A high narrow% with a low B/op is the signature of unpacking FP8\n"
              "tiles one or two bytes at a time.")

    elif a.cmd == "loops":
        print(f"{'KERNEL':46}{'insn':>8}{'branches':>10}{'backward':>10}")
        for k, ins in per.items():
            br = [i for i in ins if re.match(r"s_(cbranch\w*|branch)\b", i)]
            back = 0
            for i in br:
                m = re.search(r"\s(\d+)$", i)
                if m and int(m.group(1)) >= 32768:   # 16-bit signed, negative
                    back += 1
            print(f"{short(k)[:46]:46}{len(ins):>8,}{len(br):>10}{back:>10}")
        print("\nBackward branches are loops. Any kernel with them makes static counts a\n"
              "poor proxy for dynamic cost - read the other subcommands with that in mind.")

    elif a.cmd == "ops":
        c = collections.Counter()
        for k, ins in per.items():
            if a.kernel and a.kernel not in k:
                continue
            c.update(i.split()[0] for i in ins)
        conv = sum(v for o, v in c.items() if o.startswith("v_cvt"))
        mtx = sum(v for o, v in c.items() if MATRIX_RE.match(o))
        print(f"top {a.top} opcodes ({sum(c.values()):,} instructions total)")
        for o, v in c.most_common(a.top):
            print(f"  {o:40}{v:>8,}")
        print(f"\n  format-conversion ops : {conv:,}")
        print(f"  matrix ops            : {mtx:,}")
        if mtx:
            print(f"  conversion : matrix   = {conv / mtx:.1f} : 1")


if __name__ == "__main__":
    main()
