# dlssnr-re — static analysis toolkit for DLSS-NR-on-AMD

Tools for reverse-engineering [danielblnc/DLSS-NR-on-AMD](https://github.com/danielblnc/DLSS-NR-on-AMD),
a `version.dll` proxy shim that reimplements NVIDIA's DLSS 5 Neural Rendering denoiser in HIP so it
runs on AMD RDNA3/RDNA4 GPUs.

This repository contains **no NVIDIA code, no NVIDIA weights, and no redistributed binaries**. Every
tool operates on files you already have on your own machine.

## What these tools established

| Question | Answer | How |
|---|---|---|
| How does the shim work? | Proxies `version.dll`, inline-detours D3D12 + FidelityFX, hijacks the game's FSR3 upscale pass | `dlssnr_pe.py imports/strings` |
| Where do the weights come from? | The `WEIGHTS_HT` resource inside the user's own `nvngx_dlssnr.dll` (147,695,410 B) | `dlssnr_pe.py resources` |
| What is the weight format? | Self-describing container; a correct walk consumes **100.000%** of the blob → 153 tensors, and every tensor's redundant length field agrees (**153/153**) | `dlssnr_weights.py check` |
| What dtype? | **Mixed**: FP8 E4M3 transformer core (blocks 15–55), FP16 conv shell (blocks 0–14, 56–70) | `dlssnr_weights.py verify` |
| What is the network? | Symmetric U-Net, dim-512 transformer stage, dim-1024 bottleneck, ~146.1M parameters | `dlssnr_weights.py topology` |
| Which GPUs are supported? | gfx1100/1101/1102 (RDNA3) and gfx1201 (RDNA4). **gfx1200 is absent** | `dlssnr_bundle.py split` |
| Is RDNA4 a first-class target? | Yes — **242** native `v_wmma_f32_16x16x16_fp8_fp8` vs **0** on RDNA3 | `dlssnr_isa.py wmma` |
| Where might time go? | 85–100% of LDS ops in hot kernels are sub-dword (2.0–3.3 B/op); 23:1 conversion-to-matrix ratio | `dlssnr_isa.py lds / ops` |

Full write-up: [`docs/FINDINGS.md`](docs/FINDINGS.md).

## Requirements

- Python 3.8+. `msgpack` is optional but recommended (register/LDS detail):
  `pip install msgpack`
- For disassembly: Windows with an AMD Adrenalin driver installed. Nothing else — the script drives
  `amd_comgr_3.dll` from `System32`.

> **Why not `llvm-objdump`?** The LLVM builds shipped with Visual Studio and the Android NDK are
> compiled *without* the AMDGPU target, so on a typical Windows box they cannot decode gfx11 or
> gfx12 at all. AMD's own code-object manager can, and it is already installed.

## Walkthrough

```bash
# 1. pull the GPU code out of the shim
python tools/dlssnr_pe.py extract version.dll --section .hip_fat --out hip_fat.bin

# 2. split it into per-GPU code objects
python tools/dlssnr_bundle.py split hip_fat.bin -o obj/

# 3. per-kernel register and LDS budget
python tools/dlssnr_bundle.py kernels obj/bundle-gfx1201.bin

# 4. compare RDNA3 against RDNA4 (flags out-of-line helpers -- see the note below)
python tools/dlssnr_bundle.py compare obj/bundle-gfx1100.bin obj/bundle-gfx1201.bin

# 5. disassemble, using AMD's own disassembler
python tools/dlssnr_bundle.py text obj/bundle-gfx1201.bin -o text.bin --syms syms.txt
powershell -ExecutionPolicy Bypass -File tools/disasm_comgr.ps1 `
    -TextBin text.bin -SymFile syms.txt `
    -Isa amdgcn-amd-amdhsa--gfx1201 -OutFile dis_gfx1201.txt

# 6. instruction-mix analysis
python tools/dlssnr_isa.py wmma  dis_gfx1201.txt
python tools/dlssnr_isa.py lds   dis_gfx1201.txt
python tools/dlssnr_isa.py loops dis_gfx1201.txt

# 7. the weights (supply your own nvngx_dlssnr.dll)
python tools/dlssnr_pe.py extract nvngx_dlssnr.dll --resource WEIGHTS_HT --out weights_ht.bin
python tools/dlssnr_weights.py check    weights_ht.bin
python tools/dlssnr_weights.py verify   weights_ht.bin
python tools/dlssnr_weights.py topology weights_ht.bin
python tools/dlssnr_weights.py export   weights_ht.bin -o npy/ --limit 10

# 8. once you have actually run the mod, turn its log into numbers
python tools/dlssnr_bench.py MyGame_dlssnr_on_amd.log
python tools/dlssnr_bench.py before.log after.log --compare
```

## The tools

| Tool | Does |
|---|---|
| `dlssnr_pe.py` | PE64 parser: sections, strings, imports, exports, resources, extraction |
| `dlssnr_bundle.py` | Splits clang offload bundles; reads AMDGPU msgpack for VGPR/SGPR/LDS/scratch/spill/arg-struct size |
| `dlssnr_weights.py` | Walks the weight container; proves the layout numerically; classifies dtype per tensor; prints the topology; exports to `.npy` |
| `disasm_comgr.ps1` | Disassembles AMDGPU code objects via `amd_comgr` P/Invoke — no toolchain install |
| `dlssnr_isa.py` | WMMA census, LDS access-width breakdown, opcode histogram, loop detection, A/B diff |
| `dlssnr_segment.py` | Finds sub-tensor boundaries *inside* a blob by sliding the dtype discriminator along it |
| `dlssnr_bench.py` | Parses the mod's runtime log; median/p99, rejects confounded windows, Mann-Whitney compare |

## Two traps these tools exist to avoid

**Asymmetric symbols.** Comparing only the *named kernels* across two targets makes the inlining
target look bloated. gfx1100 carries a shared out-of-line `swin_layer` (135,096 B) that gfx1201
inlines; counting the three callers alone suggests RDNA4 grew 3–5.8×, while whole-module accounting
shows the path **shrank 1.92×**. `dlssnr_bundle.py compare` walks all `.text` symbols and prints a
warning when a symbol exists on only one side.

**Mixed precision.** `dlssnr_weights.py verify` classifies *every* tensor rather than probing one,
because the model is not uniformly FP8. The discriminator is parity entropy, not distinct-code count:
for FP16 little-endian the low byte is near-uniform while the high byte is concentrated, so
`H(even) − H(odd)` is large; a byte-granular stream collapses it. On this model the split is clean —
largest gap among FP8 `0.040`, smallest among FP16 `0.538`, a **13.5× margin**.

## Statistical vs numerical evidence

`verify` is a *statistical* argument: it infers dtype from byte distributions. `check` is a
*numerical* one. Each tensor record ends in a redundant u32 equal to `payload_size / 2`, so a walk
whose stride is wrong by even one byte lands every subsequent tensor on garbage and the invariant
collapses. On the shipped blob it holds **153/153** while the walk consumes **147,695,410 of
147,695,410 bytes**. That is what makes the container claim a fact rather than a good guess.

(The field is a length in halfwords, not a shape. It carries no dimension information — the layer
dimensions in `docs/FINDINGS.md` come from factoring the exact byte counts instead.)

## Why there is no reimplementation here

A named blob is not a tensor. Factoring a blob's byte count into
`Cout x Cin x k x k (+ bias)` fails for **16 of the 24** distinct sizes, and the fits it does find are
meaningless divisors (`48 x 14359`). Sliding the dtype discriminator along a blob explains why:
**62 blobs change dtype partway through**, so each named blob is a *bundle* of sub-tensors whose
internal boundaries the container never stores. They live in code that was never published.

`dlssnr_segment.py` recovers some of them anyway. On `block23.layer2.layer` (917,568 B) the boundary
lands at **786,432 = 512 x 1536** — the fused QKV projection of a dim-512 transformer block —
followed by 131,136 B of FP16. That is a real structural read, and it is the method that would have
to be pushed much further before any forward pass could be written. Two adjacent sub-tensors of the
same dtype stay invisible to it, so its segment counts are lower bounds.

That is the honest state: the container, the dtypes, the topology, the kernels and the ISA are
recovered; the intra-blob partition, the skip wiring and the normalisation placement are not.

## A caveat that travels with every number here

`dlssnr_isa.py` reports **static** instruction counts. Every hot kernel contains loops
(`dlssnr_isa.py loops` will show you), so a single WMMA inside a loop can dominate runtime while
contributing 1 to a static count. Static analysis proves *which instructions exist* and *what the
resource budget is*. It does not show where time goes — profile for that.

## Legal

Not affiliated with or endorsed by NVIDIA, AMD, or the DLSS-NR-on-AMD project. DLSS is a trademark of
NVIDIA Corporation. These tools contain no NVIDIA code or data and redistribute nothing; they read
files already present on the user's machine. Interoperability and performance research on hardware
you own.
