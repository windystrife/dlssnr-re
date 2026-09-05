# DLSS-NR-on-AMD — static analysis findings

Analysed build: `version.dll`, 4,549,632 B, sha256 `26fe30c10c719f491acb75002bed0e84c1e8793fe93ca1590e1cd8d4144a510e`,
PE timestamp 2026-09-04 (release v0.2.10). Weights from `nvngx_dlssnr.dll` 165,840,496 B, sha256
`e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e` (DLSS NR 310.8.0.0).

Everything below is reproducible with the tools in this repository.

---

## 1. Mechanism

`version.dll` exports the 17 genuine `version.dll` APIs so Windows preloads it, then inline-detours:

- `D3D12CreateDevice`
- `IDXGIFactory::CreateSwapChain`, `IDXGIFactory2::CreateSwapChainForHwnd`
- `IDXGISwapChain::Present`, `IDXGISwapChain1::Present1`
- `ID3D12CommandQueue::ExecuteCommandLists`
- FidelityFX: `ffxCreateContext`, `ffxDispatch`, `ffxFsr3ContextCreate`,
  `ffxFsr3UpscalerContextCreate`, `ffxFsr3ContextDispatchUpscale`, `ffxFsr3UpscalerContextDispatch`

across `ffx_fsr3upscaler_x64.dll`, `ffx_fsr3_x64.dll`, `amd_fidelityfx_dx12.dll`,
`amd_fidelityfx_upscaler_dx12.dll` and `amd_fidelityfx_loader_dx12.dll`.

It implements no NVIDIA API. It rides the game's FSR3 upscale pass, borrowing the colour, motion
vector, depth and exposure inputs FSR already prepared. Hence the user-facing string:

> `FSR is not active: enable FSR (any quality mode) in the game's graphics settings to use DLSS-NR.`

The output is applied as an **RGB residual**, not a replacement image. The runtime-compiled shader is:

```hlsl
Texture2D<float4> res : register(t0); RWTexture2D<float4> outp : register(u0);
cbuffer C : register(b0) { uint tone; uint w; uint h; float expo; }
[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= w || id.y >= h) return;
    float4 c = outp[id.xy]; float3 d = res[id.xy].rgb; float3 e;
    if (tone) { float3 s = c.rgb * expo;
                e = float3(tmf(s.r), tmf(s.g), tmf(s.b)) + d;
                e = float3(tmi(e.r), tmi(e.g), tmi(e.b)) / expo; }
    else e = saturate(c.rgb + d);
    outp[id.xy] = float4(e, c.a);
}
```

The `tone` branch costs six `pow()` per pixel at output resolution.

### Host-side imports that tell the story

- **`amdhip64_7.dll`** (29 symbols): `__hipRegisterFatBinary/Function`, `hipLaunchKernel`,
  `hipMalloc/Free`, `hipMemcpy(Async)`, `hipEventCreate/Record/ElapsedTime`,
  `hipGetDevicePropertiesR0600`, and crucially
  `hipImportExternalMemory` / `hipExternalMemoryGetMappedBuffer` / `hipDestroyExternalMemory`
  for D3D12↔HIP zero-copy.
- No `hipStream*` and no `hipGraph*` symbol exists anywhere in the image — every dispatch goes to the
  null stream, serially.
- `D3DCOMPILER_47!D3DCompile` — the apply and flag shaders are compiled at runtime.
- `bcrypt` — the shim hard-codes the known-good sha256 of `nvngx_dlssnr.dll` as a literal at
  `.rdata+0xf429`.
- `VirtualProtect`, `FlushInstructionCache`, `AddVectoredExceptionHandler`, `Get/SetThreadContext` —
  inline detour machinery.

Cross-API synchronisation is a `globallycoherent RWByteAddressBuffer` polled by a
`[numthreads(1,1,1)]` D3D12 compute shader against HIP-side `k_flag_set` / `k_flag_wait` spin
kernels, with an adaptive iteration cap.

---

## 2. The weight container

Recovered by exact-byte accounting. A walk using this layout consumes **147,695,410 of 147,695,410
bytes** — that exact landing is the proof the layout is right.

```
u64  blob_total            # equals the resource size
repeat 153×:
  u64  name_len
  char name[name_len]      # e.g. "block31.layer1.layer"
  u64  a                   # == b, == c + 40
  u64  b
  u64  c                   # payload bytes
  u32  flag                # 1
  u8   data[c]
  u64  pad                 # 0
  u64  one                 # 1
  u32  shape_hint          # two u16 fields
```

153 tensors, 147,683,778 payload bytes. The shim validates every tensor against an expected table
(`all %zu tensors match DLSS NR 310.8.0.0 exactly; using it`) and transcodes to a
`dlssnr_on_amd_weights.bin` cache with magic `DLSSNRW1`.

### Dtype: mixed precision

| Class | Tensors | Bytes | Parameters | Blocks |
|---|---:|---:|---:|---|
| FP8 E4M3 | 114 | 144,528,224 | 144,528,224 | **15–55** — the whole transformer + bottleneck core |
| FP16 | 30 | 3,155,536 | 1,577,768 | **0–14** and **56–70** — the outer conv shell |
| scalars | 9 | 18 | 9 | per-block LayerScale |

**True parameter count: 146,106,001.** Counting every byte as one parameter over-counts the FP16
shell by 2×.

Method: parity entropy. For FP16 little-endian the low byte (even index) is near-uniform while the
high byte (odd index) is concentrated, so `H(even) − H(odd)` is large; a byte-granular stream
collapses the gap. Largest gap among the FP8 group `0.040`, smallest among FP16 `0.538` — a **13.5×
margin**, so this is a clean split rather than a tuned threshold. Distinct-code count is *not* a
reliable discriminator here; it overlaps badly in the middle.

The FP16 ranges are contiguous and perfectly mirrored (0–14 ↔ 56–70): precision is kept where the
image data enters and leaves, and spent down through the core. That is exactly where the native FP8
matrix instructions get used.

Decoded as E4M3, `block31.layer1.layer` gives mean −0.00002, |max| 0.0859, zero NaNs.

---

## 3. Topology

| Blocks | Tensors each | layer0 bytes | Role |
|---|---:|---:|---|
| 0–4 | 1 | 20,672 – 22,720 | conv stem |
| 5–8 | 1 | 61,760 – 69,936 | encoder level 1 |
| 9–14 | 1 | 197,184 – 229,936 | encoder level 2 |
| 15–22 | 1 | 689,232 – 820,288 | encoder level 3 |
| **23–30** | **4** | 524,288 | **transformer, dim 512** |
| **31–38** | **5** | 4,194,320 | **bottleneck, dim 1024** |
| 39 | 1 | 525,312 | up-projection |
| 40–47 | 4 | 524,288 | decoder transformer, dim 512 |
| 48–55 | 1 | 689,232 – 820,784 | decoder level 3 |
| 56–61 | 1 | 197,184 – 230,176 | decoder level 2 |
| 62–69 | 1 | 20,672 – 70,048 | decoder levels 1 and 0 |
| 70 | 2 | 21,808 | output head + `blend_scale` |

Dimensions read off exact byte counts:

- dim-512 blocks: `layer0` = 524,288 = **512 × 1024**; `layer1` = 263,168 = **512 × 512 + 512 FP16
  biases** (262,144 + 1,024 exactly); `layer2` = 917,568 (the FFN); `layer3` = 263,168
- bottleneck: `layer0` = **1024 × 4096** + 16; `layer1` = **1024 × 4096 + 1024 FP16 biases**;
  `layer2` = **1024 × 3072** + 128; `layer3` = a 2-byte FP16 LayerScale; `layer4` = **1024 × 1024 +
  1024 FP16 biases**

Model-name strings in the binary: `VIT512_OLD`, `vit512a`, `vit512b`, `vit512_attn`, `vit512_ffwd`,
`vit512_conv1`, `vit512_conv2`, `vit1d`.

---

## 4. GPU code

`.hip_fat` is a 4,052,232 B clang offload bundle with five entries:

| Triple | Size |
|---|---:|
| `host-x86_64-unknown-linux-gnu-` | 0 |
| `hipv4-amdgcn-amd-amdhsa--gfx1100` | 1,179,328 |
| `hipv4-amdgcn-amd-amdhsa--gfx1101` | 1,179,328 |
| `hipv4-amdgcn-amd-amdhsa--gfx1102` | 1,181,048 |
| `hipv4-amdgcn-amd-amdhsa--gfx1201` | 505,096 |

`gfx1200` is **absent** — RX 9060-class parts are unsupported. The binary says so:

> `this build contains code for RDNA4 (gfx12) and RDNA3 (gfx11) only; the pass stays off`

### The 33 kernels

`k_import`, `k_export`, `k_repack`, `k_mean`, `k_flag_set`, `k_flag_wait`, `k_reproject`,
`k_conv_res`, `k_conv_res2`, `k_conv_res_views`, `k_conv_splitk`, `k_contract2`,
`k_qkv`, `k_qkv2`, `k_qkv_attn`, `k_qkv_attn2`, `k_attention`, `k_attention2`,
`k_ffwd`, `k_ffwd2`, `k_ffwd_inpview`,
`k_swin_var<32|64|128|256, bool>`, `k_expand`, `k_expand2`, `k_dec_upsample`, `k_final_head`,
`k_swin_1h_32_fp8`, `k_pre_block_1h_32_fp8`, `k_post_block_1h_32_fp8`.

The pairing (`k_qkv`/`k_qkv2`, `k_ffwd`/`k_ffwd2`, `k_attention`/`k_attention2`) corresponds to the
two transformer widths, 512 and 1024.

### Matrix instruction census

| Instruction | gfx1100 | gfx1201 |
|---|---:|---:|
| `v_wmma_f32_16x16x16_fp8_fp8` | 0 | **242** |
| `v_wmma_f32_16x16x16_f16` | 203 | 66 |
| `v_dot2_f32_f16` | 6 | 5 |
| any FP8 instruction | **0** | 2,441 |
| `v_swmmac_*` (2:4 sparsity) | 0 | 0 |

Module `.text`: 1,112,668 B → 438,504 B = **0.394×**. Per kernel: `k_qkv` 0.08×, `k_attention`
0.16×, `k_qkv_attn` 0.19×. RDNA3 has to emulate the FP8 math with f16 plus heavy unrolling.

**The `_fp8`-named kernels are not the FP8 path.** `k_swin_1h_32_fp8`, `k_pre_block_1h_32_fp8` and
`k_post_block_1h_32_fp8` contain **zero** FP8 WMMA and exactly 14 `v_wmma_f32_16x16x16_f16` each on
gfx1201. "fp8" names the storage format staged in LDS. The 242 native FP8 WMMA live in the
generically-named kernels: `k_swin_var<64>` 19 sites, `<128>`/`<256>` 16 each, `<32,*>` 13 each,
`k_qkv2` 8, `k_qkv_attn2` 6, `k_ffwd2` 6, `k_conv_res2`/`k_expand2`/`k_contract2` 4 each.

**Watch the inlining artifact.** gfx1100 carries a 34th symbol gfx1201 lacks:
`_Z10swin_layerR7SwinLDSPKhRK10BlobLayouti`, a shared out-of-line function of 135,096 B holding the
same 14 f16 WMMA, which gfx1201 fully inlines. Symbol ranges are disjoint. Counting only the three
named kernels suggests RDNA4 grew 3–5.8×; honest accounting (21,112 + 135,096 = 156,208 B vs
81,552 B) shows it **shrank 1.92×**. gfx1100 also pays a real call
(`.private_segment_fixed_size = 64`, call-ABI `.vgpr_count = 194`) where gfx1201 pays 0 and 70–72.

### Resource budget (gfx1201)

wave32; `max_flat_workgroup_size` 256 for compute kernels. Zero VGPR spills on every kernel, both
targets.

| Kernel | VGPR | SGPR | LDS | % of 65,536 |
|---|---:|---:|---:|---:|
| `k_pre_block_1h_32_fp8` | 72 | 44 | 64,640 | **98.6%** |
| `k_swin_1h_32_fp8` | 72 | 25 | 62,592 | **95.5%** |
| `k_post_block_1h_32_fp8` | 70 | 43 | 62,592 | **95.5%** |
| `k_qkv_attn` | 94 | 30 | 58,368 | **89.1%** |
| `k_attention` | 79 | 32 | 28,928 | 44.1% |
| `k_ffwd`, `k_ffwd_inpview`, `k_conv_res_views` | 60–90 | 19–49 | 24,576 | 37.5% |
| `k_conv_res2` | 203 | 107 | 0 | — |
| `k_contract2` | 170 | 55 | 16,384 | 25.0% |

The top four sit at 89–99% of the per-workgroup LDS ceiling, pinning them to 2 workgroups/WGP =
4 of 16 waves per SIMD (25%), while using only 70–72 of the 96 VGPRs available for free. LDS
allocation is **byte-identical on gfx1100 and gfx1201 for all 33 kernels** (574,608 B on both), so
the tiling was never tuned for either target.

---

## 5. Instruction mix (gfx1201, static)

Top opcodes: `s_delay_alu` 9,791 · `s_wait_alu` 6,329 · `s_mov_b32` 2,754 · `s_or_b32` 2,632 ·
`v_mov_b32` 2,458 · `v_cvt_f32_f16` 2,373 · `v_and_b32` 2,253 · `v_cvt_f16_f32` 2,201 ·
`v_cvt_pk_fp8_f32` 1,489 · `v_maxmin_num_f32` 1,489 · `v_fma_mix_f32` 1,376 · `ds_load_u16` 934 ·
`v_cvt_f32_fp8` 936 · `ds_store_b16` 921.

**Format conversion : matrix ops = 23 : 1** (7,197 vs 313).

### LDS access width

| Kernel | ds ops | sub-dword | avg B/op |
|---|---:|---:|---:|
| `k_contract2` | 65 | **100%** | 2.0 |
| `k_qkv2` | 126 | **95%** | 2.2 |
| `k_swin_1h_32_fp8` | 308 | **90%** | 3.0 |
| `k_pre_block_1h_32_fp8` | 316 | 89% | 3.2 |
| `k_post_block_1h_32_fp8` | 311 | 89% | 3.1 |
| `k_qkv_attn` | 211 | 88% | 3.3 |
| `k_swin_var<64>` | 212 | 86% | 3.1 |

Against the 16 bytes a `ds_load_b128` moves. RDNA banks LDS in 32-bit words, so a 16-bit access still
consumes a full bank cycle: sub-dword traffic caps near 50% of peak LDS bandwidth, 25% for bytes.
The combination of that with the 23:1 conversion ratio suggests the RDNA4 path spends much of its
instruction budget unpacking FP8 tiles one or two bytes at a time.

**This is a hypothesis, not a measurement.** These are static counts and every hot kernel loops
(backward branches: `k_swin_var<32>` 61–64, `k_contract2` 39, `k_post_block_1h_32_fp8` 39,
`k_pre_block` 33, `k_swin_1h_32_fp8` 28). Only a profile shows where time goes.

---

## 6. Unresolved

- **Inter-block dataflow.** The mirrored width schedule is proven; the U-Net *skip* wiring,
  normalisation placement and window-shift schedule are not.
- **The bool in `k_swin_var<T,bool>`.** The "shifted window" reading is refuted — both variants
  already contain a runtime window-base selector driven by a flags word at kernarg `0x28`.
- **What `Scale` (default 0.03125) multiplies.** It is stored to `.data 0x74508` and no instruction
  in the 4.55 MB binary reads that address. Static analysis cannot prove it reaches the GPU.
- **Semantics of most INI keys and all six `DLSSNR_*` env vars.** Undocumented upstream.


---

## 7. Sub-tensor recovery: 80 of 153 blobs resolved

A named blob is a *bundle*, not a tensor — the container stores only the bundle's total length.
`dlssnr_segment.py bundle` recovers the internal partition where it can, by exploiting the fact that
blobs of the same byte size are the same layer type repeated across blocks: averaging the **signed**
parity difference across all copies makes noise fall as sqrt(n) while the FP16 signal stays, which
buys a much smaller window — and the window is the boundary resolution.

| blob size | copies | window | recovered layout | reading |
|---:|---:|---:|---|---|
| 4,196,352 | 8 | 1 KiB | FP8 4,194,304 + FP16 2,048 | **1024x4096 + 1024 biases** |
| 4,194,320 | 8 | 1 KiB | FP8 uniform (+16 B) | **1024x4096** |
| 3,145,856 | 8 | 1 KiB | FP8 uniform (+128 B) | **1024x3072** |
| 1,050,624 | 8 | 1 KiB | FP8 1,048,576 + FP16 2,048 | **1024x1024 + 1024 biases** |
| 917,568 | 16 | 512 B | FP8 786,432 + FP16 131,136 | **512x1536 fused QKV** + an FP16 bundle |
| 524,288 | 16 | 512 B | FP8 uniform | **512x1024** |
| 263,168 | 32 | 256 B | FP8 262,144 + FP16 1,024 | **512x512 + 512 biases** |

Every boundary above lands inside its detection window. 512x512 + 512 FP16 biases accounts for
263,168 exactly; 1024x4096 + 1024 FP16 biases accounts for 4,196,352 exactly.

**The byte count is what the data determines; the a x b split is not.** 1024x4096 and 512x8192 have
the same byte count. The factorisation above comes from the tier dimension (512 for blocks 23–30 and
40–47, 1024 for the bottleneck 31–38), which the byte counts then confirm.

### What did not resolve, and why

- **The conv bundles.** `689,232` (14 copies) and `197,184` (10 copies) resolve into a consistent
  **six-segment** alternating FP8/FP16 layout, but no boundary lands on a round value. These are
  deeper bundles than the transformer blocks — likely conv weights plus per-channel norm parameters.
- **Single-copy blobs** (the transition blocks at 525,312 / 524,304 / 230,176 / 229,936 / 70,048 /
  69,936 / 22,784 / 22,720 / 21,808 / 21,696 / 820,784 / 820,288). With one copy there is nothing to
  average, so the window is forced to 8 KiB — far too coarse to place a boundary.
- **Same-dtype neighbours are invisible.** The method finds dtype changes, not tensor changes, so
  every segment count is a lower bound. The 131,136 B FP16 region inside `917,568` is almost
  certainly several tensors, not one.

### Where that leaves a reimplementation

The transformer stage and the bottleneck are now dimensioned. The conv shell is not, and the graph
that connects them — skip wiring, normalisation placement, window-shift schedule — is not in the
weights at all. Recovering those means reading the kernel argument structs (sizes already known:
`VarParams` 168 B, `ReprojParams` 128 B, `PreParams`/`PostParams` 80 B, `ConvPlParams` 72 B,
`Conv2Params`/`ExportParams` 64 B, down to 24 B) against the disassembly. That is days of work, not
an afternoon — and it still needs an oracle to verify against.


---

## 8. The whole blob is one block schema at six widths

Section 7 stopped at 80/153. Four further routes - kernel-arg structs, sharper segmentation,
architectural constraint, and mining immediates out of the ISA - closed the rest.

### The schema

Every blob is the same block instantiated at one of six widths `C` in {32, 64, 128, 256, 512, 1024}:

| part | bytes | dtype | notes |
|---|---|---|---|
| `A0` | 4C2 | FP8 | row length **128** at C=32, ISA-proven from `idx >> 7` |
| `A1` | 128C | FP8 | absent at C >= 512; role unknown |
| `A2` | C2 | FP8 | **absent at C=32** |
| `g1` | 2C | FP16 | gain vector, in a 16-byte-padded slot |
| `Q` | 3C2 | FP8 | row length **3C** at C=32, ISA-proven from the magic divide `0xaaab >> 22` = /96 |
| `T` | 256C | FP16 | all values <= 0, max exactly 0.0; role unknown; 26-40% of a narrow block |
| `S` | 4*(C/32) | FP32 | consumer unidentified |
| `P` | C2 | FP8 | |
| `g2` | 2C | FP16 | |

Closed form: **`R(C) = 9C^2 + 388C + 48 + max(16, C/8)`** for C >= 64, and
**`R(32) = 8C^2 + 388C + 48 + 16`** because the `A2` slab is genuinely absent at the narrowest
width. Downsample blocks append one `2C^2` FP8 tensor; upsample blocks prepend one.

### The falsification test

`dlssnr_model.py check` builds every blob size from the schema and compares:

```
blobs   predicted  153   observed  153
bytes   predicted  147,683,778   observed  147,683,778   residual 0
```

Every one of the 24 distinct sizes matches in count and in bytes. A wrong schema does not survive
this - the four narrow widths are constrained by four independent byte counts and one formula.

### "Conv shell" was the wrong name

**No FP8 slab in the model is divisible by 9**, and none carries any k^2 factor
(262,144 % 9 = 1, 32,768 % 9 = 8, 65,536 % 9 = 7, 196,608 % 9 = 3). There is no 3x3 convolution
anywhere. The outer tiers run the *same* block schema as the dim-512 transformer, just narrower. The
name came from the `k_conv_*` kernel symbols and is misleading.

Width ladder, closed at both ends, each rung an exact `2C^2` FP8 tensor:
`stem -> 32 -> 64 -> 128 -> 256 -> 512 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 4 out`.
Independent support: `k_swin_var` exists in exactly five instantiations - `<32,true>`, `<32,false>`,
`<64,false>`, `<128,false>`, `<256,false>` - and there is no `<512>`.

### A premise that turned out to be false

The obvious route was "count pointers in each kernel's argument struct; that is the number of
sub-tensors". It is wrong. **Every conv/attention kernel takes exactly ONE weight pointer** and
derives all sub-tensor bases from it by compile-time displacement. `k_swin_1h_32_fp8` loads one
64-bit value from kernarg `0x10` and reaches all seven of its sub-tensors by `s_add_nc_u64`
immediates: `+0`, `+0x1000`, `+0x2060`, `+0x4c70`, `+0x2c60`, plus two `global_load_u16` at
`offset:8208` / `offset:20592`. The other pointers in a struct are activation in/out/residual
buffers.

Those displacements are the strongest corroboration in this analysis: **every sub-tensor base
measured from the weight bytes also appears as a literal immediate in the RDNA4 disassembly, and
every padding offset is absent** because padding is never addressed. Two fully independent sources
agree to the byte on 7/7 bases in `k_swin_1h_32_fp8`, 7/7 in `k_pre_block`, 9/9 in `k_post_block`.

### Class totals

143,831,040 B of FP8 weights (97.4% of payload), 1,923,753 FP16 values, 714 FP32 values,
2,376 B of zero padding.

### Still missing for a forward pass

Byte extents and dtypes are settled. What is missing is shape and semantics:

1. **Factorisation of nearly every FP8 slab.** Only `A0` and `Q` have an ISA-proven row stride, and
   only at C=32. `262,144 B` is equally `1024x256`, `512x512` or `4x(256x256)` and nothing in the
   container picks one.
2. **What `A1` is** - 128C FP8 in every narrow block, absent at 512 and 1024.
3. **What `T` is** - 128C FP16 values, all <= 0 with max exactly 0.0, added pre-exponential into an
   attention-score accumulator. A quarter to two-fifths of a narrow block's bytes, unexplained.
4. **Runtime dimensions of `k_swin_var`** - window size, token blocking, shift schedule, per-launch
   channel counts. All arrive from the host in a 168-byte `VarParams`, so the C = 64/128/256 tiers
   have no ISA-visible dimensions at all.
5. **Kernel-to-blob binding.** No symbol ties any kernel to any blob; every binding is inference from
   constant coincidence.

### Highest-value next step

Recover `VarParams` (168 B) from `version.dll` itself - the host-side launch descriptor for
`k_swin_var`. It carries, as literal host constants, everything item 4 is missing and most of items 1
and 5, and it would put the wide tiers on the same ISA-grade footing the C=32 tier already has. It is
purely static and purely local. Anchor on the `k_swin_var` mangled-name strings and the
`hipModuleGetFunction` / launch call sites, then read the stores into the 168-byte argument buffer.


---

## 9. The transcode is a pure re-index (open item 8, closed)

Every displacement recovered from the GPU disassembly addresses the buffer the mod builds at first
run, `dlssnr_on_amd_weights.bin`, not NVIDIA's `WEIGHTS_HT` resource. Section 8 cross-checked those
displacements against `WEIGHTS_HT` **on the assumption** that the transcode is a straight copy, and
flagged that the mod had never been run so the assumption was untested.

It has now been run. The cache exists and the assumption holds.

### The cache format

```
char magic[8]        "DLSSNRW1"
u32  count           153
u32  index_size      5,673
repeat 153x, sorted lexicographically by name:
  u8   name_len
  char name[name_len]
  u64  offset        into the payload region, which begins at index_size
  u64  size
then the payloads, packed contiguously in the same lexicographic order
```

### Size proof, computed without transferring the file

The index size is predictable from the source container's own tensor table:

```
16 + 153*17 + 3,056 name bytes = 5,673        observed index_size = 0x1629 = 5,673
5,673 + 147,683,778 payload    = 147,689,451  observed cache size = 147,689,451
residual 0
```

The 153 index entries parse with exact consumption, every name and size matches the source table,
and the offsets form a contiguous lexicographic packing summing to exactly 147,683,778.

### Byte proof

sha256 over the same tensors on both sides — the source blob extracted from `nvngx_dlssnr.dll`, and
the cache produced by an actual game run on the RX 9070 XT:

| tensor | bytes hashed | result |
|---|---:|---|
| `block31.layer0.layer` | 65,536 | identical |
| `block23.layer1.layer` | 65,536 | identical |
| `block0.layer0.layer` | 21,696 (**whole tensor**) | identical |
| `block70.layer0.layer` | 21,808 (**whole tensor**) | identical |

**The transcode adds a header and an index and copies every payload unchanged.** Every offset in
section 8 that was measured against `WEIGHTS_HT` therefore describes, to the byte, the buffer the
GPU kernels actually read. The ISA-to-weight-byte agreement reported there is now verified rather
than assumed.

`dlssnr_cache.py verify` reproduces this on any machine that has run the mod;
`dlssnr_cache.py predict` does the size half with no cache file at all.
