# VarParams (168 B) — reconciled result

Three lanes attacked this independently. **All three had their assigned premise falsified in the same way** (HANDOFF2's kernel-key table is wrong past entry 26), and all three converged on the same real launcher. I re-derived the registration table, the shift table, the host store maps, the kernarg load maps, and the AMDGPU metadata from scratch before writing this. Everything below is my own reproduction unless explicitly attributed.

---

## 0. The premise correction (verified independently, 4th time)

HANDOFF2's key table assumed a `+8` stride throughout the 33 `__hipRegisterFunction` calls. The stride breaks after entry 26. Parsing the `(rdx, r8)` pairs in `0x18000EE00..0x18000F650` and resolving each `r8` string through the PE section table gives **34** registrations:

```
0x180055170 .. 0x180055238   26 kernels (k_swin_1h_32_fp8 … k_reproject)   [HANDOFF2 correct here]
0x180055240 .. 0x18005525F   NOT KEYS — 4x{i32,i32} shift table (see +0x20)
0x180055260  _Z11k_flag_waitPjjj
0x180055268  _Z10k_flag_setPjj
0x180055270  0x0000000000000000  (MSVC std::_Fake_allocator singleton, not a key)
0x180057A18  _Z10k_swin_varILi32ELb1EEv9VarParams
0x180057A20  _Z10k_swin_varILi32ELb0EEv9VarParams
0x180057A28  _Z10k_swin_varILi64ELb0EEv9VarParams
0x180057A30  _Z10k_swin_varILi128ELb0EEv9VarParams
0x180057A38  _Z10k_swin_varILi256ELb0EEv9VarParams
0x1800740F0  g_e4m3_lut   (via __hipRegisterVar @0x180053FF0, size 0x200)
```

Consequences that must propagate:

- **`0x18000CA02` is a `k_flag_wait` launch, not `k_swin_var<64,false>`.** Its `rbp+0x20 .. rbp+0xC8` "168-byte coincidence" is exactly that — `k_flag_wait(unsigned*, unsigned, unsigned)` takes three `args[]` entries; `k_swin_var` takes one. `0x18000D892` is `k_flag_set` (two entries). Discard both decodes.
- **`0x180017F25 / 0x18001C3D5 / 0x180023EC5` are not launches at all.** They `lea r9,[0x180055270]` = `std::_Fake_allocator`, inside inlined `std::vector::_Emplace_reallocate` bodies (adjacent max_size constants `0x38E38E38E38E38E` = SIZE_MAX/72 and `0x333333333333333` = SIZE_MAX/80).

**Every live `k_swin_var` launch is in one of two functions:**

| function | launches | note |
|---|---|---|
| `0x1800222F0` | `0x180022583` (32,f), `0x1800226D6` (128), `0x180022829` (64), `0x18002297C` (256), `0x180022ACF` (32,t) | the width dispatcher; all five instantiations inlined |
| `0x18001DDF0` | `0x18001E4CA`, `0x1800203FA` | two hand-rolled `<32,true>` sites |

The five generated `__device_stub__` thunks (`0x180021680`, `0x180023C00/C60/CC0/D20`) have zero callers.

---

## 1. The struct

`sizeof(VarParams) == 168` is **proven three ways, not assumed**:

1. AMDGPU msgpack metadata (`re/meta.msgpack`, target `amdgcn-amd-amdhsa--gfx1201`): every one of the five kernels declares `.args[0] = {.offset 0, .size 168, .value_kind by_value}`, `.kernarg_segment_size 424`, with `hidden_block_count_x` at offset **168 (0xA8)**. Therefore **kernarg offset ≡ struct offset**.
2. Host: the dispatcher builds five copies at `rsp+0x100 / 0x1A8 / 0x250 / 0x2F8 / 0x3A0` — stride exactly `0xA8`, top at `0x3A0+0xA8 = 0x448` = the whole `sub rsp,448h` frame.
3. Device: the kernels read kernarg `0xA8` (`s_load_b96 …, 0xa0` → s18, multiplied by `ttmp7`) and `0xB4` (`s_and_b32 …, 0xffff` → 256) — COV5 hidden args, which begin at `align8(sizeof(explicit))`.

```c
/* sizeof == 168 (0xA8), alignment 8.
   PROVEN     = store instruction read and checked, AND a kernarg load or use found
   INFERRED   = offset/width proven; role argued from one side only
   UNDETERMINED = offset/width proven; contents never traced
   Fields marked [narrow] are written only by the two hand-rolled <32,*> sites
   (0x18001E4CA, 0x1800203FA) and are s_load-ed only by the two C=32 kernels.
   The C=64/128/256 kernels never touch 0x40..0x9F, and the dispatcher zeroes it. */

struct VarParams {
/* 0x00 */ const void*  src;          /* PROVEN  ptr, kernel READS through it.
                                         Layout proven: [C/16][H][W][16 bytes], fp8.  */
/* 0x08 */ void*        dst;          /* PROVEN  ptr, kernel WRITES only (global_store_b8).
                                         Nullable; gated by flags bit 1.               */
/* 0x10 */ const void*  weights;      /* PROVEN  ptr, single per-block weight-blob base.
                                         All sub-tensors reached by immediate adds.    */
/* 0x18 */ int32_t      H;            /* PROVEN  slow / gridDim.y / ttmp7 extent       */
/* 0x1C */ int32_t      W;            /* PROVEN  fast / gridDim.x extent AND row stride*/
/* 0x20 */ int32_t      shiftX;       /* PROVEN  0 or -4; pairs with W                 */
/* 0x24 */ int32_t      shiftY;       /* PROVEN  0 or -4; pairs with H                 */
/* 0x28 */ uint32_t     flags;        /* PROVEN  6 live bits; see §4 and open items    */
/* 0x2C */ uint32_t     _pad0;        /* PROVEN padding — never stored by the dispatcher,
                                         never s_load-ed by any of the five kernels    */
/* 0x30 */ const void*  auxA;         /* PROVEN  ptr; kernel READS (global_load_b64)   */
/* 0x38 */ void*        auxB;         /* PROVEN  ptr; kernel READS *and* WRITES        */

/* ---- pre-block group, [narrow] : written only at 0x18001E4CA (flags 0x14) ---- */
/* 0x40 */ void*        preP0;        /* PROVEN  ptr (r13)                             */
/* 0x48 */ void*        preP1;        /* PROVEN  ptr ([rbp+618h])                      */
/* 0x50 */ float        preF0;        /* PROVEN  f32 from xmm7                         */
/* 0x54 */ uint8_t      preU0[8];     /* UNDETERMINED — one movq from xmm8 at a 4-mod-8
                                         offset, so NOT a pointer; contents not traced */
/* 0x5C */ uint32_t     preZero;      /* PROVEN  always literal 0, even at the live site*/
/* 0x60 */ uint8_t      preU1[8];     /* UNDETERMINED — one movq from xmm6             */
/* 0x68 */ uint32_t     preI0;        /* PROVEN  u32 ([rbp+688h])                      */
/* 0x6C */ uint32_t     _pad1;        /* PROVEN padding — not written at 0x18001E4CA,
                                         not s_load-ed (b96@0x60 ends at 0x6B)         */

/* ---- post-block group, [narrow] : written only at 0x1800203FA (flags 0x20) ---- */
/* 0x70 */ uint8_t      postU0[16];   /* UNDETERMINED — pshufd xmm8,0x4E (halves swapped);
                                         xmm8's source never traced                    */
/* 0x80 */ void*        postP0;       /* PROVEN  ptr (rsi)                             */
/* 0x88 */ float        postF0;       /* PROVEN  f32 from xmm7                         */
/* 0x8C */ float        postF1;       /* PROVEN  f32/u32 from xmm6 (movd)              */
/* 0x90 */ uint32_t     postI0;       /* PROVEN  literal 1 at the only live site        */
/* 0x94 */ uint32_t     _pad2;        /* PROVEN padding — not written at 0x1800203FA,
                                         not s_load-ed (b96@0x88 ends at 0x93)         */
/* 0x98 */ void*        postP1;       /* PROVEN  ptr ([rbp+668h])                      */

/* 0xA0 */ void*        workspace;    /* PROVEN  hipMalloc'd scratch, ctx+0x180        */
};                                     /* 0xA8 */
```

### Coverage

| | bytes | share |
|---|---|---|
| Offset + width proven (every byte is in a named slot) | 168/168 | **100 %** |
| Offset + width + **kind** proven | 136/168 | **81 %** |
| Contents undetermined (`preU0`, `preU1`, `postU0`) | 32/168 | 19 % |
| Semantic role fully proven (src/dst/weights/H/W/shiftX/shiftY/workspace + 3 pads) | 60/168 | 36 % |
| Pointers whose *tensor identity* is unknown (0x30, 0x38, 0x40, 0x48, 0x80, 0x98) | 48/168 | 29 % |

**For the wide tiers (C = 64/128/256) the struct is closed.** Only bytes `0x00..0x3F` and `0xA0..0xA7` are live there — 72 bytes — and 100 % of those are determined in offset, width and kind. The device kernarg load map proves the complement exactly:

```
C=32 (both):  0x00 b128 | 0x10 b64 | 0x18 b128 | 0x28 b32 | 0x30 b256 | 0x50 b128 |
              0x60 b96  | 0x70 b128| 0x80 b64  | 0x88 b96 | 0x98 b128 | (0xa8/0xb4 hidden)
C=64/128/256: 0x00 b128 | 0x10 b64 | 0x18 b128 | 0x28 b32 | 0x30 b128 | 0xa0 b96 | 0xb4
```
The three ranges never loaded by *any* kernel are `0x2C-0x2F`, `0x6C-0x6F`, `0x94-0x97` — precisely the three holes never written by *some* host site. Two independent sources, perfect complement.

### Key instructions (C = 64 lane, base `rsp+0x250`; base proven by `lea rax,[rsp+250h]` @`0x1800227E9` → `mov [rsp+40h],rax` @`0x1800227F1` → `lea r9,[rsp+40h]` @`0x180022AD6` → `call hipLaunchKernel` @`0x180022AE1`, `args[]` has exactly one entry)

| off | w | store | src | device use |
|---|---|---|---|---|
| 0x00 | 8 | `0x180022740` | `[rsp+60h]` = arg4 | `s_load_b128 s[12:15],0x0`; `v_mad_co_u64_u32 v[1:2],null,v5,s38,s[12:13]` → `global_load_b32` |
| 0x08 | 8 | `0x180022748` | `r12` = arg5 | s[14:15]; 9 address forms, **all** feeding `global_store_b8`, zero loads |
| 0x10 | 8 | `0x180022750` | `r15` = arg6 | `s_load_b64 s[10:11],0x10`; `+0x7000/0x7010/0x70a0/0x9000/0x9080` |
| 0x18 | 4 | `0x18002275F` | `[rsp+4C0h]` = arg7 | s8; `v_cmp_gt_i32 vcc_lo, s8, v5` where `v5 = shiftY + 8*ttmp7 + …` |
| 0x1C | 4 | `0x18002276D` | `[rsp+4C8h]` = arg8 | s9; `v_cmp_gt_i32_e64 s0, s9, v9`; **and** `s_lshl_b32 s38, s9, 4` = row pitch |
| 0x20 | 4 | `0x180022774` | `r13d` = tbl[arg9].lo | s10; `s_lshl_b32 s7,ttmp9,3` / `s_add_co_i32 s31,s10,s7` |
| 0x24 | 4 | `0x18002277C` | `ebp` = tbl[arg9].hi | s11; `s_lshl_b32 s12,ttmp7,3` / `s_add_co_i32 s30,s11,s12` |
| 0x28 | 4 | `0x180022787` | `[rsp+3Ch]` = arg3 | `s_load_b32 s36,0x28`; `s_bitcmp1_b32 s36,0/1`, `s_bitcmp0_b32 s36,2`, `s_and_b32 s0,s36,12/8` |
| 0x30 | 8 | `0x180022796` | `[rsp+4D8h]` = arg10 | `s_load_b128 s[4:7],0x30`; s[4:5] → `global_load_b64 v[7:8],v[3:4]` |
| 0x38 | 8 | `0x1800227A6` | `[rsp+4E0h]` = arg11 | s[6:7] → `global_load_b64` (L5093-5096) **and** `global_store_b8` (L5079) |
| 0x40..0x9F | 96 | `xorps xmm0` @`0x1800227AE` + 6×`movups` | zero | never loaded |
| 0xA0 | 8 | `0x1800227E1` | `r14` = `[ctx+0x180]` | `s_load_b96 s[16:18],0xa0`; `+ (gridX*ttmp7+ttmp9)<<13` |

**The axis assignment is proven, not conventional.** The C=64 address chain is
`addr = src + y*(16*W) + (x*16 + lane) + (W*H)*chgroup`, from
`s_lshl_b32 s38,s9,4` (=16·W) → `v_mad_co_u64_u32 v[1:2],null,v5,s38,s[12:13]`, then
`v_lshl_or_b32 v5,v9,4,v10` (x·16), then `s_mul_u64 s[18:19],s[0:1],s[2:3]` (s0=s9=W, s2=s8=H) →
`v_mad_co_u64_u32 v[1:2],null,s18,v9,v[1:2]` with `v9 = v3 & 48` ∈ {0,16,32,48} = the four 16-channel groups of C=64.
So the tensor is **`[C/16][H][W][16 bytes fp8]`**: `+0x1C` multiplies the row index (it *is* W), `+0x18` only ever bounds the `ttmp7` axis (it is H). Lane `crosscheck` reported this axis as undetermined; it is determined.

### Dispatcher signature (frame: 8 pushes + `sub rsp,448h` ⇒ bias 0x488; the function reads `[rsp+4B0h..4E0h]` and no home slot)

```c
void launch_swin_var(Ctx* ctx /*rcx*/, int C /*edx*/, uint32_t flags /*r8d*/,
                     const void* src /*r9*/, void* dst /*arg5*/, const void* weights /*arg6*/,
                     int H /*arg7*/, int W /*arg8*/, int shiftIdx /*arg9*/,
                     const void* auxA /*arg10*/, void* auxB /*arg11*/);
```
Width select: `add ebx,0FFFFFFE0h; rol ebx,1Bh; cmp ebx,7; ja` @`0x1800223FB` ⇒ index = ror32(C−32, 5); jump table at `0x180057A40` (RVA-relative, `add rcx,rax`): idx0→`0x180022431` (C=32, splits on `test byte ptr [rsp+3Ch],8`), idx1→C=64, idx3→C=128, idx7→C=256, idx 2/4/5/6→`0x180022AE6` (bail). **C = 96/160/192/224 are silently unsupported.**

---

## 2. Per-width values

**The five lanes are byte-for-byte identical.** I extracted all 17 stores per lane programmatically and compared the `(offset, width, source-operand)` tuples:

```
Identical to C=64 lane: {C=256: True, C=128: True, C=64: True, C=32,false: True, C=32,true: True}
```

There is **no per-width field anywhere in VarParams**. C is carried solely by the template instantiation.

| field | C=32 true<br>`rsp+0x3A0` | C=32 false<br>`rsp+0x2F8` | C=64<br>`rsp+0x250` | C=128<br>`rsp+0x1A8` | C=256<br>`rsp+0x100` | site `0x18001E4CA`<br>`<32,true>` | site `0x1800203FA`<br>`<32,true>` |
|---|---|---|---|---|---|---|---|
| +0x00 src | arg4 @`…29E6` | arg4 @`…249A` | arg4 @`…2740` | arg4 @`…25ED` | arg4 @`…2893` | **literal 0** @`0x18001E3B8` | `r14` @`0x180020323` |
| +0x08 dst | r12 @`…29EE` | r12 @`…24A2` | r12 @`…2748` | r12 @`…25F5` | r12 @`…289B` | `[rbp+690h]` @`0x18001E3CA` | **literal 0** @`0x180020327` |
| +0x10 weights | r15 @`…29F6` | r15 @`…24AA` | r15 @`…2750` | r15 @`…25FD` | r15 @`…28A3` | `rdi` @`0x18001E3D1` | `rdi` @`0x18002032F` |
| +0x18 H | arg7 @`…2A05` | arg7 @`…24B9` | arg7 @`…275F` | arg7 @`…260C` | arg7 @`…28B2` | `esi` @`0x18001E3D8` | `r15d` @`0x180020333` |
| +0x1C W | arg8 @`…2A13` | arg8 @`…24C7` | arg8 @`…276D` | arg8 @`…261A` | arg8 @`…28C0` | `r15d` @`0x18001E3DE` | `r12d` @`0x180020337` |
| +0x20 shiftX | tbl.lo @`…2A1A` | tbl.lo @`…24CE` | tbl.lo @`…2774` | tbl.lo @`…2621` | tbl.lo @`…28C7` | **0** (qword-0 @`0x18001E3E5`) | **0** (qword-0 @`0x18002033B`) |
| +0x24 shiftY | tbl.hi @`…2A22` | tbl.hi @`…24D6` | tbl.hi @`…277C` | tbl.hi @`…2629` | tbl.hi @`…28CF` | **0** (same store) | **0** (same store) |
| +0x28 flags | arg3 @`…2A2D` | arg3 @`…24E1` | arg3 @`…2787` | arg3 @`…2634` | arg3 @`…28DA` | **0x14** @`0x18001E3F0` | **0x20** @`0x180020343` |
| +0x2C | *not written* | *not written* | *not written* | *not written* | *not written* | 0 | 0 |
| +0x30 auxA | arg10 @`…2A3C` | arg10 @`…24F0` | arg10 @`…2796` | arg10 @`…2643` | arg10 @`…28E9` | **0** | **0** |
| +0x38 auxB | arg11 @`…2A4C` | arg11 @`…2500` | arg11 @`…27A6` | arg11 @`…2653` | arg11 @`…28F9` | `r12` @`0x18001E40F` | **0** |
| +0x40..0x6F | **0** (`xorps`) | **0** | **0** | **0** | **0** | **LIVE** (pre group) | **0** |
| +0x70..0x9F | **0** | **0** | **0** | **0** | **0** | **0** (`pxor` @`0x18001E45A`) | **LIVE** (post group) |
| +0xA0 ws | r14 @`…2A87` | r14 @`…253B` | r14 @`…27E1` | r14 @`…268E` | r14 @`…2934` | `rbx` @`0x18001E476` | `rbx` @`0x1800203AF` |

**The discriminating axis is the call site, not C.** The dispatcher's five lanes are compiler-duplicated copies of one construction sequence; they carry **zero** width-discriminating information. The only three genuinely distinct VarParams instances in the binary are: dispatcher (all 5), pre-block site, post-block site.

**Correcting lane `wide64` on the tail.** It reported the two `<32,true>` sites as filling `0x40..0x9F` "in mutually incompatible ways… a variant/union region". They do not overlap at all. My store dump:

```
0x18001E4CA site fills: 0x38 0x40 0x48 0x50 0x54 0x5C 0x60 0x68        (zeroes 0x70..0x9F)
0x1800203FA site fills:                          0x70 0x80 0x88 0x8C 0x90 0x98  (zeroes 0x2C..0x6F)
```
Two **disjoint** field groups, one per role, selected by flags bit 4 (`0x14`) vs bit 5 (`0x20`). The "xmm7 lands at +0x50 at one site and +0x88 at the other" is two different fields, not a contradiction. `k_swin_var<32,*>` is a merged pre-block + swin + post-block kernel; the struct is a plain superset, not a union.

### Launch geometry (identical at all five widths; verified)

```
blockDim  = (256, 1, 1)          mov rax,100000100h @0x18002243C  (+ metadata max_flat_workgroup_size 256)
sharedMem = 0                    xor r8d,r8d   before every __hipPushCallConfiguration
stream    = 0 (default)          xor r9d,r9d
gridDim.y = ceil((H - shiftY)/8) 0x180022332..0x18002233F   (lea+7 / add 0Eh / cmovns / sar 3)
gridDim.x = ceil((W - shiftX)/8) 0x180022342..0x180022350
workspace = gridX*gridY << k     k = 12 or 13 (C=32, per flags&0x38), 13 (C=64), 14 (C=128), 15 (C=256)
                                 shl r15,cl @0x18002237D / shl r15,0Dh @…2394 / 0Eh @…23AC / 0Fh @…23C4
                                 grow-only: cmp r15,[ctx+188h]; hipFree(0x180054090); hipMalloc(0x180054100)
```
`1<<13 = 8·8·64·2` and `1<<14 = 8·8·128·2` ⇒ **8×8 tokens per workgroup, C channels, 2 bytes (fp16) per element** in the workspace. 256 threads / 64 tokens = 4 threads per token.

### The shift table (raw `.rdata` bytes at VA `0x180055240`, read from the file)

```
[0] (  0,   0)     [1] ( -4,  -4)     [2] ( -4,   0)     [3] (  0,  -4)
```
Exactly four `{i32,i32}` entries — half of the window size 8 in each axis independently. Selected by `movsxd rax,[rsp+4D0h]` / `mov r13d,[rcx+rax*8]` @`0x18002232A`, **with no bounds check**. `arg9 = 4` would read `0x180055260 = 0x180001AE0` (the `k_flag_wait` stub pointer) as shiftX. Callers load arg9 from a runtime int array, so "only 0 or −4" is caller discipline, not a property of this function.

---

## 3. What it unlocks

**Sub-tensor displacements for the wide tiers — NO, not from VarParams.** The struct contains no offsets, no channel counts, no shapes. Not one stored constant equals C, 2C, 3C, C², 4C², 128C, 256C, or any within-block byte offset; the only literals ever stored anywhere are `{0, −4, 1, 20, 32}`, and the one nominal match (`32` at +0x28) is killed by the other narrow site storing `20` into the same field for the same kernel.

What VarParams *does* unlock is the **anchor**: `+0x10` is the single weight-blob base, and the wide kernels reach sub-tensors from it by immediate adds, exactly as `k_swin_1h_32_fp8` does at C=32 (FINDINGS §8's "one weight pointer" premise, now confirmed for C ≥ 64). I confirmed in the C=64 disassembly:

```
absolute off VarParams+0x10:  s_cselect_b32 s2, 0x7000, 0        ; g1 slot base
                              s_cselect_b32 s4, s1(0x9000), 0x7010   ; g1   (0x7010 down / 0x9000 up)
                              s_cselect_b32 s4, s1,        0x70a0    ; Q    (0x70a0 down / 0x9080 up)
Q-relative (s[12:13] = weights + Q):
                              s_add_nc_u64 s[22:23], s[12:13], 0x7000   ; S
                              s_add_nc_u64 s[12:13], s[12:13], 0x8010   ; g2
```
Those two Q-relative immediates independently confirm lane `crosscheck`'s **corrected** C=64 layout — `Q@28832, T@41120, S@57504, P@57520, g2@61616` (32 B higher than my brief's numbers; the brief omitted the two 16-byte pads bracketing `g1`). Check: `S−Q = 3C²+256C = 0x7000` ✓, `g2−Q = 3C²+256C+0x10+C² = 0x8010` ✓, and the schema sums to `61,760 = 9C²+388C+48+16` ✓. The up-block alternates (`0x9000/0x9080` at C=64) are exactly "prepend one 2C² FP8 tensor, unpadded 2C gain vector" — 2/2 at C=64, 6/6 across widths per `crosscheck`.

**Window / shift schedule — YES, the mechanism; NO, the sequence.**
Proven: window = 8×8 elements, one workgroup per window, 256 threads, shift ∈ {0, −4} = half-window, and the phase set is exactly the four rows above. Grid = `ceil((W−shiftX)/8) × ceil((H−shiftY)/8)`. Shifting is realised by **an enlarged grid plus border masking**, not by a cyclic roll — `v_cmp_gt_i32_e64 s0, s9, v9` / `v_cmp_lt_i32_e64 s1, -1, v9` mask lanes out at the edges; a `torch.roll` reimplementation will not match. What is *not* recovered: which phase each block uses. `arg9` comes from a runtime array (`mov r15d,[rcx+rax*4]` @`0x18001EA89`), and the caller loops `cmp r9d,4`.

**Token counts — the formula, not the numbers.** Tokens = H·W elements at 1 token per element (layout `[C/16][H][W][16 B]`). H and W arrive as `arg7`/`arg8`, read from a runtime 12-byte-stride descriptor `{C, H, W}` (`mov edx,[rbx]` / `mov eax,[rbx+4]` / `mov esi,[rbx+8]` @`0x18001EACA/EA8D/EA96`). The static binary contains **no resolution**. At 1920×1080 phase 0 the arithmetic is exact and self-consistent — grid 240×135 = 32,400 workgroups × 64 = 2,073,600 = 1920·1080, workspace 32,400×4096 = 1920·1080·32·2 — which is a consistency check on "1 token per pixel at the C=32 tier", not a measurement.

### FINDINGS open items

| item | status |
|---|---|
| §6 "The bool in `k_swin_var<T,bool>`" | **CLOSED** — see §4 below |
| §6 "Window-shift schedule" (part of inter-block dataflow) | **CLOSED at the mechanism level** — 8×8 window, 4-phase {0,−4}² table at `0x180055240`, grid formula, masking-not-rolling. Phase sequence still runtime. |
| §6 "U-Net skip wiring" | **PARTIAL** — `+0x38` is non-null only on the last block of a stage at caller `0x18001EB08` (`[r13+rax*8+0x1F8]`), and `+0x30` only at caller `0x18001FCCC` (`[rbp+0x668]`). I additionally proved `+0x38` is both read and written by the C=64 kernel. Direction of the skip is still open. |
| §6 "What `Scale` (0.03125) multiplies" | **NOT CLOSED.** Two f32 fields exist (`+0x50`, `+0x88`), both fed from `xmm7`, both only at narrow hand-rolled sites. Neither traced to a source. Candidate carriers only. |
| §6 "INI keys / env vars" | not addressed |
| §8.1 "Factorisation of FP8 slabs" | **NOT CLOSED** — VarParams carries no shapes. |
| §8.2 "What `A1` is" | **NOT CLOSED** |
| §8.3 "What `T` is" | **NOT CLOSED** |
| §8.4 "Runtime dimensions of `k_swin_var`" | **MOSTLY CLOSED**, and the premise partly **falsified**. Window size, token blocking and shift schedule: recovered. Per-launch channel count: **not in VarParams** — C is a template parameter dispatched from the runtime descriptor. FINDINGS predicted VarParams "carries as literal host constants everything item 4 is missing"; it carries no literal constants at all beyond `{0,−4,1,20,32}`. |
| §8.5 "Kernel-to-blob binding" | **CLOSED for `k_swin_var`.** `+0x10` = return of `0x1800210C0(ctx, blockIdx, 0)`, which composes the string `"block" + itoa(i) + ".layer" + itoa(j)` (literals present in `.rdata`) and passes it to `0x180023780`, whose success path is `mov rcx,[rsi+10h]; add rcx,[rax+40h]` = weight-arena base + that blob's byte offset (miss path logs `missing blob %s` and returns the bare arena base). This is the **first name→kernel binding in the whole analysis** — every prior one was constant-coincidence inference. |
| §8 "S: consumer unidentified" | **CLOSED per lane `crosscheck`** — `k_swin_1h_32_fp8` addresses `S` at C=32. Flagged: `crosscheck` cites `0x4c60` where FINDINGS §8 lists `0x4c70`; reconcile before publishing. |

---

## 4. The bool template parameter — SETTLED

`<32,true>` means: **stage the second window tile in the global workspace instead of in LDS, and double the per-workgroup workspace stride from 4 KiB to 8 KiB.** It is not "shifted window" (that is the runtime `+0x20/+0x24` pair) and not "reads more tensors" (the two kernels' kernarg load maps are byte-identical).

Five independent confirmations, four of which I re-ran:

1. **Host sizing.** `test r8b,38h; sete cl; xor cl,0Dh; shl r15,cl` @`0x18002236A..0x18002237D`: `(flags & 0x38) == 0` → `cl=1` → shift 12 (4096 B); non-zero → `cl=0` → shift 13 (8192 B).
2. **Device constant.** Line 64 of each kernel: `s_lshl_b64 s[2:3], s[2:3], 12` in `<32,false>` vs `s_lshl_b64 s[2:3], s[2:3], 13` in `<32,true>`. One arithmetic constant.
3. **The storage substitution, exact.** `<32,false>` has eight LDS ops at offsets **15616, 15680, 15744, 15808, 15872, 15936, 16000, 16064** (stride 64). `<32,true>` has **none of them**, and instead eight `global_store_b16` at **offset:4096, 4160, 4224, 4288, 4352, 4416, 4480, 4544** — the same eight slots, same 64-byte stride, relocated to `workspace + 0x1000`. Arithmetic closes: 256 lanes × 8 half-words × 2 B = 4096 B = exactly the extra workspace.
4. **Op counts.** `ds_store_b16` 35→33, `ds_load_u16` 79→71, `global_store_b16` 0→11, `global_load_u16` 25→39.
5. **Metadata LDS.** `.group_segment_fixed_size`: `<32,false>` = **15,632**, `<32,true>` = **15,616**.

**Trigger:** `(flags & 0x38) != 0`. Bit 3 in the dispatcher (`test byte ptr [rsp+3Ch],8` @`0x180022431`), bit 4 at the pre-block site (flags `0x14`), bit 5 at the post-block site (flags `0x20`). Note the dispatcher selects the *template* on bit 3 alone but sizes the *workspace* on `0x38`; a caller passing `0x10` or `0x20` into the dispatcher would get `<32,false>` code with an 8 KiB allocation. No dispatcher caller was observed doing so, but nothing enforces it.

**Anomaly, unexplained (carried from lane `wide256`, and I reproduced it).** `<32,false>` declares 15,632 B of LDS yet its `ds_load_u16` reaches `offset:16064` (+2 B = 16,066), 434 B past the allocation. Either the address register carries a compensating negative bias I did not trace, or the region is only reached on a dead path. It does not touch the conclusion, which rests on items 1–3 and 5.

---

## 5. What is still missing for a forward pass

1. **The per-stage schedule `{C, H, W, shiftIdx}` — the single biggest gap.** It lives in runtime arrays the dispatcher's three callers (`0x18001EB08`, `0x18001FCCC`, `0x18001FDE9`) index: a 12-byte-stride `{C,H,W}` descriptor at `[rbx]` / `[r8+rdx*4]`, and a separate int array for `arg9`. Those tables are populated further up a chain nobody has traced. Without them there are no resolutions and no shift sequence. **This is the obvious next target** — it is one or two levels of pointer chase above `0x18001DDF0`.
2. **Tensor identity of six pointer fields** — `+0x30`, `+0x38`, `+0x40`, `+0x48`, `+0x80`, `+0x98`. I proved directions for two (`+0x30` read; `+0x38` read+write) — an advance on all three lanes, which called both undetermined — but nothing names them. Nullability pattern is the lever: `+0x38` non-null only on the last block of a stage, `+0x30` only at one caller.
3. **Flags bits 0/1/2.** Proven: bit 0 gates the `+0x00` read region, bit 1 gates the `+0x08` write, bit 2 is tested with `s_bitcmp0`, `(flags & 12) == 8` selects an alternate weight-displacement pair, bits 3/4/5 pick the bool + workspace size + which extra-tensor group is live. Three callers encode "first"/"last" three different ways: `first + 4*last` (`lea r8d,[rdx+rcx*4]` @`0x18001EAC6`), `2*last` (`add r8d,r8d` @`0x18001FDB2`), and the constant `8`. **No single consistent naming of bits 0/1/2 is established.**
4. **The 32 undetermined tail bytes** — `+0x54` (movq xmm8), `+0x60` (movq xmm6), `+0x70..0x7F` (`pshufd xmm8, 0x4E`). Offsets and widths are nailed; nobody traced what those XMM registers hold. `+0x54` at a 4-mod-8 offset rules out a pointer, which contradicts HANDOFF.md's heuristic "ptr at 0x58" — treat that as a false positive.
5. **`Scale = 0.03125`** — still unproven to reach the GPU. `+0x50` and `+0x88` are the only candidate carriers found and neither was traced.
6. **FP8 slab factorisation, `A1`, `T`** — untouched by this work. VarParams contains no shape information whatsoever, so the route to these is the ISA, not the host.
7. **Wide-tier sub-tensor displacements are only partly mined.** At C=32 they are literal immediates; at C ≥ 64 several are register-computed (`s_add_nc_u64 s[..], s[10:11], s[..]`) and need arithmetic traced. `crosscheck` proved `S` and `g2` by register trace at C=64/128/256 and `P` at C=128/256 only — `P` at C=64 (`0x7010`) is numerically indistinguishable from `g1`'s absolute offset and is **not** independently confirmed. `T`'s `3C²` immediate is present at all three widths but not closed by register trace.

---

## Lane assessment

- **`wide64`** — premise false (its two assigned launch sites are `k_flag_wait`/`k_flag_set`). Recovered fully; found the real dispatcher, proved the empty width-diff programmatically, and got the axis assignment right. **One claim I refute:** the `0x40..0x9F` tail is not a "variant/union region with mutually incompatible layouts" — the two narrow sites write strictly disjoint byte ranges.
- **`wide256`** — premise false (its three sites are `std::_Fake_allocator` in inlined `_Emplace_reallocate`). Produced the single most valuable result in the batch: the bool, settled on four independent lines. **One claim I refute:** its VGPR/SGPR table (103/92/125/136/136, 69/65/76/76/76) does not match either msgpack file; both give `<32,true>` = 81 VGPR / 82 SGPR. Its LDS numbers are correct.
- **`crosscheck`** — most complete field table; I reproduced it store-for-store with zero discrepancies. Its weight-schema correction (+32 B from `Q` onward) is confirmed by two register-traced immediates I found independently (`0x7000`, `0x8010` off the Q base). It is over-cautious on the H/W axis, which the C=64 addressing settles.

No lane produced nothing usable.