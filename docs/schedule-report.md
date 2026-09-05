=========== ROUTE 1 ===========
SUMMARY: Route 1 is closed. `rbp+0x560` is neither a std::vector header nor a single inline array — it is a reused stack slot that holds a std::string and a std::vector<char> earlier in function 0x18001DDF0 and, at the site in question, a 4-element inline array of 40-byte stage records. Base and stride are proven twice over: by the four literal constructions at 0x18001FA02/0x18001FA84/0x18001FAF7/0x18001FB6D and by the EH unwind funclet at 0x180020AE1 that walks -0x28 down to `lea rax,[rbp+0x560]`. The record is {int first; int last; int shiftIdx0; int pad; std::vector<int> shifts} — no C, no H, no W, no tensor pointer. A second, 32-byte encode record type at rbp+0x4B0 (which HANDOFF3 did not have) drops the shiftIdx0 field and puts the vector at +8; its own funclet at 0x180020BB4 steps -0x20. C/H/W come from a six-entry {C,H,W} table at ctx+0x190 built by resize() at 0x18001D133..0x18001D1F6, wh

RECORD LAYOUT:
40-byte DECODE stage record, array of 4 at rbp+0x560 in function 0x18001DDF0 (base and stride 0x28 proven by the unwind funclet at 0x180020AE1..0x180020B4A, which walks -0x28 down to `lea rax,[rbp+0x560]`):

  +0x00  int32   firstBlock      written as the low half of one imm64 (0x18001FA02 / 0x18001FA84 / 0x18001FAF7 / 0x18001FB6D); read at 0x18001FC6A as the blob index of the stage's up-block, and at 0x18001FD4C / 0x18001FD81 as the inner loop's lower bound and array origin
  +0x04  int32   lastBlock       high half of the same imm64; read at 0x18001FD72 as the loop's inclusive upper bound
  +0x08  int32   shiftIdx0       shift-table row for the up-block only; written 0/2/0/0 (0x18001FA13, 0x18001FA95, 0x18001FB08, 0x18001FB7E); read at 0x18001FC71 -> dispatcher arg9
  +0x0C  int32   padding         never written, never read
  +0x10  int*    shifts._Myfirst written 0x18001FA4C/0x18001FAC3/0x18001FB36/0x18001FBAC; read at 0x18001FD87, indexed `shifts[i-1-firstBlock]` for blocks firstBlock+1..lastBlock -> arg9
  +0x18  int*    shifts._Mylast  written 0x18001FA76/0x18001FAE9/0x18001FB5F/0x18001FBCB; never read by the loop
  +0x20  int*    shifts._Myend   written 0x18001FA5A/0x18001FAD1/0x18001FB44/0x18001FBBA; read only by the destructor (0x180020B55) for the aligned-vs-plain free test

Allocation size is 4*(lastBlock-firstBlock) in all four cases (0x1C, 0x14, 0x0C, 0x0C at 0x18001FA42 / 0x18001FAB9 / 0x18001FB2C / 0x18001FBA2), which is exactly the number of blocks the inner loop runs — an arithmetic check on the whole reading.

There is NO C, H, W, pointer-to-tensor or flags word in the record. C/H/W come from ctx+0x190 (see below); flags are computed at the call site.

The ENCODE record is a different, 32-byte type, array of 4 at rbp+0x4B0 (stride from `shl rcx,5` @0x18001E9EA, funclet -0x20 @0x180020BC4):
  +0x00 int32 firstBlock | +0x04 int32 lastBlock | +0x08/+0x10/+0x18 std::vector<int> shifts (read at [r8+8] @0x18001EA85, indexed `shifts[i-firstBlock]` over firstBlock..lastBlock inclusive). No shiftIdx0 field, no padding.

The per-tier geometry lives in a SIXTH-order table at ctx+0x190, 12-byte stride, {int32 C; int32 H; int32 W}, written by resize() at 0x18001D133..0x18001D1F6:
  [0]={32, H>>1, W>>1} [1]={64, H>>2, W>>2} [2]={128, H>>3, W>>3} [3]={256, H>>4, W>>4} [4]={512, H>>5, W>>5} [5]={1024, H>>6, W>>6}
with ctx+0x18 = H and ctx+0x1C = W set at 0x18001D10F/0x18001D112, each = ceil(render dim / 128)*128 (0x18001C850-0x18001C87F). Its base addres

SCHEDULE:
Recovered in full and statically. Function 0x18001DDF0 runs the whole network; `blockIdx` is the argument to the blob resolver 0x1800210C0 ("block"+itoa, .rdata 0x180063F3B). H_k = H>>(k+1), W_k = W>>(k+1) from ctx+0x190; the shift column is the row index into .rdata 0x180055240 = [(0,0), (-4,-4), (-4,0), (0,-4)].

  blk   what                       C      H,W (tier)  shift          site
  0     pre  (hand-rolled <32,t>)  32     tier0 / full-res stem   flags 0x14   0x18001E4CA, blob 0 @0x18001E139
  1-4   enc0                       32     H/2,  W/2   0,1,2,3      0x18001EB08, rec rbp+0x4B0 @0x18001E7E1
  5-8   enc1                       64     H/4,  W/4   0,1,2,3      rec rbp+0x4D0 @0x18001E84E
  9-14  enc2                      128     H/8,  W/8   0,1,2,3,0,1  rec rbp+0x4F0 @0x18001E8A9
  15-22 enc3                      256     H/16, W/16  0,1,2,3,0,1,2,3   rec rbp+0x510 @0x18001E91B
  23-30 vit512a                   512     H/32, W/32  (i-23)%4     loop 0x18001ED33, launcher 0x180022B60
  31-38 vit1024 loop             1024     (token-count grid)       loop 0x18001F0E5
  39    vit1d                    1024     -                        0x18001F8CA
  40-47 vit512b                   512     H/32, W/32  (i-40)%4     loop 0x18001F977
  48-55 dec0 (48 = swinup)        256     H/16, W/16  0 | 1,2,3,0,1,2,3   0x18001FCCC + 0x18001FDE9, rec rbp+0x560 @0x18001FA02
  56-61 dec1 (56 = swinup)        128     H/8,  W/8   2 | 3,0,1,2,3       rec rbp+0x588 @0x18001FA84
  62-65 dec2 (62 = swinup)         64     H/4,  W/4   0 | 1,2,3           rec rbp+0x5B0 @0x18001FAF7
  66-69 dec3 (66 = swinup)         32     H/2,  W/2   0 | 1,2,3           rec rbp+0x5D8 @0x18001FB6D
  70    post (hand-rolled <32,t>)  32     tier0       flags 0x20   0x1800203FA, blob 70 @0x1800201D7

("a | b,c" = the shiftIdx0 field at record+8 for the stage's up-block, then the vector for the rest. Stage names are the binary's own log strings: 'enc%d' 0x1800645BE, 'dec%d' 0x1800645C4, 'swin%d_C%d' 0x1800645DB, 'swinup%d_C%d' 0x1800645CE, 'vit512a/b' 0x1800646CE/C6, 'vit1d' 0x1800645AD, 'pre' 0x18006424B.)

Dataflow, all proven at the call sites: encode stage s ping-pongs ctx[0x2A8+8*(3-s)] and ctx[0x1D8+8*s]; its FIRST block (flags bit0) reads ctx[0x1F8+8*(s-1)] (or ctx+0x220 = the 'pre' output for s=0); its LAST block (flags bit2) additionally writes the half-resolution downsample into auxB = ctx[0x1F8+8*s]; the stage result is normalised into ctx[0x1D8+8*s] by a D2D copy of C*H*W bytes (0x18001EBCA). 

FINDINGS:
  [proven] `rbp+0x560` in function 0x18001DDF0 is NOT a single container. It is a REUSED stack slot. At the site HANDOFF3 asks about (0x18001FC6A) it holds a 4-element INLINE ARRAY of 40-byte stage records; earlier in the same function the s
     @ 0x18001DE03 (lea rbp,[rsp+0x80], sub rsp,778h -> rbp+0x560 is inside the frame, not a heap ptr); 0x18001E58D/0
  [proven] The 40-byte decode stage record is `struct { int32 firstBlock; int32 lastBlock; int32 shiftIdxOfFirstBlock; int32 _pad; std::vector<int> shifts; }` with the MSVC vector header {_Myfirst@0x10,_Mylast@0x18,_Myend@0x20}. There is no 
     @ construction 0x18001FA02 (mov rax,3700000030h -> [rbp+0x560]), 0x18001FA13 (+8 = 0), 0x18001FA21/0x18001FA29 (
  [proven] The ENCODER uses a DIFFERENT record type: a 4-element array of 32-byte records at rbp+0x4B0, `struct { int32 firstBlock; int32 lastBlock; std::vector<int> shifts; }` (vector at +0x08/+0x10/+0x18). HANDOFF3 only saw the decode one.
     @ construction 0x18001E7E1..0x18001E90D (records at rbp+0x4B0, 0x4D0, 0x4F0, 0x510); indexing 0x18001E9EA (mov r
  [proven] C, H and W do NOT live in the stage record. They come from a 6-entry, 12-byte-stride {int32 C; int32 H; int32 W} table at ctx+0x190, built by the resize routine 0x18001D0B0. C = 32,64,128,256,512,1024 and H_k = H>>(k+1), W_k = W>>
     @ 0x18001D133 (mov [rsi+0x190],20h) + 0x18001D13D (movq [rsi+0x194],xmm1 = {H/2,W/2}); 0x18001D160/0x18001D16A (
  [proven] Field->VarParams mapping, end to end. desc.C -> dispatcher arg2 (edx) -> template instantiation only (never stored in VarParams). desc.H -> arg7 -> VarParams+0x18. desc.W -> arg8 -> VarParams+0x1C. record shift value -> arg9 -> ro
     @ encode: 0x18001EACA (mov edx,[rbx] = C), 0x18001EA8D->0x18001EAE1->0x18001EAE7 ([rsp+0x30]=arg7=H), 0x18001EA9
  [proven] Records are WRITTEN nowhere but two literal, fully-static construction sequences — there is no runtime schedule computation. All eight records (4 encode + 4 decode) and all six shift arrays are compile-time constants; only H and W
     @ encode records: 0x18001E7E1 (1,4), 0x18001E84E (5,8), 0x18001E8A9 (9,14), 0x18001E91B (15,22). decode records:
  [proven] The shift table at .rdata 0x180055240 is four {i32,i32} rows [(0,0),(-4,-4),(-4,0),(0,-4)] and every shift value that reaches it is in 0..3, so the missing bounds check is never exercised by this schedule.
     @ table read at 0x18002232A/0x18002232E; file bytes at VA 0x180055240 (file offset 342592) = 0,0,-4,-4,-4,0,0,-4
  [proven] Block-index coverage is complete and exactly matches the independently derived weight-tier table: 0=pre, 1-4/5-8/9-14/15-22=encode, 23-30 and 40-47=C512 middle, 31-38 and 39=C1024 middle, 48-55/56-61/62-65/66-69=decode, 70=post. 7
     @ pre getBlob(ctx,0,0) @0x18001E139/0x18001E13E and 0x18001E279/0x18001E27E; C512a loop `mov edi,17h` @0x18001ED
  [proven] The decode stage index s maps to tier j=3-s, confirmed a second time by the buffer allocator: ctx+0x1D8[k] and ctx+0x1F8[k] are sized from desc[k], while ctx+0x2A8[k], ctx+0x2C8[k] and ctx+0x2E8[k] are sized from desc[3-k].
     @ allocator loop 0x18001D4AA (r13=0x198, forward) / 0x18001D4A3 (r12=ctx+0x1BC, stepped -0xC @0x18001D4D3) / sto
  [proven] The network resolution is the caller's size rounded UP to a multiple of 128, and the whole ladder derives from it by halving. HANDOFF3/VARPARAMS's '1920x1080 -> 240x135 grid at the C=32 tier' consistency check is wrong: the C=32 t
     @ 0x18001C83D..0x18001C87F (movd/punpckldq {W,H}; paddd with .rdata 0x180056B90 = {127,127,0,0}; psrad 31 / psrl

OPEN: Where [rsp+0x2F0] and [rsp+0x2F8] at 0x18001C83D come from — the pre-alignment render H and W. I did not trace them out of function 0x18001C7A0, so every absolute pixel figure I give is conditional on 'the caller passes 1920x1080'. | The C=1024 middle (blocks 31-38 loop at 0x18001F0E5, plus block 39 at 0x18001F8CA) uses hand-emitted launches through 0x180053FC0/0x1800540F0 with gridDim {ceil(ctx[0x314]/64), 32, 1} (0x18001F148-0x18001F15F). ctx+0x314 is a token count I did not trace, so the C=1024 tier has no recovered H/W — desc[5] exists but I never saw it read. | The C=512 launcher 0x180022B60 (called at 0x18001ED9B and 0x18001F9D0) was not disassembled. Its arg1 is a 32-byte stack struct
=========== ROUTE 2 ===========
SUMMARY: The whole resolution chain is recovered and closed, end to end, from the DXGI staging dimensions to each kernel's gridDim.

Chain: colour staging W=[0x180074858], H=[0x18007485C] (proven to be the very words printed by the `staging ready: colour %ux%u` line) -> padded UP to a multiple of 128 per axis at 0x18001C850..0x18001C879 -> stored as ctx+0x18 (padH) / ctx+0x1C (padW) at 0x18001D10F/0x18001D112 (these are the words printed by `padded %dx%d`) -> a 6-entry, 12-byte-stride {C,H,W} descriptor table at ctx+0x190 built by one SSE sequence at 0x18001D115..0x18001D1F6, where tier k has C = 32<<k and H,W = padH>>(k+1), padW>>(k+1) -> descriptor +4/+8 become dispatcher arg7/arg8 -> VarParams+0x18/+0x1C -> gridDim = (ceil((W-shiftX)/8), ceil((H-shiftY)/8), 1) at 0x180022332..0x180022350.

At 1920x1080 the padded base is 1920x1152 and the ladder is exact at every tier (padding to 128 = 2^7 is 

RECORD LAYOUT:
Two different stage-record layouts exist, and they are NOT the same struct. HANDOFF3 saw only the second.

ENCODER stage record - 32 bytes, 4 records at rbp+0x4B0 / 0x4D0 / 0x4F0 / 0x510. Stride proven by `mov rcx,rdx; shl rcx,5; lea r8,[rcx+rbp]; add r8,4B0h` at 0x18001E9E7..0x18001E9F2.
  +0x00 i32 firstBlockIdx   read `mov edi,[r8]` @0x18001EA7B (also the loop seed, `mov r14d,[rbp+rcx+4B0h]` @0x18001E9F9)
  +0x04 i32 lastBlockIdx    read `mov r12d,[r8+4]` @0x18001EA46, loop test `cmp r14d,r12d; jg` @0x18001EA4A
  +0x08 ptr  int* shiftIdx  read `mov rcx,[r8+8]` @0x18001EA85, indexed by (blockIdx - first) @0x18001EA89
  +0x10 ptr  end            std::vector end   (written 0x18001E82F pattern)
  +0x18 ptr  capacity       std::vector cap   (written 0x18001E840 pattern)
  Initialisers: 0x18001E7E1 {1,4}, 0x18001E84E {5,8}, 0x18001E8A9 {9,14}, 0x18001E91B {15,22}.

DECODER stage record - 40 bytes, 4 records at rbp+0x560 / 0x588 / 0x5B0 / 0x5D8. Stride proven by `lea rdi,[rdx+rdx*4]` (=5*k) then `[rbp+rdi*8+560h]` @0x18001FC66/0x18001FC6A and `lea r13,[rdi*8+560h]; add r13,rbp` @0x18001FCD1.
  +0x00 i32 firstBlockIdx   read @0x18001FC6A (arg2 of the blob resolver for the tier's swinup call) and @0x18001FCEA, @0x18001FD81
  +0x04 i32 lastBlockIdx    read `mov edi,[r13+4]` @0x18001FD72, loop test `cmp eax,edi; jge` @0x18001FD76
  +0x08 i32 upShiftIdx      read `mov esi,[rbp+rdi*8+568h]` @0x18001FC71 -> dispatcher arg9 for the upsample call; values 0,2,0,0
  +0x0C i32 (padding)       never written, never read - the struct is 8-aligned because +0x10 is a pointer
  +0x10 ptr  int* shiftIdx  read `mov rcx,[r13+10h]` @0x18001FD87, indexed by (blockIdx - first)
  +0x18 ptr  end            (written 0x18001FA5A / 0x18001FAD1 / 0x18001FB44 / 0x18001FBBA)
  +0x20 ptr  capacity       (written 0x18001FA76 / 0x18001FAE9 / 0x18001FB5F / 0x18001FBCB)
  Initialisers: 0x18001FA02 {48,55} shift0=0, 0x18001FA84 {56,61} shift0=2, 0x18001FAF7 {62,65} shift0=0, 0x18001FB6D {66,69} shift0=0.

Neither record carries C, H or W. Those come from the separate 12-byte {C,H,W} descriptor at ctx+0x190 + 12*tierIdx, where tierIdx = k for the encoder and 3-k for the decoder.

Caveat worth carrying forward: rbp+0x560 is a shared stack slot. Before the decoder tables are built, the same bytes are the gridDim out-param of __hipPopCallConfiguration (0x18001E2FD, 0x18001E48B) and the char buffer for the `swin%d_C%d` name formatter (0x18001EB1E).

SCHEDULE:
The H,W ladder is fully derived. Measured at colour 1920x1080, padded base 1920x1152 (padW = ceil(1920/128)*128 = 1920, padH = ceil(1080/128)*128 = 1152):

  stage                 C      H      W    grid ph0   grid ph1(-4,-4)   tensor bytes C*(H/4)*(W/4)*16
  block 0 / block 70    32   1152   1920   240 x 144  (fixed 240x144)   70,778,880   = C*H*W  exact
  enc tier0 / dec3      32    576    960   120 x 72   121 x 73          17,694,720   = C*H*W  exact
  enc tier1 / dec2      64    288    480    60 x 36    61 x 37           8,847,360   = C*H*W  exact
  enc tier2 / dec1     128    144    240    30 x 18    31 x 19           4,423,680   = C*H*W  exact
  enc tier3 / dec0     256     72    120    15 x 9     16 x 10           2,211,840   = C*H*W  exact
  bottleneck           512     36     60     8 x 5      8 x 5            1,105,920   = C*H*W  exact
  bottleneck          1024     18     30     4 x 3      5 x 3              458,752   != C*H*W (552,960) TRUNCATED

Rounding behaviour, step by step:
  frame -> padded : rounds UP to a multiple of 128 per axis (0x18001C850..0x18001C879). This is the only rounding in the chain.
  padded -> tier  : arithmetic right shift by k+1, i.e. signed division truncating toward zero (0x18001D115..0x18001D1F6). Never actually truncates, because 128 = 2^7 and the deepest shift is 6.
  tier -> grid    : ceil((extent - shift)/8) as signed (x+7)/8 (0x180022332..0x180022350). Exact at phase 0 for every tier that k_swin_var actually runs (0-3 plus full-res), since 128/2^4 = 8. Phases 1-3 add one half-window row and/or column, masked in-kernel.
  tier -> 4x4 blk : x/4 truncating. Exact everywhere except tier 5 (18/4 -> 4, 30/4 -> 7).

Design rationale, provable rather than guessed: padding to 2^7 is exactly what keeps the deepest tier reached by k_swin_var (tier 3, padded/16) a multiple of the window size 8. It is NOT enough to keep tier 5's 4x4 blocking exact; 2^8 would have been.

Stage order (recovered as a by-product, matches the weight schema term for term):
  block 0        C=32  full res 1152x1920  hand-rolled pre-block, flags 0x14, shift (0,0)
  blocks 1-4     C=32   576x960   enc tier0, shift phases [0,1,2,3]
  blocks 5-8     C=64   288x480   enc tier1, phases [0,1,2,3]
  blocks 9-14    C=128  144x240   enc tier2, phases [0,1,2,3,0,1]
  blocks 15-22   C=256   72x120   enc tier3, phases [0,1,2,3,0,1,2,3]
  blocks 23-30   C=512   36x60    bottleneck loop 0x18001ED33..0x18001EDA5, phase = (blk-23) mod 4
  blocks 31-38   C=1024  18

FINDINGS:
  [proven] The frame size is padded UP to a multiple of 128 in each axis before anything else. padded = ((x + 127) & ~127), i.e. ceil(x/128)*128 for x>=0.
     @ 0x180056B90 (constant), 0x180056B90->0x18001C850 movdqa xmm0,[180056B90h]; 0x18001C858 paddd xmm0,xmm11; 0x180
  [proven] The raw inputs to that padding are the colour staging dimensions: W = dword [0x180074858], H = dword [0x18007485C]. These are literally the two values printed as `colour %ux%u`.
     @ 0x180009924 mov eax,[180074858h]; 0x18000992A mov ecx,[18007485Ch]; 0x180009930 mov [rbp+30h],rax; 0x180009946
  [proven] The padded pair is stored at ctx+0x18 = H and ctx+0x1C = W, and those two words are what the debug dump prints as `padded %dx%d` (W first).
     @ 0x18001D10F mov [rsi+18h],ebp; 0x18001D112 mov [rsi+1Ch],ebx; printed at 0x18000D679 mov edx,[1800744F0h] -> [
  [proven] The tier ladder is one straight-line SSE block: six 12-byte {C,H,W} records at ctx+0x190, tier k has C = 32<<k and (H,W) = (padH >> (k+1), padW >> (k+1)). No ceil, no pad, no divisibility check - it is a signed shift (C-style divi
     @ 0x18001D115..0x18001D11D pack [H,W]; 0x18001D121/125/12A/12E psrad 1; 0x18001D133 mov [rsi+190h],20h; 0x18001D
  [proven] Descriptor +0 -> dispatcher arg2 (C, the template selector), +4 -> arg7 -> VarParams+0x18 (H), +8 -> arg8 -> VarParams+0x1C (W). Both dispatcher callers do it identically.
     @ encoder: 0x18001EA08/0x18001EA13 (rbx = ctx+0x190 + 12*idx), 0x18001EACA mov edx,[rbx], 0x18001EA8D mov eax,[r
  [proven] gridDim = ( (W - shiftX + 7)/8 , (H - shiftY + 7)/8 , 1 ) with signed truncating division, blockDim = (256,1,1). The grid is therefore the 8x8 window count, and for every tier the mod actually launches it is exact at phase 0.
     @ 0x180022332 sub edi,ebp; 0x180022334 lea eax,[rdi+7]; 0x180022337 add edi,0Eh; 0x18002233C cmovns edi,eax; 0x1
  [proven] There IS a stage at output resolution, and it is exactly blocks 0 and 70. Its VarParams H,W are ctx+0x18/ctx+0x1C = 1152,1920, and its grid is computed directly as (padW>>3, padH>>3, 1) = 240x144 without going through the dispatch
     @ grid: 0x18001E202 movq xmm0,[r10+18h] .. 0x18001E21F psrad xmm0,3 .. 0x18001E224 movq [rbp+438h] / 0x18001E22C
  [proven] 64 tokens per workgroup at every width, confirmed on both sides. Host workspace stride per workgroup is 1<<12/13/14/15 for C=32(false)/32(true)&64/128/256; device uses the identical shift; and the workgroup spatial origin steps by
     @ host: 0x18002236A..0x18002237D (C=32, shl 12 or 13), 0x180022394 (shl 0Dh, C=64), 0x1800223AC (shl 0Eh, C=128)
  [proven] Item 3's `v_cmp_gt_u32 0x800` premise is misattributed. Zero occurrences in any of the five k_swin_var kernels. In k_swin_1h_32_fp8 it bounds a flat WEIGHT-tile index, not a token count.
     @ dis_gfx1201.txt:24 `v_cmp_gt_u32_e32 vcc_lo, 0x800, v3` (kernel k_swin_1h_32_fp8, off=0); other occurrences at
  [proven] Tiers 4 and 5 (C=512, C=1024) exist in the descriptor table but never reach k_swin_var - the dispatcher bails for any C outside {32,64,128,256}. They are consumed by the bottleneck section instead, at H,W = padH/32,padW/32 = 36,60
     @ bail: 0x1800223FB add ebx,0FFFFFFE0h / 0x1800223FE rol ebx,1Bh / 0x180022401 cmp ebx,7 / 0x180022404 ja 0x1800

OPEN: Whether the tier-5 4x4-block truncation (18/4 -> 4, 30/4 -> 7, a 17% short allocation at ctx+0x248 and ctx+0x30C) is a live defect. No C=1024 kernel launch was traced, so I cannot say which extent those kernels actually walk. Padding to 256 instead of 128 would make it exact; 1080 -> 1280 and 1920 -> 2048. | Blocks 31-38 (C=1024) and 40-47 (C=512, the second half of the bottleneck) were not traced to launch sites. The 23-30 loop at 0x18001ED33 is proven; the rest of the bottleneck is not. | Two tensor layouts coexist inside the same k_swin_var kernel and I did not reconcile them: a 4x4-blocked index (originY/4)*(W/4)+(originX/4) built at dis_gfx1201.txt 61250-61277, and the [C/16][H][W][16] 
=========== ROUTE 3 ===========
SUMMARY: Route 2 is fully recovered. The orchestration function is 0x18001DDF0..0x1800207F5 (one function; HANDOFF3's 0x18001E000..0x18001F400 was a sub-range of it, and 0x18001DDF0 is not a separate function holding the hand-rolled C=32 sites - those are the pre- and post-block stages of the same network pass). It runs 71 stages, blocks 0..70, as: pre-block, four encoder tiers (2-deep nest, trip counts 4/4/6/8), an 8-block C=512 transformer, an 8-block C=1024 1-D transformer with a head and two repacks, a decoder upsample, an 8-block C=512 transformer, four decoder tiers (2-deep nest, 1+7/1+5/1+3/1+3), post-block. Two of HANDOFF3's structural claims were wrong and I corrected them from the code: rbp+0x560 is an inline array of four 40-byte decoder records (proven by the EH array-destructor's -0x28 stride at 0x180020B3C), not a vector header; and ctx+0x1D8 is not an array of descriptors whose fir

RECORD LAYOUT:
Two different record types, both inline arrays of 4, both proven by their EH array-destructor funclets (which give the stride) and by the loop reads (which give the fields).

ENCODER record - 32 bytes, array base rbp+0x4B0, construction cursor rbp+0x558:
  +0x00  i32   firstBlock      written packed with +0x04 as one qword: 0x18001E7E1 (0x0000000400000001), 0x18001E84E (0x0000000800000005), 0x18001E8A9 (0x0000000E00000009), 0x18001E91B (0x0000001600000015)
  +0x04  i32   lastBlock       INCLUSIVE (loop is `cmp r14d,r12d; jg exit` at 0x18001EA4A); read as [r8+4] @0x18001EA46, [r8] @0x18001EA7B
  +0x08  ptr   shiftVec._Myfirst   0x18001E821 / 0x18001E883 / 0x18001E8DE / 0x18001E973 (from operator new of 0x10,0x10,0x18,0x20 bytes)
  +0x10  ptr   shiftVec._Mylast    0x18001E840 / 0x18001E89B / 0x18001E90D / 0x18001E99D
  +0x18  ptr   shiftVec._Myend     0x18001E82F / 0x18001E891 / 0x18001E8EC / 0x18001E981
  read of the shift array: `mov rcx,[r8+8]; mov r15d,[rcx+rax*4]` @0x18001EA85, rax = blk - firstBlock
  stride proof: funclet 0x180020B85, `mov rsi,[rbp+558h]; lea rdi,[rbp+4B0h]; add rsi,0FFFFFFFFFFFFFFE0h` (-32)

DECODER record - 40 bytes, array base rbp+0x560, construction cursor rbp+0x550:
  +0x00  i32   firstBlock      packed qword with +0x04: 0x18001FA0C (0x0000003700000030), 0x18001FA8E (0x0000003D00000038), 0x18001FB01 (0x000000410000003E), 0x18001FB77 (0x0000004500000042)
  +0x04  i32   lastBlock       INCLUSIVE; the pre-loop launch handles firstBlock, the loop handles firstBlock+1..lastBlock (`cmp eax,edi; jge exit` @0x18001FD76)
  +0x08  i32   firstShiftIdx   0x18001FA13 (0), 0x18001FA95 (2), 0x18001FB08 (0), 0x18001FB7E (0); read at 0x18001FC71 as [rbp+rdi*8+568h] -> arg9 of the pre-loop launch
  +0x0C  i32   padding         never written, never read
  +0x10  ptr   shiftVec._Myfirst   0x18001FA4C / 0x18001FAC3 / 0x18001FB36 / 0x18001FBAC (operator new of 0x1C,0x14,0x0C,0x0C bytes)
  +0x18  ptr   shiftVec._Mylast    0x18001FA76 / 0x18001FAE9 / 0x18001FB5F / 0x18001FBCB
  +0x20  ptr   shiftVec._Myend     0x18001FA5A / 0x18001FAD1 / 0x18001FB44 / 0x18001FBBA
  read of the shift array: `mov rcx,[r13+10h]; mov esi,[rcx+rax*4]` @0x18001FD87, rax = blk-1 - firstBlock
  stride proof: funclet 0x180020AB5, `mov rsi,[rbp+550h]; lea rax,[rbp+560h]` and `add rsi,0FFFFFFFFFFFFFFD8h` (-40) @0x180020B3C, with the vector freed from rsi-0x18 = record+0x10

NEITHER contains C, H or W. Those come from a SEPARATE 6-entry table in the context:

CONTEXT TIER DESCRIPTO

SCHEDULE:
Full recovered schedule. 71 stages, blocks 0..70. H,W shown for 1920x1080 output (ctx+0x18 = roundUp(1080,128) = 1152, ctx+0x1C = 1920); the general rule is tier k has H/2^(k+1), W/2^(k+1).

  #   block(s)  kernel(s)                          C     H     W    shiftIdx      flags
  --  --------  ---------------------------------  ----  ----  ---  ------------  -----
  1   0         k_swin_var<32,true>  @0x18001E4CA    32  1152  1920  0 (hard 0)    0x14  pre-block
                (or k_pre_block_1h_32_fp8 @0x18001E33C if DLSSNR_SLOW_PREPOST)
  2   1-4       k_swin_var<32,false> via dispatcher  32   576   960  0,1,2,3       1 / 0 / 0 / 4
  3   5-8       k_swin_var<64,false>                 64   288   480  0,1,2,3       1 / 0 / 0 / 4
  4   9-14      k_swin_var<128,false>               128   144   240  0,1,2,3,0,1   1 / 0 0 0 0 / 4
  5   15-22     k_swin_var<256,false>               256    72   120  0,1,2,3,0,1,2,3   1 / 0.. / 4
  6   23-30     [k_ffwd2, k_conv_res2, k_qkv_attn2, 512    36    60  (b-23) mod 4  n/a (own runner)
                 k_conv_res2 | k_conv_res_views]
  7   31-38     [k_expand2, k_contract2, k_qkv2,   1024    18    30  n/a (1-D)     n/a
                 k_attention2, k_contract2]              tokens = roundUp(18*30,64) = 576
      (interleaved: k_final_head @0x18001EF4F using block30.layer4.layer, then
       k_repack @0x18001F0AA (mode 1) BEFORE the loop; k_repack @0x18001F816 (mode 0) AFTER it)
  8   39        k_dec_upsample @0x18001F94F         512    36    60  n/a           n/a
  9   40-47     same 4-kernel C=512 runner          512    36    60  (b-40) mod 4  n/a
 10   48-55     k_swin_var<256,false>               256    72   120  0,1,2,3,0,1,2,3   8 / 0.. / 2
 11   56-61     k_swin_var<128,false>               128   144   240  2,3,0,1,2,3   8 / 0 0 0 0 / 2
 12   62-65     k_swin_var<64,false>                 64   288   480  0,1,2,3       8 / 0 0 / 2
 13   66-69     k_swin_var<32,true>                  32   576   960  0,1,2,3       8 / 0 0 / 2
 14   70        k_swin_var<32,true>  @0x180020416    32  1152  1920  0 (hard 0)    0x20  post-block
                (or k_post_block_1h_32_fp8 @0x18002029F if DLSSNR_SLOW_PREPOST)

Flags column reads "first / middle / last" within the tier. Encoder: first=1, last=4 (`lea r8d,[rdx+rcx*4]` @0x18001EAC6). Decoder: first=8 (constant, 0x18001FCC3), last=2 (`add r8d,r8d` @0x18001FDB2). All three encodings are the same three bits.

The shift index is per BLOCK and cycles 0,1,2,3, so consecutive blocks 

FINDINGS:
  [proven] The orchestration function is 0x18001DDF0..0x1800207F5 (one function), not 0x18001E000..0x18001F400. 8 pushes + `sub rsp,778h` + `lea rbp,[rsp+80h]`, single `ret`, cold tails to 0x1800208DD, then EH funclets from 0x1800208F0.
     @ prologue 0x18001DDF0-0x18001DE03; epilogue 0x1800207CC-0x1800207F5; `ud2` 0x1800207F6; funclet boundary 0x1800
  [proven] rbp+0x4B0 is an INLINE ARRAY of 4 x 32-byte encoder stage records; rbp+0x560 is an INLINE ARRAY of 4 x 40-byte decoder stage records. Neither is a vector header (HANDOFF3's caution resolved). Each record embeds a std::vector<int> 
     @ array dtor funclets 0x180020B85 (stride 0x20, base rbp+0x4B0, cursor rbp+0x558) and 0x180020AB5/0x180020B3C (`
  [proven] ctx+0x190 is an array of SIX 12-byte tier descriptors {i32 C, i32 H, i32 W} with C = 32,64,128,256,512,1024. It ends exactly at ctx+0x1D8.
     @ 0x18001D133, 0x18001D160, 0x18001D184, 0x18001D1A8, 0x18001D1CC, 0x18001D1EC (the six `mov dword ptr [rsi+..],
  [proven] HANDOFF3 is WRONG that the first u32 of a ctx+0x1D8 descriptor becomes dispatcher arg2. At 0x18001EACA rbx is the ctx+0x190 tier descriptor, not the ctx+0x1D8 entry. ctx+0x1D8 holds four RAW DEVICE POINTERS (encoder skip buffers) 
     @ 0x18001EA08-0x18001EA13 (`lea rcx,[rdx+rdx*2]` with rdx=[rbp+498h]=ctx+0x190; `lea rbx,[rdx+rcx*4]`) vs 0x1800
  [proven] ctx+0x1D8[i] (i=0..3) is allocated by hipMalloc of C_i*H_i*W_i bytes and zeroed, inside a 4-iteration loop over the tier table.
     @ alloc 0x18001D528 (hipMalloc), memset 0x18001D537, store 0x18001D570 (`mov [r15+FFFFFFFFFFFFFEF0h],rax` with r
  [proven] The shift index advances PER BLOCK, from a per-tier precomputed int array, cycling 0,1,2,3 through the four-phase table. Consecutive blocks therefore alternate window phase - the defining shifted-window behaviour.
     @ encoder: 0x18001EA85-0x18001EA89 (`mov rcx,[r8+8]; mov r15d,[rcx+rax*4]` with rax = blk-tier.start) then `mov 
  [proven] The C=512 blocks also take a shift index, computed inline as (blockIdx-base) mod 4, not from a table.
     @ 0x18001ED61-0x18001ED77 (blocks 23-30) and 0x18001F9A1-0x18001F9B3 (blocks 40-47): `lea ecx,[rdi-17h]/[rdi-28h
  [proven] Flag bits, consistently named across all three dispatcher call sites: bit0=first-block-of-encoder-tier, bit1=last-block-of-decoder-tier, bit2=last-block-of-encoder-tier, bit3=first-block-of-decoder-tier (and at C=32 selects the <3
     @ encoder 0x18001EAC6 `lea r8d,[rdx+rcx*4]` (dl=isFirst @0x18001EAB4, cl=isLast @0x18001EAAC); decoder inner 0x1
  [proven] The encoder is a 2-deep nest: outer tier loop of 4, inner block loop with trip counts 4,4,6,8 over blocks 1-4, 5-8, 9-14, 15-22 (inclusive both ends).
     @ outer 0x18001E9B2 (`xor ecx,ecx; cmp ecx,4; jae 0x18001ECBF`) / 0x18001ECAD-0x18001ECB9; inner 0x18001EA4A (`c
  [proven] The decoder is a 2-deep nest: outer tier loop of 4 (indexing the tier table in REVERSE via 3-r9), inner block loop with trip counts 7,5,3,3, plus one pre-loop launch per tier. Blocks 48-55, 56-61, 62-65, 66-69.
     @ outer 0x18001FBE7 (`xor r9d,r9d; cmp r9d,4; jae`) / 0x18001FE8D-0x18001FEA8; inner 0x18001FD76 (`cmp eax,edi; 

OPEN: The exact within-block kernel ordering of 0x180022B60 for non-zero VIT512_OLD bit combinations. MSVC interleaved the alternate parameter-construction blocks and I only linearised the default (global==0) path with confidence. Bits 1, 2, 3 clearly select k_conv_res_views-extra / k_qkv_attn / a k_conv_res_views tail, but I did not fully verify the register conventions on every cross-edge. | What arg5 ([rsp+3B0h], loaded into r14 at 0x18002338D) does inside 0x180022B60. Loop 2 passes ctx[0x2A0] there on block 47 and loop 1 passes 0 always; I traced arg6 (r15, the conv_res2/conv_res_views selector) but not arg5's consumer. | Why blocks 31-38 have a layer3 blob that no code path fetches. Candidate

=========== VERDICTS ===========
[REFUTED] Inside the single function 0x18001C740 (its only caller is 0x18000CEBD), the two spatial extents passed as stack args 5 and 6 are each rounded UP to a multiple of 128, and the result is used for the working-buffer extent and for the first kernel launch only — the unpadded values remain live and driv
[HELD] CONFIRMED, with the justification strengthened from printf-order to GetDesc-derived.

W = dword [0x180074858] and H = dword [0x18007485C] are the colour staging resource's width and height, and they are the raw (unpadded) inputs to the 128-alignment in 0x18001C740.

Origin (this is the load-bearing 
[HELD] `rbp+0x560` in function 0x18001DDF0 is a REUSED stack slot, and at 0x18001FC6A it holds a 4-element inline array of 40-byte records. Confirmed, with two corrections to the supporting argument.

FRAME (proven): 0x18001DDFC `sub rsp,778h`; 0x18001DE03 `lea rbp,[rsp+80h]`. PE .pdata entry `00000F3C 000
[REFUTED] The record-driven part of the schedule is static, but the schedule as a whole is not, and the records are two distinct types.

Within the single orchestration function at 0x18001DDF0 (rbp fixed at `0x18001DE03 lea rbp,[rsp+80h]`, never rewritten), there are two unconditional, fully literal construct
[REFUTED] REFUTED AS STATED; the underlying two-record-type finding is real. Corrected version, every field with the instruction that produces it. All addresses are VA (image base 0x180000000) and all live in ONE function, VA 0x18001DDF0..0x1800208E2 (single .pdata entry `0001DDF0 000208E2`, dumpbin /unwindin
[HELD] The claim stands. Sharpened, with each field tied to a use site rather than a value:

```c
struct DecodeStage {           // 40 bytes, stride read at 0x18001FA34/FA7D/FAF0/FB66 and
  int32_t firstBlock;          // +0x00  confirmed -0x180020B3C `add rsi,-28h`
  int32_t lastBlock;           // +0x04 
[REFUTED] There is a 12-byte-stride, inline 3xu32 descriptor array at ctx+0x190 (ctx = the dispatcher's arg1). Its three fields route to the dispatcher as:

  desc+0 -> arg2  (rdx)      -> channel-width selector
  desc+4 -> arg7  ([rsp+30h]) -> VarParams+0x18 = H
  desc+8 -> arg8  ([rsp+38h]) -> VarParams+0x1
[HELD] The padded pair is stored at ctx+0x18 = padded HEIGHT and ctx+0x1C = padded WIDTH, where ctx is the global at 0x1800744D8, so ctx+0x18 = 0x1800744F0 and ctx+0x1C = 0x1800744F4; the debug dump prints them as `padded <W>x<H>`, width first.

Storage. The writes are `mov [rsi+18h],ebp` @0x18001D10F and 
[HELD] Field -> VarParams mapping (dispatcher path only), all offsets and strides read from instructions.

Descriptor array: ctx+0x190, 12-byte stride, three u32. Base written once at 0x18001E95E (`mov [rbp+498h],rcx`, rcx = ctx+0x190; only other writer in the image is 0x18001162B, a different function). S