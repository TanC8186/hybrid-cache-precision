# Figure 2a: Byte-to-page-to-block mechanism

Create one publication-grade scientific vector mechanism diagram for a
two-column MLSys paper. Landscape aspect ratio about 1.8:1, white background,
flat vectors, no shadows, gradients, 3D, icons, or decorative cards. Use a
small bold lowercase panel label `a` at top left.

Use the verified Qwen3.5-2B, int4-KV, 4K-context worked example. Arrange four
aligned stages left to right, using proportional tile counts only where noted:

1. **Per-layer bytes**
   - Orange attention tile: `int4 KV = 528 B / token / GQA layer`.
   - Blue recurrent tile: `fp32 temporal + bf16 conv = 1,085,440 B / GDN layer`.
   - Blue recurrent tile below: `bf16 temporal + bf16 conv = 561,152 B / GDN layer`.
   - Direct label: `recurrent bytes -48.30% (not exactly half)`.
2. **Padded recurrent page**
   - fp32 row: `1,089,792 B`, block size `B = 2,064 tokens`.
   - bf16 row: `566,016 B`, block size `B = 1,072 tokens`.
   Show alignment/padding as a thin hatched neutral tail, not as unexplained
   empty space.
3. **Shared fixed GPU pool**
   - fp32 row: `K = 3,287 allocated blocks`.
   - bf16 row: `K = 6,330 allocated blocks`.
   Draw more small blocks in the bf16 row, but include an ellipsis so the count
   is symbolic rather than literally thousands of tiles.
4. **Discrete capacity at L = 4,096**
   Show the equation once: `S_alloc = K / (H + ceil(L/B)), H = 3`.
   - fp32: `3,287 / (3 + 2) = 657.4 allocator-equivalent slots`.
   - bf16: `6,330 / (3 + 4) = 904.3 allocator-equivalent slots`.
   Under both results, add a neutral boundary label:
   `These are allocator-equivalent slots, not concurrent requests.`

Use orange `#E58B2A` for KV, blue `#2A6FBB` for recurrent state, dark neutral
`#4E5965` for fp32/reference, pale blue `#DCEAF7`, pale orange `#F9E6CC`, and
ink `#20252B`. Use thin orthogonal arrows and direct labels. Do not invent or
imply scheduler behavior, request admission, throughput, or latency.

