# Figure 1a: Hybrid cache accounting and allocator boundary

Create one publication-grade scientific vector schematic for a two-column
MLSys paper. Landscape aspect ratio about 1.65:1. White background, flat vector
design, no gradients, shadows, 3D, device clip-art, or decorative cards. Use
Arial/Helvetica-like sans serif type and thin orthogonal connectors. Add a
small bold lowercase panel label `a` at top left.

## Visual story, left to right

1. **Qwen3.5 hybrid layer stack**. Show a compact repeated strip with many blue
   `GDN` cells and a few orange `GQA` cells. The repetition should read as a
   real layer topology, not as two generic boxes.
2. Branch into two persistent cache stores:
   - Orange **Attention KV** lane from GQA: `per-token cost A x L`, with two
     small dtype tags `fp16` and `int4`.
   - Blue **Recurrent state** lane from GDN: `fixed per-sequence cost G`, split
     into `bf16 convolution state` plus `temporal state: fp32 or bf16`.
3. Merge both lanes into a shared outlined **vLLM GPU cache byte ledger**. Show
   the exact continuous accounting `bytes / sequence = A L + G`; use an orange
   variable-width segment for `A L` and a blue fixed-width segment for `G`.
4. Feed the ledger into a compact **discrete allocator** glyph made of aligned
   page/block tiles. Label the output `allocator token capacity` and
   `allocator-equivalent sequence slots`.
5. End with a narrow neutral scope strip, separated by a vertical rule:
   `Measured: cache allocation` above `Not measured: scheduler admission or SLO completion`.

## Semantic palette

- Ink: `#20252B`
- GDN / recurrent state: `#2A6FBB`, pale fill `#DCEAF7`
- GQA / attention KV: `#E58B2A`, pale fill `#F9E6CC`
- Neutral structure: `#59636E`, pale fill `#F3F5F7`
- Scope warning only: `#B94A3A`, very pale fill `#F8E4E0`

## Scientific restrictions

Do not draw request queues, simultaneous users, throughput arrows, speedup,
latency reduction, quality preservation, or a scheduler. Do not call any value
concurrency. This is byte accounting and allocator capacity only.

