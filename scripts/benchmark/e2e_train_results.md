# AF3 end-to-end TRAINING: NestedTensor (ragged) vs padded-dense

Full training step = `model(batch)` → `CompositeLoss` (single-step EDM denoising + distogram + confidence)
→ `loss.total.backward()`. Real structures, fp32, node8 (H100 80GB). Config: recycling=1, pairformer=2,
confidence=1, denoiser 1/1/1, diffusion num_samples=4, rollout=2.

Trunk NT==per-element parity is bitwise-tight (single|d|≈6.6e-6, pair|d|≈3.8e-6); both paths give a finite
loss and full gradients (NT grad_params=342, padded=394 — padded reaches a few extra params via padded slots).

## Real sabdab (`mmcif_sabdab_all`, crop 384; token range [160,384], median 384, atoms ≤3170)

```
bsz  waste |     NT fwd      pad fwd  fwd spd/mem |      NT f+b        pad f+b  f+b spd/mem
  2  1.54x |  299ms/3865M  261ms/6455M  0.87x/0.60 |  688ms/5502M  1823ms/14675M  2.65x/0.37
  4  1.26x | 579ms/13839M  496ms/12513M 0.86x/1.11 | 1619ms/18113M 3579ms/29026M  2.21x/0.62
  8  1.16x | 1247ms/53708M 974ms/24626M 0.78x/2.18 |          OOM  7054ms/57734M       -
 16  1.12x |          OOM  1934ms/48867M      -     |          OOM           OOM        -
```
(`spd` = padded_time / NT_time, >1 means NT faster; `mem` = NT_peak / padded_peak, <1 means NT lighter.)

## Reading the result

- **Training (fwd+bwd) is the real metric, and NT wins decisively at realistic batch sizes**: 2.2–2.65× faster
  and 0.37–0.62× peak memory at bsz 2–4. The win is in the **backward**: padded computes (and stores) gradients
  for every padded position; NT does only real atoms/tokens. NT bwd ≈ 0.39s vs padded bwd ≈ 1.56s at bsz 2.
- **Forward alone is ~0.85× (NT marginally slower)**: per-element NT overhead (windowing/denoiser packing
  bridges, per-element cdist, the confidence head's per-sample Python loop) is not yet amortized by the modest
  forward FLOP saving at crop-384 waste (~1.1–1.5×, compressed because the 384 crop caps most antibodies at max).
- **bsz≥8 NT memory overhead / OOM**: the per-element bridge + per-sample-confidence intermediates inflate the
  retained autograd graph past the clean padded-dense graph once waste is low and structures are large. This is
  the main NT scaling limitation to optimize (free/checkpoint bridge intermediates; the confidence head).

## Tiny-structure control (`data/debug/pdb_cifs`, token range [4,48], waste ~1.3×)

NT is 2–5× slower on forward and ~0.6–0.9× memory: with tiny structures the per-element Python overhead dwarfs
the compute. NT needs real structure size + size variance (i.e. real padding waste) to win — exactly the sabdab
training regime above.

## CORRECTION — TRUE AF3 config (368M, 48 pairformer, recycle 4, 24-block diffusion token-transformer)

The proxy above (2 pairformer blocks, num_samples=4, 2 timed steps) was ~1/24th the model and badly
under-sampled; its "NT 2.2-2.65x faster" was an ARTIFACT. With the true architecture (bf16, real sabdab crop
256, diffusion_samples=8 — reduced from 48 only to fit 80GB; sample count scales both paths equally), **16 timed
steps**:

```
bsz  waste |     NT fwd       pad fwd  fwd spd/mem |      NT f+b        pad f+b  f+b spd/mem
  2  1.39x | 4139ms/31477M 3072ms/26471M 0.74x/1.19 | 6465ms/34386M 5239ms/26560M  0.81x/1.29
  4  1.22x |          OOM  4768ms/49390M     -       |          OOM  8851ms/51492M       -
```
NT is SLOWER (0.74-0.81x) and uses MORE memory (1.19-1.29x), and OOMs at bsz=4 where padded fits. A historical
one-off phase profile (bsz=2, two 256-token structs) measured: trunk NT 1.53x slower / 0.94x mem; diff_denoise (grad)
2.02x slower / **1.76x mem**; diff_rollout (no_grad) 1.13x slower / **0.20x mem**.

**Why NT loses here:**
1. **Fixed crop kills token-length variance.** Antibodies cropped to 256 cluster AT 256, so the trunk (the bulk
   of compute, token×token pair) has ~no padding waste to skip; the fused segmented/varlen Triton kernels are
   then pure overhead vs cuBLAS dense batched matmul (1.53x). NT's trunk win needs structures meaningfully
   SHORTER than the crop.
2. **Diffusion per-element bridge + its autograd retention.** The diffusion atom path runs per-element for NT
   (not fused like the trunk). With gradients it retains duplicated bridge intermediates -> denoise 1.76x memory
   (the bsz=4 OOM driver); the SAME path without grad (rollout) uses 0.20x memory, proving the cost is autograd
   retention of the bridge, not the packing itself.
3. Confidence head loops over samples in Python (another per-element cost; not measured — uniform-token unbind
   edge bug).

NT only wins at HIGH token-length variance (cf. the atom-path-only bench at crop 640). For the true e2e model at
a realistic fixed crop, NT currently loses. Levers: fuse the diffusion atom path (kill the per-element bridge) +
free/checkpoint bridge intermediates; or accept NT only for high-variance (no-crop / length-bucketed) batches.

## HEADROOM — is NT worth pushing? (trunk = fused ceiling, real crop 640)

The trunk is already fully fused (segmented/varlen OOps kernels, no per-element bridge), so its NT-vs-padded
ratio is the CEILING the full model could reach if the diffusion/confidence bridges were also fused. Measured
fwd+bwd, bf16, 12 blocks (per-block ratio is depth-invariant):

```
waste   NT speedup   NT mem      (speedup = pad_time/NT_time; >1 = NT faster)
1.00x   0.51-0.53x   0.46-0.49x  <- real random batches at crop 640 sit HERE
1.03x   0.53x        0.46x
1.11x   0.56x        0.43x
1.95x   0.94x        0.27x       <- ~break-even
3.49x   1.73x        0.15x       <- NT wins
5.07x   (pad OOM)    NT fits     <- NT wins big
```

Real sabdab token distribution at crop 640: median=640, p25=535 — **75%+ of structures exceed 640 and crop to
the cap**, so real random batches have token-pair waste only **1.0-1.17x**.

**Verdict — SPEED: ~zero headroom at realistic training.** Even with the diffusion bridge perfectly fused, NT
stays ~**1.8-2x SLOWER**, because the fused trunk itself is ~1.9x slower at the ~1.1x waste of real cropped
batches. The limiter is a **kernel-efficiency gap**: the segmented/varlen Triton kernels are ~1.9x slower
per-FLOP than cuBLAS dense batched matmul, so NT needs **waste ≥ ~2x just to break even** — and cropped training
never gets there (everything clusters at the crop). NT-speed only wins on **uncropped / length-bucketed**
batches (waste ≥ 2x). Fixing the per-element bridge does NOT change this — the trunk dominates and is already
as-NT-favorable as it gets.

**MEMORY: real ~2x headroom (the only win).** The trunk uses ~0.46-0.49x memory at real waste (and far less at
variance); fixing the diffusion bridge (today's OOM driver) would make the whole model ~0.5x memory → enables
~2x larger batch/crop. That's a capacity lever, not a per-step speedup, and only matters if memory-bound
(training is batch_size=1 today).

Bottom line: not worth pushing NT for AF3 training speed at a fixed crop. The honest path to NT speedups would
be (a) closing the Triton-vs-cuBLAS gap (huge kernel effort) or (b) length-bucketed/uncropped batching (changes
the data pipeline). The e2e NT path stays valuable as a correct, ~2x-lighter-memory option if capacity is ever
the bottleneck.
