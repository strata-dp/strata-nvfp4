# DP two-GPU prompt path for strata-nvfp4 (unofficial)

This branch (`dp-prompt-path`) is [sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) `v0.1.44.1-nvfp4.2`
with the dual data-parallel prompt path from [nicejunjie/Strata-rdna3](https://github.com/nicejunjie/Strata-rdna3)
ported onto it. It is not an official release of either project.

## What DP does

A normal two-GPU Strata run is a layer split: card 0 runs layers 0-23, card 1 runs 24-47, and each prompt chunk passes
through card 0 and then card 1. With `STRATA_DP_PREFILL=1` both cards also hold the other half's dense weights and state,
take prompt chunks in turn and run all 48 layers each, handing each layer's state to the other card. It applies to prompts
of 8,000 tokens and more; decode is unchanged.

## Commits on top of v0.1.44.1-nvfp4.2

| Commit | From | What |
|---|---|---|
| `9c9e93bc` | nicejunjie f60e317b | DP prompt path. Five conflicts merged by hand: `prefill.cpp` takes nicejunjie's `step_at` / `next_mine` chunk walk; `generate.cpp` keeps sergqwer's pipeline chunk sizing (`pipe_chunk`) with the DP override after it |
| `dc0e6472` | nicejunjie 7ee3bc4f | `STRATA_DP_MIN` default 8000 |
| `3abe2d9a` | nicejunjie 90ea76cd | a segment the DP plan refuses lends the pipeline's chunk; `STRATA_PIPE_HALF_MIN/MAX` |
| `9555eeea` | nicejunjie 49a5eebb | `dp_takes` checks the DP plan's real last chunk and the head before the first K/V block boundary |
| `f85aa37b` | port | the chunk loop's increment and the last-row index follow `step_at` (from nicejunjie 012ce902, which is not otherwise ported) |

Not ported: `STRATA_DP_SHARE` (a5cb144b, no gain measured), early/late refill (866c1544, 829e660f: they depend on
nicejunjie's tensor-parallel code), the tensor-parallel commits.

## Build and run

```bash
docker build --build-arg CUDA_ARCHITECTURES=86 --build-arg BUILD_VISION=1 -t strata-nvfp4:dp .
```

Run as sergqwer's Docker image, plus `STRATA_DP_PREFILL=1 STRATA_KV_HOST_DMA=1 STRATA_PREFILL_AUTO_MAX=16384`
(environment or the config's `"env"`). Requirements: exactly two GPUs in a layer split, streamed K/V (`--kv-resident`,
which setup writes from a 64K context when the RAM allows; not under WSL), and no `--batch`, `--peer-device` or
`--trim-stage-weights`. The engine log shows `strata serve: DP prompt path linked (chunk up to 16384 tokens)`, or
`STRATA_DP_PREFILL is off (...)` with the condition that failed.

## Measured

2× RTX 3090 24 GB (PCIe 4.0 x16, P2P on), 32-core EPYC (AVX2), 256 GB DDR4, Ubuntu 24.04, CUDA 13. Swift 1.5
Qwen3.8-Flash-Next IQ4_XS, 512K context (YaRN ×2), int8 KV, `--kv-resident 32768`, `--pipeline-windows 2`,
`--pcie-frac 0`, `--spec 4` + MTP. Every engine ran the same config and expert profile. Prefill = fresh prompts with a
unique salt, `max_tokens` 1; decode = greedy, thinking off, 1,024 tokens.

Prefill (tok/s, both cards at 350 W):

| Engine | 19K | 43K | 89K | 123K |
|---|---|---|---|---|
| **This branch, DP on** | **4,719** | **5,354** | **5,705** | **5,592** |
| nicejunjie/Strata-rdna3 + DP (CUDA build) | 4,493 | 5,046 | 5,481 | 5,418 |
| Niko1221/Strata `main` @ 46c152a1 (engine 0.1.53), no DP | 2,567 | 3,885 | 4,611 | 4,737 |

Decode at 350 W (tok/s; zh prose / zh chat / EN technical / code): this branch 118 / 130 / 127 / 209; nicejunjie + DP
121 / 125 / 118 / 199; upstream `main` 127 / 131 / 126 / 188.

One RTX 3090 at 350 W (DP does not apply; this is sergqwer's engine): prefill 2,414 / 2,966 / 2,921 / 2,864 and decode
110 / 113 / 109 / 163.

Correctness: needles at ~89K and ~123K (60 % depth) pass; a tool call and an image request pass.
Longer needles (~206K to ~520K) were run on nicejunjie's fork only.

## Notes

- With DP each card holds both halves' dense weights, so it caches about 300 fewer experts than nicejunjie's fork on
  24 GB cards; one "only 435 MiB of VRAM free" warning appeared after an image request.
- Without P2P (stock GeForce drivers) the per-layer state hand-off goes through host memory; not measured here.

## License and credits

MIT, as the projects it is built from (see `LICENSE`). The DP prompt path is nicejunjie's work; the NVIDIA engine is
sergqwer's; Strata is Niko1221's and its contributors'.
