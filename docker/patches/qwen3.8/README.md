# Regular-lane Qwen3.8 patches

These frozen source patches are applied before wheel compilation. The `series`
files define application order. Builds do not fetch moving PR heads for this
stack; refresh the files deliberately when updating it.

The vLLM series follows the existing preset selection: ordinary upstream source
builds include it; custom refs, local sources, and explicit PR selections skip it
unless `--apply-preset-vllm-prs` is requested. The FlashInfer series is enabled
for the regular upstream lane with the default FlashInfer ref and no explicit
FlashInfer PRs. Custom FlashInfer refs/PRs skip it.

Both series also check the source repository before applying anything. Neither
can apply to `local-inference-lab/vllm` or `local-inference-lab/flashinfer`, even
if its enable flag is accidentally set. The B12X build script disables both.
Existing explicit `--apply-vllm-pr` and `--apply-flashinfer-pr` behavior remains
available independently.

## Included changes

The vLLM order is:

1. [#59958](https://github.com/vllm-project/vllm/pull/59958), including #56273 and #58815: NVFP4 PLE tables.
2. [#59973](https://github.com/vllm-project/vllm/pull/59973): quantized draft heads.
3. [#59740](https://github.com/vllm-project/vllm/pull/59740) and [#60226](https://github.com/vllm-project/vllm/pull/60226): sparse draft vocabulary, including quantized heads.
4. [#60068](https://github.com/vllm-project/vllm/pull/60068): adaptive draft depth.
5. [#54912](https://github.com/vllm-project/vllm/pull/54912): QSA ring sizing.
6. [#50897](https://github.com/vllm-project/vllm/pull/50897): lookahead-aware prefix hashing.
7. [#54614](https://github.com/vllm-project/vllm/pull/54614): dynamic NVFP4 linear kernels.
8. [#54617](https://github.com/vllm-project/vllm/pull/54617): dynamic NVFP4 MoE.
9. [#60252](https://github.com/vllm-project/vllm/pull/60252): DFlash2 draft cache block sizing.
10. [#59021](https://github.com/vllm-project/vllm/pull/59021) and [#59025](https://github.com/vllm-project/vllm/pull/59025): separate unquantized linear selection and cuDNN BF16 GEMMs.
11. The supplied 32,768-token Flash Next draft vocabulary.
12. [#60646](https://github.com/vllm-project/vllm/pull/60646): CSF-compressed NVFP4 scale loading.
13. Wheel package-data inclusion for the draft vocabulary.

[#55601](https://github.com/vllm-project/vllm/pull/55601) is already in the
validated upstream base. [#59244](https://github.com/vllm-project/vllm/pull/59244)
modified a BF16-only warmup workaround that upstream
[#60434](https://github.com/vllm-project/vllm/pull/60434) subsequently removed.
The series retains the current full-model autotune path, including cuDNN, and
does not restore the obsolete 32-token cap or BF16-only pass.

The FlashInfer order is
[#5863](https://github.com/flashinfer-ai/flashinfer/pull/5863) (canonical NVFP4
W4A4 MoE), then [#5943](https://github.com/flashinfer-ai/flashinfer/pull/5943)
(SM12x CuTe DSL MXFP8 GEMMs). #5943 includes head
`15c5c8d21388c118558c8370299204aa4b110ced`.

## Integration and refresh

On October 11, #59973 and #50897 were rebased after two upstream changes
landed between validation and the nightly build. Their PR revisions are
unchanged; the integration diffs now preserve:

- [#60937](https://github.com/vllm-project/vllm/pull/60937)'s native MRV2 weight
  reload implementation and cache resets, with quantized draft heads refreshed
  through `reset_lora_state` after target reloads and weight transfers.
- [#61012](https://github.com/vllm-project/vllm/pull/61012)'s shared
  `replay_boundaries` helper. Core caching and Mooncake both disable EAGLE
  back-off when lookahead hashes already prove the boundary, and retain the
  upstream sliding-window tail fix when EAGLE block dropping applies.

The complete 14-patch vLLM series and all 17 subsequent source patchers pass on
the failed nightly base `2339bd39d661965b821bfbe652db8f536e0b3dcd` and current main
`b99fdbe1f896abd7f4b133685a37397c3427f7d5`. All 117 changed Python files parse on
both bases, and vocabulary packaging is intact. The 9 series tests (including
CPU regressions for native reload and shared replay boundaries), 28 SWA tests,
and 100 build-selection checks pass. GPU compilation and inference remain for
nightly CI. Validation applies to these commits; builds that track moving main
can encounter later upstream conflicts.

All 16 patches were refreshed on October 10, 2026 against upstream vLLM
`3c062d391d15cfa41d3f321cad24f3030c892a97` and FlashInfer
`34bb66e4d0b420b5a03c99b2b6a1c0831dfaa738`. Each patch header records the full
upstream PR revision(s) and integration base, including bundled dependencies.
The vocabulary and package-data patches remain local additions. These are
validation bases; normal builds still follow their selected refs.

The refresh includes:

- #56273's review fixes, which were missing from #59958's bundled dependency:
  dummy-load handling, byte-based E4M3 scale decoding, 64-bit lookup row indices,
  pinned-row sizing, and an explicit ROCm rejection. The shared NVFP4 row decoder
  serves resident, pinned, and file-gather storage; dummy file-gather tables still
  bind their staging buffers even though checkpoint validation is skipped.
- #50897 at `5f36a6cd20f83e660876f432facbf2ed07a998e7`, including updated Mamba
  prefix-cache expectations, spawned MRV2 tests, and missing test imports/mock
  arguments. The allocation-time cache fix and Mooncake integration below remain.
- #5943's autotuner override fix, preserving the selected timer and first-tactic
  margin, plus parallel SM12x MXFP8 compilation and queued bucket preparation.
- A refreshed #54614 test hunk preserving both its bounded-logits test and the
  autotune cache-key test added by upstream
  [#60085](https://github.com/vllm-project/vllm/pull/60085). The overlapping
  insertion previously stopped regular-lane builds.

Validation: both exported series replay cleanly from the bases above and
reproduce the prepared source trees exactly. All 17 subsequent vLLM source
patchers and the FlashInfer loader patch pass; all 141 changed Python files
parse. The 7 series tests, 28 SWA tests, and 100 build-selection checks pass.
Focused CPU checks cover cache-boundary publication, Mooncake save limits and
references, PLE dummy binding/byte-scale lookup, and autotuner overrides. The
32,768-token vocabulary and its wheel package-data rule are retained. GPU
compilation and inference remain for nightly CI.

The October 8, 2026 series was replayed against upstream vLLM
`ab905a885dfbfc60a2c02286cc9c608c93884de3` and FlashInfer
`fa7c741e0af4493a73352bf240c1bbc802cca288`. These identify the validation bases;
normal builds still follow their selected refs.

On October 9, the #54614 test hunk was refreshed to preserve the quantized-input
shape test added by upstream [#59612](https://github.com/vllm-project/vllm/pull/59612).
Both tests had been inserted at the same location, which caused the October 9
nightly to stop on a test-file conflict. The complete vLLM series was then
replayed successfully against the failed nightly's base
`59ffd73e8597d860730f4a8c4f778f27c75a0dfa` and current main
`75e3b17c055c609f02c64cd4ad3f8692b8170d8e`. That test-only refresh left runtime
patch content and lane selection unchanged. A CPU-only regression in
`tests/test_regular_patch_series.py` checks that both tests survive application.

Later on October 9, #50897 was refreshed to upstream head
`92176bd44606a3cd5857b5fd05b739cbcc434daf`. The old snapshot registered Mamba
tails at allocation but deferred full-block publication until step output.
At a full-block checkpoint, that could publish the same boundary twice and
trip `BlockPool.cache_full_blocks`'s `block_hash_num_tokens` assertion. The
updated PR restores allocation-time caching, removes the publication fence
and special Mamba-tail path, and only publishes the final lookahead block at
request finish. It also includes the upstream multi-module MTP fallback and
unified lookahead hasher.

The full 14-patch vLLM series was replayed against upstream main
`7d0b4e57aac4c4323225b0eee9ae960ce75ffffe`. Integration with the newer
Mooncake fix [#60695](https://github.com/vllm-project/vllm/pull/60695) preserves
explicit boundary puts, original prompt and resumed-prefill lengths, and
references for asynchronous store jobs. Lookahead hashes replace the extra
EAGLE proof margin for partial-tail saves too. Both sides of the overlapping
MTP tests are retained, and the Mooncake tests cover lookahead partial tails.
Focused CPU checks using extracted source methods reproduced the old cache
assertion and passed with the refreshed source; save limits, tail-block
references, and the build-selection tests also passed. Full inference
validation remains with the next nightly CI run.

The supplied conflict resolutions are retained, with adaptations for newer
upstream Mamba cache-boundary and KV-transfer fixes (#60533, #59197, and #60695).
Lookahead hashing disables the EAGLE block drop while retaining the resolved
cache-hit alignment and the one-token replay boundary. Both sets of tests/docs
are preserved. Dynamic linear warmup keeps upstream's `round_up=True` and
full-model BF16 tuning. FlashInfer's benchmark retains both cuDNN and SM12x MoE.
The existing SWA workaround recognizes #60252's divisor-aware replacement and
leaves it intact; its behavior on the B12X source is unchanged.

To refresh, compare the referenced PR revisions, reapply the complete sequence
to the intended upstream bases, resolve conflicts, and export the updated
per-step diffs. Remove changes already provided by the selected base. Preserve
both sides of test/doc conflicts; the series applier fails on unresolved
conflicts instead of choosing one side. Recheck source application, wheel
packaging, and the regular/B12X build-selection tests.

No additional wheel-cache metadata is introduced. Rebuild and publish both
regular FlashInfer and vLLM wheels when introducing or updating this series.

## Runtime options

The series supplies capabilities; recipe choices remain explicit. In particular,
`--prefix-match-unit 48`, adaptive draft depth, and dynamic kernel backends are
not enabled globally.

For an MTP drafter using Model Runner V2, set `draft_token_map` to the installed
vocabulary path and `draft_token_map_quantization` to `nvfp4` in
`--speculative-config`. Resolve the path inside the built image with:

```bash
python3 -c 'from importlib.resources import files; print(files("vllm") / "model_executor/layers/draft_vocab_lists/qwen38_flash_next_32k.json")'
```

The JSON is included in the exported vLLM wheel, so the runner does not need the
builder's source checkout.
