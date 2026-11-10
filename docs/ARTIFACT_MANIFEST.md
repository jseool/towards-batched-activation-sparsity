# Artifact manifest

## Included

- Compact result snapshots required to audit Chapter 4 and Chapter 5 claims.
- Figure snapshots corresponding to Figures 4.1–4.4 and Appendix A.
- The batch-shared block oracle, GPU kernel benchmark, online energy heuristic,
  grouping/dataset ablations, and Triton kernel implementation.
- A CPU-only plotting and consistency-check path.

## Excluded

- Llama 2 weights and tokenizer: governed by Meta's license and downloaded by
  the user after access approval.
- Thesis PDF: distributed through the university's official repository rather
  than duplicated in this source-code repository.
- TEAL activation histograms and model caches: large derived artifacts; paths
  are supplied through environment variables.
- Profiler captures, intermediate binaries, logs, caches, and exploratory runs
  not used in the thesis.
- Complete copies of third-party research repositories. Only code needed for
  this artifact is retained, with the relevant TEAL license notice.

## Claim boundaries

- The oracle uses future next-token quality measurements and is not a
  deployable runtime policy.
- Kernel speedup aggregates linear-projection kernels only. It is not
  end-to-end decoding throughput.
- Results were measured on one NVIDIA A100X 80GB GPU with Llama 2 7B.
- Online selection overhead can reverse kernel gains; Chapter 5 reports this
  explicitly.

## Provenance

The committed JSON/CSV files are snapshots from the thesis experiments. The
plotting script derives reported values directly from these files. SHA-256
hashes can be generated with:

```bash
sha256sum results/**/*.json results/**/*.csv
```
