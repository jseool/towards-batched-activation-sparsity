# Towards Batched Activation Sparsity in LLM Decoding

Artifact for the master's thesis **“Towards Batched Activation Sparsity in
LLM Decoding”** (Junseo Lee, Seoul National University, 2026).

Thesis: [official full text at Seoul National University dCollection](https://dcollection.snu.ac.kr/public_resource/pdf/000000199763_20261107004756.pdf).

The thesis asks how much activation sparsity can remain when all sequences in
a decoding batch must share one hardware-friendly block mask. It provides:

- a per-step oracle search under a TEAL-50% next-token NLL budget;
- a batch-shared, 64-channel block-sparse GEMM kernel and offline search;
- batch-size, grouping, and dataset sensitivity results; and
- practical online heuristics that expose the gap between oracle sparsity and
  end-to-end runtime realization.

The main finding is that oracle mean block sparsity remains about 64% at batch
size 16 and above 50% at batch size 64. The corresponding projection-kernel
speedups are about 1.60x and 1.20x. These are oracle/kernel upper bounds, not
end-to-end serving speedups.

![Achievable sparsity and kernel speedup](figures/figure_4_4_sparsity_speedup.png)

## Repository layout

```text
figures/     figures used in Chapter 4 and Appendix A
results/     compact raw JSON/CSV snapshots behind the reported numbers
scripts/     oracle, kernel, heuristic, and plotting programs
kernels/     Triton block-sparse GEMM implementation
docs/        reproduction protocol and result-to-artifact map
```

## Reproducing the figures

This path is CPU-only and uses the committed result snapshots.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-plot.txt
python scripts/plot_paper_results.py
```

Generated files are written to `reproduced_figures/`. The script verifies the
reported batch-16 and batch-64 values against the committed measurements.

## Full GPU reproduction

The oracle and kernel experiments require an NVIDIA GPU, Llama 2 access, and
TEAL calibration histograms. See [docs/REPRODUCING.md](docs/REPRODUCING.md)
for setup and commands. Measurements in the thesis used an NVIDIA A100X 80GB.

## Paper-to-artifact map

| Thesis item | Committed result | Reproduction code |
|---|---|---|
| Figure 4.1, PPL vs. TEAL sparsity | `figures/figure_4_1_ppl_vs_sparsity.png` | Snapshot assembled from separate TEAL sweeps |
| Figure 4.2, oracle heatmap | `results/oracle/perstep_batch_sweep.json` | `scripts/plot_paper_results.py` |
| Figure 4.3, oracle sparsity vs. batch | `results/oracle/perstep_batch_sweep.json` | `scripts/eval_perstep_batch_sweep.py`, `scripts/plot_paper_results.py` |
| Figure 4.4, sparsity and speedup | oracle JSON + kernel JSON | `scripts/bench_perstep_actual_kernel_speedup.py`, `scripts/plot_paper_results.py` |
| Table 4.1, sensitivity | `results/ablations/` | `scripts/eval_random_grouping_b16.py`, `scripts/eval_wikitext_vs_alpaca_b16g3.py` |
| Table 5.1, online heuristic | `results/practical/` | `scripts/bench_alpaca_energy_budget_e2e.py` |

For exact scope and limitations, see [docs/ARTIFACT_MANIFEST.md](docs/ARTIFACT_MANIFEST.md).

## Citation

Please use [CITATION.cff](CITATION.cff), or:

```bibtex
@mastersthesis{lee2026batchedactivation,
  author = {Junseo Lee},
  title = {Towards Batched Activation Sparsity in LLM Decoding},
  school = {Seoul National University},
  year = {2026},
  month = {August}
}
```

## License and third-party components

Artifact code is released under the MIT License. The thesis PDF is available
from the institutional repository linked above. The kernel and experiment
code builds on TEAL; its MIT license is preserved in
`third_party/TEAL_LICENSE`. Model weights and datasets are not redistributed
and remain subject to their own licenses and access terms.
