# Reproduction protocol

## 1. Plot-only path (CPU)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-plot.txt
python scripts/plot_paper_results.py
```

This reproduces Figures 4.2–4.4 from the committed measurements and validates
the batch-16 and batch-64 values. Figure 4.1 is retained as a snapshot
because its curve was assembled from multiple TEAL calibration sweeps.

## 2. GPU prerequisites

The thesis measurements and committed kernel timings were obtained on an
**NVIDIA A100X 80GB**. Kernel configurations are not portable performance
claims. On another GPU, generate a new lookup table by rerunning the complete
kernel configuration search and profiling procedure before measuring Figure
4.4 or the online methods.

- Linux, CUDA-capable NVIDIA GPU, and a recent CUDA toolkit.
- Python 3.10 or 3.11.
- Access to `meta-llama/Llama-2-7b-hf` on Hugging Face.
- A checkout of the TEAL repository and its Llama-2-7B activation histograms.
- The packages in `requirements-gpu.txt`.

The measurements were produced with Ubuntu 22.04, Python 3.10.14, CUDA 12.1,
cuDNN 8.9, PyTorch 2.3.1, Triton 2.3.1, Transformers 4.43.3, Datasets 2.20.0,
and FlashAttention 2.8.3. `Dockerfile` and `requirements-gpu-lock.txt` capture
this environment. The unpinned `requirements-gpu.txt` remains available for
newer compatible environments, which must be reprofiled before reporting
kernel timings.

Build and enter the pinned environment from the repository root:

```bash
docker build -t batched-activation-sparsity .
docker run --rm -it --gpus all \
  -v "$PWD":/artifact \
  -v /path/to/huggingface-cache:/models \
  -e HF_HOME=/models \
  batched-activation-sparsity
```

Use TEAL commit `fb7373c93ac3594817c9ee64d4e08b47430a1822` for the
calibration utilities used by the oracle experiments:

```bash
git clone https://github.com/FasterDecoding/TEAL.git
git -C TEAL checkout fb7373c93ac3594817c9ee64d4e08b47430a1822
```

Set portable paths rather than editing the scripts:

```bash
export HF_HOME=/path/to/huggingface-cache
export TEAL_ROOT=/path/to/TEAL
export TEAL_HIST_PATH=/path/to/Llama-2-7B/histograms
export MODEL_NAME=meta-llama/Llama-2-7b-hf
```

## 3. Oracle batch sweep (Figure 4.3)

The full sweep tries every layer/projection sparsity increment at each selected
decoding step. A minimal run is:

```bash
python scripts/eval_perstep_batch_sweep.py \
  --out_dir outputs/oracle-minimal \
  --batch_sizes 2 \
  --num_groups 1 \
  --steps 0 \
  --grid 0.1
```

Full configuration:

```bash
python scripts/eval_perstep_batch_sweep.py \
  --out_dir outputs/oracle-full \
  --batch_sizes 2,4,8,16,32,64 \
  --num_groups 5 \
  --decode 32 \
  --steps 0,1,2,3,4,5,6,7,8,9,10 \
  --block_k 64 \
  --budget_sparsity 0.5 \
  --grid 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
```

## 4. Kernel configuration search and replay (Figure 4.4)

Kernel configuration lookup tables must first be produced on the target GPU.
The search covers metadata representation, Split-K, tile shape, warp count,
and pipeline stages for each GEMM shape and active density. Once a lookup
directory for the target hardware exists:

```bash
python scripts/profile_kernel_configs.py \
  --out_dir outputs/kernel-profile \
  --batch_sizes 2,4,8,16,32,64 \
  --densities 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
```

The full search evaluates hundreds of configurations per operating point. For
a quick check of a new environment, use `--batch_sizes 16 --densities 0.5`.
After profiling, measure the selected configurations against the oracle maps:

```bash
python scripts/bench_perstep_actual_kernel_speedup.py \
  --perstep_json outputs/oracle-full/perstep_batch_sweep.json \
  --lookup_dir outputs/kernel-profile \
  --out_dir outputs/kernel
```

Run `python scripts/bench_perstep_actual_kernel_speedup.py --help` to inspect
the timing controls. CUDA graphs and repeated median measurements are used.

## 5. Practical online heuristic (Table 5.1)

```bash
python scripts/bench_alpaca_energy_budget_e2e.py \
  --out_dir outputs/energy-b16 \
  --sweep_json outputs/kernel-profile/llama2_kernel_best_sweep_B16.json \
  --batch_size 16 \
  --group_id 3 \
  --energy_keeps 0.99,0.97,0.95,0.90,0.85,0.80 \
  --profile
```

## 6. Expected variability

Oracle sparsity should be deterministic given the same model, dataset revision,
group construction, and thresholds. Kernel timing and the selected kernel
configuration depend on the GPU, driver, CUDA, PyTorch, and Triton versions;
record the full environment and reprofile after changing it.
