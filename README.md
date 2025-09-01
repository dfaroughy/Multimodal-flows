# Multimodal‑flows

Multimodal Flow for particle cloud generation with hybrid continuous (kinematics) and discrete (particle IDs) variables. The model combines flow matching for continuous features with Markov jump dynamics for discrete features to learn realistic particle clouds from jets.

This repo contains:
- Training and sampling scripts powered by Pytorch Lightning
- Experiments are monitored via Comet
- Data utilities for AspenOpenJets (AOJ)
- Particle transformer backbones (ParticleFormer)
- Lightning Callbacks for logging, EMA, and generation

If you’re just getting started, follow the Quickstart below.

<!-- **Quickstart**
- Clone, create an environment, install dependencies, and install the package in editable mode.
- Download or point to AOJ data.
- Train a model, then generate samples. -->

**Installation**
- Python: 3.10+
- Then install project deps and package:
  - `pip install torch --index-url https://download.pytorch.org/whl/cu118`
  - `pip install -e .`

<!-- Note: if PyTorch install via `requirements.txt` conflicts with your CUDA setup, install PyTorch first as shown above, then install the rest. -->

** AspenOpenJets (AOJ) dataset**
- Default code expects AOJ `.h5` files under a base directory: `--dir <BASE>`, data in `<BASE>/aoj/RunG_batch*.h5`.
- You can let the loader download files (uses AOJ URL) by passing `download=True` in code, or manually place files under `<BASE>/aoj/`.
- Relevant loader: `multimodal_flows/utils/aoj.py` (`AspenOpenJets`).

Example layout:
- `<BASE>/aoj/RunG_batch0.h5`
- `<BASE>/aoj/RunG_batch1.h5`

**Training**
- Script: `scripts/train_mmf.py`
- Minimal example (set your experiment base dir):
- `python scripts/train_mmf.py --dir ./experiments --project jet_sequences --data_files RunG_batch0.h5 --batch_size 256 --max_epochs 50 --model ParticleFormer`

Key flags:
- `--dir`: Base directory for experiments and data (expects `aoj/` under it)
- `--project`: Project name; outputs under `<dir>/<project>/<experiment_id>`
- `--data_files`: AOJ file name(s); comma‑separated or pass multiple times
- `--num_jets`: Optional cap on jets for quicker runs
- `--model`: Backbone key registered in `networks/registry.py` (e.g., `ParticleFormer`, `FusedParticleFormer`, `EPiC`)
- `--use_ema_weights`: Enable EMA model weights during training/validation
- `--multitask_loss`: `sum`, `weighted`, or `time-weighted`

Outputs:
- Checkpoints, logs, and a `config.yaml` are saved under `<dir>/<project>/<experiment_id>/` (experiment ID comes from the logger).

**Sampling / Generation**
- Script: `scripts/sample_mmf.py`
- Example to generate at multiple temperatures and time steps:
- `python scripts/sample_mmf.py --dir ./experiments --project jet_sequences --experiment_id <ID> --data_files RunG_batch0.h5 --num_jets 100000 --batch_size 256 --checkpoint best --num_timesteps 50 100 --temperature 1.0 0.8 --tag demo`

What it does:
- Loads the trained checkpoint and generates particle clouds starting from noise, writing results to `<dir>/<project>/<experiment_id>/generation_results_<tag>/`.
- Continuous features are de‑standardized using training metadata.
- Optionally computes and logs plots/metrics when `--make_plots true`.

**Repo Layout**
- `scripts/`: CLI entry points for training and sampling
- `multimodal_flows/model/`: Bridge dynamics, solvers, Lightning module
- `multimodal_flows/networks/`: Transformer backbones and registry
- `multimodal_flows/utils/`: Datasets, tensor dataclass, callbacks, helpers, metrics, plotting
- `notebooks/`: Analysis notebooks and figures

**Tips & Common Pitfalls**
- Install the package with `pip install -e .` so imports like `from utils...` resolve via the package mapping in `setup.py`.
- Set your own Comet credentials or choose a different logger if you don’t use Comet. The logger persists `config.yaml` into the run directory.
- Use GPU: set `accelerator=gpu` in the Trainer (already the default in the scripts) and ensure CUDA is visible.

**Configuration**
- The training script saves a `config.yaml` into the experiment directory. Use `--experiment_id` in subsequent runs to resume or to run inference using the saved config.
- Important knobs:
  - Data: `continuous_features`, `discrete_features`, `max_num_particles`
  - Dynamics: `gamma` (discrete), `sigma`, `time_eps`
  - Optim: `lr`, `lr_final`, `warmup_epochs`
  - Sampling: `num_timesteps`, `temperature`, `top_k`, `top_p`

**Reproducibility**
- Pin dependency versions where possible and seed all RNGs (Lightning allows passing a seed or calling `seed_everything`).
- Save `requirements.txt` or `pip freeze` per run in your experiment folder for exact replication.

**Citations**
- Aspen OpenJets dataset: please cite the AOJ source when using the data.
- If you use this code in academic work, please cite this repository (bib entry TBD).

**License**
- Add your license of choice (e.g., MIT) in `LICENSE`.

Questions or issues? Open an issue with details about your environment, command, and logs.
