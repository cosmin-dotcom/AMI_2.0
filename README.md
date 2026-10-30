# AMI 2.0

AMI 2.0 is an active-material screening workflow for efficiently exploring a large number of candidate structures and identifying the most promising ones for a target property (for example, adsorption performance). The project combines:

- a structured feature table (`FEATURES.csv`),
- a surrogate model that ranks unexplored candidates,
- an external property calculator based on RASPA,
- and an iterative loop that selects the next best candidates to evaluate.

The main execution entry point is `AMI_core/main.py`.

## About

This repository is designed for materials screening and active learning workflows in which evaluating every structure is expensive. Instead of blindly testing everything, the code:

1. loads a set of candidate structures and their descriptors,
2. builds a surrogate model (Gaussian process or random forest),
3. ranks the remaining candidates by expected improvement,
4. runs the expensive simulation for the most promising structures,
5. updates the model and repeats until a stopping point is reached.

The project is especially relevant for adsorption and porous materials workflows, where the number of possible structures can be very large but each simulation can be computationally demanding.

The repository contains:

- `AMI_core/` — the core AMI implementation and model logic,
- `examples/` — example use cases and folders for sample studies,
- `misc_simulation_inputs/` — supplementary simulation inputs,
- `environment_raspa2.yml` and `environment_raspa3.yml` — Conda environment files,
- `requirements.txt` — project dependencies.

## Installation

This project is intended to be used with a Conda environment. The repository includes an environment file for the RASPA 2 setup:

```bash
conda env create -f environment_raspa2.yml
conda activate <environment-name>
```

If your Conda installation does not pick up the environment name automatically, you can also create the environment explicitly with:

```bash
conda env create -f environment_raspa2.yml --name ami_raspa2
conda activate ami_raspa2
```

You may also install the Python requirements directly if needed:

```bash
pip install -r requirements.txt
```

Note: the project is built around Python 3.7 and includes dependencies such as NumPy, SciPy, scikit-learn, pandas, h5py, RASPA2, ASE, and gpflow.

## Usage

The main script is located in `AMI_core/main.py`.

From the repository root:

```bash
cd AMI_core
python main.py -n 20 -r gp -p 1
```

Common arguments:

- `-n` : total number of structures to screen,
- `-r` : ranker to use (`gp`, `rf`, or `rr`),
- `-p` : whether to load prior results (`1` = yes, `0` = no).

Example:

```bash
python main.py -n 50 -r gp -p 1
```

This will:

- read `FEATURES.csv`,
- remove the `cif_path` column,
- standardize the feature matrix,
- save the processed data as an HDF5 file,
- build the surrogate model,
- initialize the AMI data manager and scheduler,
- optionally load prior data from `PRIOR.csv`,
- and then run the screening loop.

`PRIOR.csv` is used when you already have known values for a subset of candidates. If you do not, run with `-p 0`.

## Workflow overview

A simple way to think about the process is this:

```text
main.py
  -> load FEATURES.csv
  -> standardize features
  -> build surrogate model
  -> set up AMI configuration
  -> optionally load prior results
  -> run iterative screening loop
```

The most important pieces are:

- `main.py` — the main driver script,
- `ranking_models.py` — defines the rankers that decide which candidate is next,
- `surrogate/` — contains the surrogate model and acquisition logic,
- `ami/` — contains the AMI framework components such as scheduler, worker pool, data manager, and config,
- `raspa.py` (or the simulation wrapper used by the workflow) — calls the external simulation engine.

### How `main.py` calls the different pieces

The script does the following in a simplified sequence:

1. Reads the feature table from `FEATURES.csv`.
2. Standardizes the input features and stores them as HDF5.
3. Creates a surrogate ranking model using either:
   - `DenseGaussianProcessregressor`, or
   - `DenseRandomForestRegressor`.
4. Creates an expected-improvement ranker (`ExpectedImprovementRanker`) and an initial random ranker.
5. Builds an AMI `Configuration` object with:
   - a `SerialSchedulerFactory`,
   - a `ShareMemorySingleThreadWorkerFactory`,
   - an `InMemoryDataManager`,
   - a `SingleNodeWorkerPoolFactory`.
6. Loads prior data if requested.
7. Runs the AMI screening loop using `runner.run(50)`.

### Factory and worker pool in plain English

A `factory` in this project is a small builder object that creates the correct type of component for the job. Instead of manually wiring everything together, the code asks a factory to construct a scheduler, worker, or pool with the right settings.

A `worker pool` is a collection of workers that can do calculations in parallel or in a coordinated way. Here, `SingleNodeWorkerPoolFactory` is used to define how many CPUs should be available on the local machine, with:

```python
pool.set("ncpus", pool_size)
```

This makes the screening loop easier to scale while keeping the setup simple.

### AMI terminology explained simply

In this codebase, some terms are used in a more technical/lifecycle sense:

- `dirty`: a candidate or data item that has not yet been processed cleanly or has not yet received a valid final result. In practical terms, it is an item that is still in an incomplete state and must be evaluated or refreshed.
- `future`: a placeholder for a result that will exist at some point in the future. In asynchronous or parallel workflows, the code may launch a task and later receive the result once the computation is complete.
- `ranker`: the logic that decides which remaining candidate should be evaluated next. It is essentially the “next best candidate selector.”
- `surrogate model`: a fast approximation model used to predict the target property before running the expensive simulation.

These ideas are common in active learning and asynchronous computing: the system keeps track of what has been done, what is pending, and what should be evaluated next.

## Typical execution flow

The project is essentially a closed loop:

```text
Candidate structures
      |
      v
Feature extraction + standardization
      |
      v
Surrogate model predicts values
      |
      v
Ranker chooses best candidates
      |
      v
RASPA / simulation computes actual property
      |
      v
Results are stored and fed back into the model
      |
      v
Repeat until target number of structures screened
```

This is the core idea behind AMI: use machine learning to reduce the number of expensive calculations needed while still finding strong candidates.

## AI statement

This project includes sections that were developed or refined with AI assistance. Some source files contain inline comments noting where AI was used to help implement or improve parts of the code. AI was used to support code generation, debugging, refactoring, and explanation, while the scientific design, validation, and final interpretation remain the responsibility of the project owner and users.

If you want, I can also create a more formal project-specific README tailored to your exact use case (for example, one focused on adsorption screening and RASPA workflows rather than a generic software overview).

