# NeuroAI_and_MAC_InputDSA
# Task and architecture in learned RNN dynamics

InputDSA analysis of how recurrent network architecture and task shape learned dynamics.

## 1. Project summary

Title: Task and architecture in learned RNN dynamics, studied with InputDSA.

This project asks whether the architecture of a recurrent neural network leaves a detectable signature in its learned dynamics and whether this signature depends on the task the network was trained on. Our working hypothesis is that architecture matters more on a task that admits many possible dynamical solutions and matters less on a task that admits few. We train small recurrent networks on two neuroscience tasks, fit a linear input driven model to their hidden states with InputDSA (Huang et al. 2025) and then compare the fitted dynamics matrix A and the fitted input matrix B across all networks. The core analysis compares two architectures (vanilla RNN and GRU) and an extension adds a third architecture (LSTM) to test whether the pattern holds more broadly.

## 2. Repository structure

The repository has two parts that tell one story. The core part is the primary analysis in a notebook. The extension part is a modular pipeline that adds a third architecture and two extra comparison methods as controls.

```
NeuroAI_DSA/
├── README.md              this file
├── technical_note.md      method, main results, key figures and limitations
├── requirements.txt       Python dependencies for a reproducible environment
├── notebooks/
│   └── InputDSA.ipynb      primary analysis (2 architectures, A and B matrices)
├── src/                    extension pipeline (adds LSTM as a third architecture)
│   ├── config.py           all hyperparameters and the experiment grid
│   ├── utils.py            seeding, device selection and model naming
│   ├── tasks.py            NeuroGym task wrappers and the fixed test set
│   ├── models.py           vanilla RNN, GRU and LSTM model definitions
│   ├── train.py            train one model and save weights and metadata
│   ├── extract.py          run a trained model and save hidden states and inputs
│   ├── analysis.py         pairwise InputDSA, plain DSA and Procrustes distances
│   ├── comparisons.py      the structured contrasts (noise floor, within, across)
│   ├── plots.py            distance heatmaps and MDS projections
│   ├── run_mvp.py          minimal viable run
│   ├── run_grid.py         full grid runner
│   └── sweep.py            hyperparameter sweeps
├── figures/               the figures shown in the presentation
└── results/               distance matrices and summary json files
```

What each part covers:

The notebook in `notebooks/InputDSA.ipynb` is the main result of the project. It trains 20 models (2 architectures times 2 tasks times 5 seeds), fits A and B matrices with SubspaceDMDc and runs the full comparison. It contains the A matrix analysis, the rank sensitivity checks and the B matrix analysis that gives our most robust finding.

The pipeline in `src/` is the extension. It re-implements the same idea as reusable scripts and adds LSTM as a third architecture so we can check whether the two architecture pattern generalises. It also runs plain DSA and geometric Procrustes as method controls next to InputDSA.

The `figures/` folder holds the exact figures we show in the presentation. The `results/` folder holds the saved distance matrices and summary files so that the figures can be regenerated without retraining.

## 3. How to run

Python version: 3.12 (developed on 3.12.4).

Install the dependencies into a fresh virtual environment:

```
python -m venv .venv
source .venv/bin/activate            # on Windows use .venv\Scripts\activate
pip install -r requirements.txt
pip install "git+https://github.com/mitchellostrow/DSA.git"
```

The last line installs the DSA package that provides both DSA and InputDSA. It is kept separate from `requirements.txt` because it is installed straight from GitHub and not from PyPI.

Note for Google Colab. On Colab the pre-installed numpy and scipy can conflict with the DSA package. If you hit an import error, pin the versions with:

```
pip install numpy==2.2.6 scipy==1.14.1 --force-reinstall --no-deps
```

Run the primary analysis (the main result):

```
Open notebooks/InputDSA.ipynb and run the cells from top to bottom.
```

The notebook trains all 20 models, extracts hidden states, fits the matrices and produces every figure used for the core result. The cells are grouped and each group has a short markdown note before and after it that explains what the cell does and what we saw.

Run the extension (the three architecture check):

```
cd src
python run_mvp.py                     # small check that the pipeline runs end to end
python run_grid.py                    # full grid with InputDSA as the primary method
python run_grid.py --method dsa       # same grid with plain DSA (control)
python run_grid.py --method procrustes # same grid with geometric Procrustes (control)
```

Outputs land in `results/` (distance matrices and summary json) and in `saved_models/` (weights and per model metadata).

Expected runtime (rough, on a single GPU, please adjust to your machine):

- one model trains in about 15 seconds for vanilla RNN and about 30 seconds for GRU
- training all 20 models for the notebook takes a few minutes
- fitting the matrices and computing the 190 pairwise distances takes about 3 minutes
- the full three architecture grid in `src/` is heavier because it adds LSTM and two extra methods, so budget more time for it

Generating the figures. The notebook writes its figures inline and to `figures/`. The pipeline writes its figures to `figures/` as well. If you only want to regenerate the figures from saved results without retraining, load the saved distance matrices from `results/` and re run the plotting cells or `src/plots.py`.

## 4. Author contributions

Please verify and adjust this section so that it matches exactly what each of you did.

Arina [surname] carried out the primary InputDSA analysis in `notebooks/InputDSA.ipynb`. This includes training the 20 two architecture models with decision accuracy checkpointing, fitting the A and B matrices with SubspaceDMDc, building the 20 by 20 distance matrix, the rank selection study (participation ratio, Optimal Hard Threshold and the MASE and AIC sweep), the full set of sensitivity checks and the B matrix analysis with its mechanistic decomposition.

Apekshith Thirnahalli Jayadeva built the modular pipeline in `src/` and added the LSTM third architecture. This includes the reusable training and extraction scripts, the grid runner and the plain DSA and geometric Procrustes method controls that sit next to InputDSA.

Both authors planned the project together and made all the main decisions jointly. Throughout the work we discussed each new result together and decided the next steps based on what we found. We also shared the literature review, the design of the final presentation and the organisation of this repository.

## 5. Documentation of LLM usage

We used Claude (Anthropic, Opus 4.8 and Opus 4.7) to help write code and to consult when we ran into errors. The interpretation of the results was done by us, based on the literature and other sources. We reviewed and checked all generated code and results. We are responsible for the content of this project and we understand the methods, the results and the derivations that it contains.
