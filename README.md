# Task and architecture in learned RNN dynamics

Apekshith Thirnahalli Jayadeva  &  Arina Kleizer 

NeuroAI and ML for Neuroscience / TUM School of Life Sciences / Summer Semester 2026

## 1. Project summary

We ask whether the task or the architecture shapes the learned dynamics of a recurrent network more, and whether the answer depends on the task. The project was originally planned around standard DSA (Ostrow et al. 2023), but during instructor consultation we realised that standard DSA compares whole trajectories and cannot separate the intrinsic dynamics from the input processing, which turned out to be central to our hypothesis. We therefore switched to InputDSA (Huang et al. 2025), which fits an input driven linear model $x_{n+1} = A x_n + B u_n$ to each network and lets us compare the intrinsic dynamics A and the input matrix B separately. We train small vanilla RNN, GRU and LSTM networks on two NeuroGym tasks, PerceptualDecisionMaking (PDM) and DelayComparison (DC), and compare the fitted matrices across models. The main analysis compares vanilla RNN and GRU, and LSTM is added as a third architecture to test whether the pattern generalises.

## 2. Repository structure

| Path | What it holds |
| --- | --- |
| `README.md` |  |
| `technical_note.md` | method, main results, key figures and limitations |
| `requirements.txt` | Python dependencies |
| `notebooks/InputDSA.ipynb` | full analysis: training, A and B matrix fits, rank study, sensitivity checks, mechanism, LSTM extension |
| `figures/` | all figures produced during the project, with the ones embedded in the technical note marked in `figures/README.md` |

For our work with default parameters please visit - https://github.com/apekshithtj/NeuroAI_DSA
## 3. How to run

Python 3.12. Two ways to install:

**Option A: venv + pip.**

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install "git+https://github.com/mitchellostrow/DSA.git"
```

**Option B: conda.**

```
conda env create -f environment.yaml
conda activate inputdsa
```

Conda installs DSA automatically as part of the environment.

The DSA package is installed from GitHub, not from PyPI. On Colab pin `numpy==2.2.6` and `scipy==1.14.1` with `--force-reinstall --no-deps` if you hit an import error.

To reproduce the results, open `notebooks/InputDSA.ipynb` and run the cells from top to bottom. Runtime: about 15 seconds per vanilla model, 30 seconds per GRU model and slightly more per LSTM model. The full notebook takes a few minutes end to end on a Colab GPU.

## 4. Author contributions

This project was a joint effort at every stage. We planned it, made all main decisions and discussed every new result as a team, deciding the next steps side by side. 

The actual runs happened in parallel on our own machines. Apekshith Thirnahalli Jayadeva ran the full three-architecture design, including LSTM as the third architecture. Arina Kleizer ran the same design in her notebook, which is the version we used for the final numbers because it kept the same setup as the two-architecture main analysis. Both runs gave the same qualitative picture, which is a nice cross-check between our machines. Both authors also shared the literature review, the presentation design and the repository organisation.

## 5. Documentation of LLM usage

We used Claude (Anthropic, Opus 4.8 and Opus 4.7) to help write code and to consult when we ran into errors. The interpretation of the results was done by us, based on the literature and other sources. We reviewed and checked all generated code and results. We are responsible for the content of this project and we understand the methods, the results and the derivations that it contains.
