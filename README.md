# Task and architecture in learned RNN dynamics

**Apekshith Thirnahalli Jayadeva** / **Arina Kleizer**

TUM School of Life Sciences , NeuroAI and ML for Neuroscience , Summer Semester 2026

## 1. Project summary

We ask whether the task or the architecture shapes the learned dynamics of a recurrent network more, and whether the answer depends on the task. We train small vanilla RNNs and GRUs on two NeuroGym tasks, PerceptualDecisionMaking (PDM) and DelayComparison (DC), fit an input driven linear model with InputDSA (Huang et al. 2025) and compare the fitted matrices across models. An extension adds an LSTM as a third architecture.

## 2. Repository structure

| Path | What it holds |
| --- | --- |
| `README.md` |  |
| `technical_note.md` | backgruond, hypothesis, method, main results, key figures and limitations |
| `requirements.txt` | Python dependencies |
| `notebooks/InputDSA.ipynb` | primary analysis, 2 architectures, A and B matrices |
| `src/` | extension pipeline that adds LSTM as a third architecture |
| `figures/` | figures used in the presentation |
| `results/` | saved distance matrices and summary files |

The notebook is the main result. The pipeline in `src/` re-runs the same comparison with LSTM added and with plain DSA and geometric Procrustes as method controls.

## 3. How to run

Python 3.12. Install:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install "git+https://github.com/mitchellostrow/DSA.git"
```

The DSA package is installed from GitHub, not from PyPI. On Colab pin `numpy==2.2.6` and `scipy==1.14.1` with `--force-reinstall --no-deps` if you hit an import error.

Reproduce the main result: open `notebooks/InputDSA.ipynb` and run top to bottom.

Reproduce the extension: `cd src && python run_grid.py` for the InputDSA grid, `--method dsa` or `--method procrustes` for the controls.

Runtime: about 15 seconds per vanilla model and 30 seconds per GRU model. The full notebook takes a few minutes end to end.

## 4. Author contributions

Arina Kleizer ran the primary analysis in `notebooks/InputDSA.ipynb`: training the 20 two-architecture models, fitting A and B, the rank selection study, the sensitivity checks and the B matrix analysis with its mechanistic decomposition.

Apekshith Thirnahalli Jayadeva built the pipeline in `src/` and added the LSTM as a third architecture, together with plain DSA and geometric Procrustes as method controls.

Both authors planned the project together and made all main decisions jointly. Throughout the work we discussed each new result and decided the next steps together. We also shared the literature review, the presentation design and the repository organisation.

## 5. Documentation of LLM usage

We used Claude (Anthropic, Opus 4.8 and Opus 4.7) to help write code and to consult when we ran into errors. The interpretation of the results was done by us, based on the literature and other sources. We reviewed and checked all generated code and results. We are responsible for the content of this project and we understand the methods, the results and the derivations that it contains.
