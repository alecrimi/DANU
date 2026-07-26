# DANU: Dendritic Attractor Neuron Unit

<p align="center">
  <img src="danu.png" width="650" alt="DANU idea">
</p>

Implementation accompanying the paper:

> **DANU: The Dendritic Attractor Neuron Unit for Infant Degenerated EEG Prediction**  
> Alessandro Crimi, Cris Micheli, Max Talanov

---
# Experiments

The repository reproduces the experiments presented in the paper.

## 1. Lorenz-96 Forecasting

Chaotic multivariate time-series prediction on the Lorenz-96 system.

**Task**

- Input sequence: 100 time steps
- Prediction horizon: 10 future steps
- Loss: Mean Squared Error (MSE)

Run:

```bash
python training/train_lorenz.py
```

---

## 2. Cork Neonatal EEG Dataset

Next-step prediction of neonatal EEG recordings acquired from infants with hypoxic-ischaemic encephalopathy (HIE).

Dataset:

- 169 recordings
- 53 neonates
- 4 EEG severity grades

Task:

- Window length: 512 samples
- Predict next EEG sample

Run:

```bash
python training/train_cork.py
```

---

## 3. Synthetic Hippocampal Recordings

Prediction of simulated hippocampal activity generated from a biologically-inspired CA3–CA1 spiking neural network.

The simulator progressively increases recurrent excitation in CA3 to generate four dynamical regimes.

Run:

```bash
python danu_ca3_main.py
```

---

# Implemented Models

The repository contains implementations of

- DANU (proposed)
- Tractable Dendritic RNN
- GRU
- LSTM

All models share identical training protocols for fair comparison.

---

# Installation

Clone the repository

```bash
git clone https://github.com/alecrimi/DANU.git
cd DANU
```

Install dependencies

```bash
pip install -r requirements.txt
```

Main dependencies

- Python ≥ 3.10
- PyTorch
- NumPy
- SciPy
- Matplotlib
- pandas
- scikit-learn
- NEST Simulator (hippocampal experiments only)

---

# Reproducing the Paper

| Experiment | Script |
|------------|--------|
| Lorenz-96 | `train_lorenz.py` |
| Cork EEG | `train_cork.py` |
| Hippocampus | `train_hippocampus.py` |

Each experiment automatically

- trains all compared models
- performs early stopping
- evaluates the validation MSE
- computes statistical comparisons
- generates publication-quality figures

---

# Citation

If you use this code, please cite

```bibtex
@article{crimi2026danu,
  title={DANU: The Dendritic Attractor Neuron Unit for Infant Degenerated EEG Prediction},
  author={Crimi, Alessandro and Micheli, Cris and Talanov, Max},
  journal={},
  year={2026}
}
```

---

# Acknowledgements

This work builds upon previous research in

- recurrent neural networks
- dendritic computation
- hippocampal attractor dynamics
- neonatal EEG forecasting

---

# License

MIT License.
