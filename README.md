# NBA Draft Predictor – College & Combine Data Draft Pick Forecast

Predict NBA draft picks as continuous values (1–60 + undrafted) using college statistics and combine metrics (2009 - 2021).

## Project Overview

This project builds a neural network model to predict NBA draft positions using player college stats and combine measurements. The goal is to forecast draft outcomes as a continuous value rather than classifying players into rounds.

Key highlights:

* Predicts draft pick as a continuous value (1–60, with undrafted players handled as 61)
* Achieved ±5 pick accuracy of **44%** on the test set
* Feature importance analysis identifies most impactful predictors (e.g., player grade, scoring efficiency, strength, position, team)
* Data pipeline includes feature scaling, one-hot encoding, and train/test split
* Implemented in **Python**, leveraging **PyTorch**, **Pandas**, and **Scikit-Learn**

---


## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/nba-draft-predictor.git
cd nba-draft-predictor
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages include:

* `pandas`
* `numpy`
* `torch`
* `scikit-learn`
* `joblib`

---

## Dataset

* Place your dataset in the `data/` folder as `dataset.csv`.

* The dataset should contain the following:

  * `draft_pick`: original draft pick (use `-1` for undrafted)
  * `player_name`: player identifier
  * Other features: college stats and combine metrics

* The pipeline automatically converts categorical variables to one-hot encoding and scales features.

---

## Usage

1. Run the training and evaluation script:

```bash
python train_nba_draft_model.py
```

2. The script will:

* Preprocess data
* Train the neural network
* Evaluate exact and ±n pick accuracy
* Save the model and scaler for future use

---

## Model Architecture

* Fully-connected neural network with **4 hidden layers**
* Each hidden layer has **BatchNorm**, **ReLU activations**, and **dropout**
* Output layer uses **sigmoid** to predict a continuous value in `[0, 1]`
* Continuous predictions are scaled back to draft pick range `1–61`

---

## Training

* Optimizer: **Adam**, learning rate `0.001`
* Loss function: **Mean Squared Error (MSE)**
* Epochs: 150
* Target is scaled to `[0, 1]` to match sigmoid output
* Features are standardized (mean=0, std=1) for better convergence

---

## Evaluation

After training, predictions are rescaled:

* Undrafted players (originally `-1`) mapped to `61`
* Metrics computed:

  * **Exact pick accuracy**
  * **±1 pick accuracy**
  * **±2 pick accuracy**
  * **±5 pick accuracy**

Example output:

```
Exact Pick Accuracy: 12.50%
±1 Pick Accuracy: 24.00%
±2 Pick Accuracy: 33.00%
±5 Pick Accuracy: 44.00%
```

---

## Feature Importance

* Permutation feature importance identifies the most impactful features

* Features shuffled individually to measure performance drop

* Top predictors for draft outcome include:

  * Player grade
  * Scoring efficiency
  * Strength
  * Position
  * Team

* Results printed for top 20 features by importance

---

## Saving & Loading the Model

* Model weights saved as:

```text
continuous_draft_model.pt
```

* Feature scaler saved as:

```text
continuous_scaler.pkl
```

* To load the model and scaler for inference:

```python
import torch
import joblib

model.load_state_dict(torch.load("continuous_draft_model.pt"))
scaler = joblib.load("continuous_scaler.pkl")
```


