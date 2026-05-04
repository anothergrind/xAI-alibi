# xAI-alibi
## Project Topic: Explainable AI

ECS 2192
Advised by Beiyu Lin

## Overview

This project uses the Alibi library to generate counterfactual explanations for machine learning models. It demonstrates how changing specific features can flip a model's prediction.

## Wine Quality Classifier

`wine.py` trains a Random Forest classifier on the UCI Wine Quality dataset and generates counterfactual explanations for low-quality wine predictions.

### Running the Script

```bash
python wine.py
```

### Multiple Experiments

By default, the script runs 5 experiments on different low-quality test instances. To change the number of experiments, edit the `NUM_EXPERIMENTS` constant at the top of `wine.py`:

```python
NUM_EXPERIMENTS = 5  # Change this number
```

### Output

For each experiment, the script shows:
- Original prediction and probabilities
- Counterfactual prediction and probabilities
- Features that changed to flip the prediction (Low → High Quality)
- The magnitude of each feature change
