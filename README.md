# xAI-alibi

## What this project does

This project trains a machine learning model on the UCI Red Wine Quality dataset and then explains one prediction with a counterfactual explanation.

In simple terms:
- The model predicts whether a wine is low quality or high quality.
- The explainer finds the smallest feature changes that would flip a low-quality prediction into a high-quality prediction.

## What happens in `wine.py`

1. Load data
- Downloads the Wine Quality dataset from UCI.
- Uses chemical properties (features) and wine quality score (target).

2. Convert to binary classification
- Original quality score is converted to two classes:
	- `0` = low quality (quality < 6)
	- `1` = high quality (quality >= 6)

3. Split and scale
- Splits into train/test sets.
- Standardizes features with `StandardScaler`.

4. Train model
- Trains a `RandomForestClassifier`.
- Prints test accuracy.

5. Fit explainer
- Builds an Alibi `CounterfactualProto` explainer on training data.

6. Explain one low-quality test example
- Selects one test sample predicted as low quality.
- Generates a counterfactual example that flips prediction to high quality.

7. Show changes
- Converts scaled values back to original units.
- Prints a table of only the changed features and how much they changed.

## How to interpret the output

The final table answers this question:
"What feature changes would be enough for this model to change its prediction from low to high quality?"

Important: these are model-based explanations, not guaranteed real-world causal effects.

## Advisor-ready summary

"I trained a Random Forest classifier on the UCI wine dataset after converting the quality score into a binary target (low vs high). Then I used Alibi's CounterfactualProto explainer to generate counterfactuals for a low-quality test wine, showing the minimal feature changes needed to flip the model prediction to high quality."