import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from alibi.explainers import CounterfactualProto
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

RANDOM_STATE = 42
QUALITY_THRESHOLD = 6

# load data
def load_wine_quality_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    wine = fetch_ucirepo(id=186)
    features = wine.data.features
    targets = wine.data.targets.values.ravel()
    feature_names = list(features.columns)

    # quality <= 5 -> 0 (low), quality >= 6 -> 1 (high)
    binary_targets = (targets >= QUALITY_THRESHOLD).astype(int)
    return features.values, binary_targets, feature_names


def split_and_scale(
    features: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=targets,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, y_train, y_test, scaler


def build_model(x_train_scaled: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(x_train_scaled, y_train)
    return model


def build_explainer(model: RandomForestClassifier, x_train_scaled: np.ndarray) -> CounterfactualProto:
    explainer = CounterfactualProto(
        model.predict_proba,
        shape=(1, x_train_scaled.shape[1]),
        use_kdtree=True,
        theta=10.0,
        feature_range=(x_train_scaled.min(axis=0), x_train_scaled.max(axis=0)),
        max_iterations=500,
        c_init=1.0,
        c_steps=5,
    )
    explainer.fit(x_train_scaled, d_type="abdm", disc_perc=[25, 50, 75])
    return explainer


def describe_label(label: int) -> str:
    if label == 1:
        return "High Quality" 
    else:
        return "Low Quality"


def main() -> None:
    x, y, feature_names = load_wine_quality_data()
    x_train_sc, x_test_sc, y_train, y_test, scaler = split_and_scale(x, y)

    model = build_model(x_train_sc, y_train)
    print(f"Model accuracy: {model.score(x_test_sc, y_test):.3f}")

    explainer = build_explainer(model, x_train_sc)
    print("Explainer fitted")

    low_quality_indices = np.where(y_test == 0)[0]
    x_instance = x_test_sc[low_quality_indices[1] : low_quality_indices[1] + 1]

    original_prediction = model.predict(x_instance)[0]
    original_probabilities = model.predict_proba(x_instance)[0]
    print(f"\nOriginal prediction : {describe_label(original_prediction)}")
    print(
        "Probabilities       : "
        f"Low={original_probabilities[0]:.3f}  High={original_probabilities[1]:.3f}"
    )

    explanation = explainer.explain(x_instance)
    if explanation.cf is None:
        print("\nNo counterfactual found. Try increasing max_iterations or adjusting theta.")
        return

    counterfactual_x = explanation.cf["X"]
    cf_prediction = model.predict(counterfactual_x)[0]
    cf_probabilities = model.predict_proba(counterfactual_x)[0]

    print(f"\nCounterfactual prediction : {describe_label(cf_prediction)}")
    print(
        "Probabilities             : "
        f"Low={cf_probabilities[0]:.3f}  High={cf_probabilities[1]:.3f}"
    )

    original_values = scaler.inverse_transform(x_instance)[0]
    counterfactual_values = scaler.inverse_transform(counterfactual_x)[0]
    deltas = counterfactual_values - original_values

    results = pd.DataFrame(
        {
            "Feature": feature_names,
            "Original Value": original_values.round(3),
            "Counterfactual": counterfactual_values.round(3),
            "Change": deltas.round(3),
        }
    )

    changed = results[results["Change"].abs() > 0.001].reset_index(drop=True)
    print("\nFeatures changed to flip Low -> High Quality")
    print(changed.to_string(index=False))


if __name__ == "__main__":
    main()