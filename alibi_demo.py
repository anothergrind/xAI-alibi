from alibi.datasets import fetch_adult

from alibi.explainers import AnchorTabular

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestRegressor


adult = fetch_adult()
data = adult.data
target = adult.target

print("Data:")
print(data)

print("Target: ")
print(target)

print(adult.data.shape, adult.target.shape)

# * Wine model from UCI
wine = load_wine(as_frame=True)
X, y = wine.data, wine.target
rf_w = RandomForestRegressor().fit(X, y.squeeze()) # training model

# Anchor explainer
explainer_w = AnchorTabular(rf_w.predict, feature_names=X.columns)
explainer_w.fit(X.values)

explanation_w = explainer_w.explain(X.iloc[0].values)
print(explanation_w.data['anchor'])


# * Titanic model
# titanic = fetch_openml(name="titanic", version=1, as_frame=True)
# U, v = titanic.data, titanic.target
# U = pd.get_dummies(U.astype("object").fillna("missing"), drop_first=True)
# v = pd.Series(v).astype("category").cat.codes
# rf_t = RandomForestRegressor().fit(U, v.squeeze())
# explainer_t = AnchorTabular(rf_t.predict, feature_names=U.columns)
# explainer_t.fit(U.values)

# explanation_t = explainer_t.explain(U.iloc[0].values)
# print(explanation_t.data['anchor'])