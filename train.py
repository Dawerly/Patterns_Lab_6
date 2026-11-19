import json
from datetime import datetime, timezone
from importlib.metadata import version
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from core import ROOT, CONFIG, NUM, CAT, FEATURES, load_data, make_pipeline, Repository


def metrics(y, prediction):
    return {"mae": float(mean_absolute_error(y, prediction)),
            "rmse": float(mean_squared_error(y, prediction) ** 0.5),
            "r2": float(r2_score(y, prediction))}


def run():
    train, validation, test = load_data()
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    results, best, best_score = [], None, float("inf")
    variants = []
    for name in CONFIG["models"]:
        if name == "ridge":
            variants += [(name, scale, added, k) for scale in CONFIG["scalers"]
                         for added in [False, True] for k in ["all", 15]]
        else:
            variants.append((name, "none", True, "all"))
    for name, scale, added, k in variants:
        model = make_pipeline(name, scale, added, k)
        model.fit(train[FEATURES], train["SalePrice"])
        row = {"model": name, "scaler": scale, "added": added, "k": k}
        for label, frame in [("train", train), ("val", validation)]:
            score = metrics(frame["SalePrice"], model.predict(frame[FEATURES]))
            row.update({f"{label}_{key}": value for key, value in score.items()})
        results.append(row)
        print(name, scale, added, k, round(row["val_mae"], 2))
        if row["val_mae"] < best_score:
            best, best_row, best_score = model, row, row["val_mae"]
    pd.DataFrame(results).to_csv(docs / "results.csv", index=False)
    prediction = best.predict(test[FEATURES])
    metadata = dict(best_row, version="v1", date=datetime.now(timezone.utc).isoformat(),
                    test=metrics(test["SalePrice"], prediction),
                    versions={name: version(name) for name in ["scikit-learn", "numpy", "pandas", "joblib"]})
    schema = {"feature_order": FEATURES,
              "types": {name: "number" for name in NUM} | {name: "string" for name in CAT},
              "categories": {name: sorted(train[name].dropna().unique().tolist()) for name in CAT},
              "derived_features": ["TotalSF", "HouseAge"] if best_row["added"] else []}
    schema["categories"]["Neighborhood"].append("Other")
    schema["categories"]["KitchenQual"] = ["Ex", "Gd", "TA", "Fa", "Po"]
    Repository().save(best, schema, metadata)
    sample = validation[FEATURES].head(5)
    reload_data = {"inputs": json.loads(sample.to_json(orient="records")),
                   "predictions": [round(float(value), 2) for value in best.predict(sample)]}
    (docs / "reload.json").write_text(json.dumps(reload_data, indent=2))
    for name, plot in [("predictions", "scatter"), ("residuals", "hist")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        if plot == "scatter":
            ax.scatter(test["SalePrice"], prediction, s=15)
            low, high = test["SalePrice"].min(), test["SalePrice"].max()
            ax.plot([low, high], [low, high], color="red")
            ax.set(xlabel="Actual price, USD", ylabel="Predicted price, USD")
        else:
            ax.hist(test["SalePrice"] - prediction, bins=25, edgecolor="black")
            ax.set(xlabel="Actual - predicted, USD", ylabel="Count")
        fig.tight_layout()
        fig.savefig(docs / f"{name}.png", dpi=150)
        plt.close(fig)
    print("Best:", best_row)
    print("Test:", metadata["test"])


if __name__ == "__main__":
    run()
