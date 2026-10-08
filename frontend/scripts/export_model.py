import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "tarifftruck1.csv"
OUTPUT_PATH = PROJECT_ROOT / "frontend" / "public" / "model.json"


def load_data(data_path):
    """Load dataset, analyze target leakage, and prepare feature/target matrices."""
    data = pd.read_csv(data_path)

    # Clean corrupted categorical values if present
    data["road"] = data["road"].replace({"yes": np.nan})
    data["seasonalimpact"] = data["seasonalimpact"].replace({"high": np.nan})
    data["demand"] = data["demand"].replace({"6": np.nan})

    # Target leakage investigation:
    # Check if tariff == labour + toll + miscellaneous
    misc_filled = data["miscellaneous"].fillna(0)
    calculated_tariff = data["labour"] + data["toll"] + misc_filled
    max_diff = float((data["tariff"] - calculated_tariff).abs().max())

    target_leakage_found = max_diff < 1e-5
    leakage_info = {
        "targetLeakageFound": target_leakage_found,
        "maxDifference": max_diff,
        "formula": "tariff = labour + toll + miscellaneous",
        "explanation": (
            "In the dataset, tariff is mathematically identical to labour + toll + miscellaneous "
            "(with 0 difference across all rows when missing miscellaneous is 0). Including labour, "
            "toll, and miscellaneous as predictive features constitutes target component leakage "
            "because post-hoc line-item costs directly sum to the target. To build an honest, "
            "leak-free predictive model for predicting tariff before trip execution, labour, toll, "
            "and miscellaneous are excluded from the model feature set."
        ),
    }

    # Exclude direct additive leakage features (labour, toll, miscellaneous) for leak-free modeling
    leakage_features = ["labour", "toll", "miscellaneous"]
    feature_cols = [col for col in data.columns if col != "tariff" and col not in leakage_features]

    features = data[feature_cols].copy()
    target = data["tariff"].copy()

    return features, target, leakage_info


def build_preprocessor(numeric_features, categorical_features):
    """Build leakage-safe scikit-learn ColumnTransformer."""
    numeric_preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_preprocessor, numeric_features),
            ("categorical", categorical_preprocessor, categorical_features),
        ]
    )


def get_candidate_models(random_state=42):
    """Return dictionary of candidate regression models to evaluate."""
    return {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=random_state),
        "Lasso Regression": Lasso(alpha=1.0, max_iter=10000, random_state=random_state),
        "ElasticNet": ElasticNet(alpha=1.0, max_iter=10000, random_state=random_state),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=random_state),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=random_state),
        "Support Vector Regression": SVR(),
    }


def evaluate_models(candidates, preprocessor, x_train, y_train, cv_splits=5, random_state=42):
    """Evaluate candidate models using K-fold cross-validation on the training set."""
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    cv_results = {}

    for name, regressor in candidates.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])
        scores = cross_validate(
            pipeline, x_train, y_train, cv=cv, scoring=scoring, return_train_score=True
        )
        cv_results[name] = {
            "cv_rmse_mean": float(-scores["test_rmse"].mean()),
            "cv_rmse_std": float(scores["test_rmse"].std()),
            "cv_mae_mean": float(-scores["test_mae"].mean()),
            "cv_r2_mean": float(scores["test_r2"].mean()),
            "train_r2_mean": float(scores["train_r2"].mean()),
        }

    return cv_results


def select_best_model(cv_results, primary_metric="cv_rmse_mean"):
    """Select the best model based on primary cross-validation metric (RMSE)."""
    return min(cv_results, key=lambda name: cv_results[name][primary_metric])


def train_final_model(candidate_models, selected_model_name, preprocessor, x_train, y_train):
    """Train the selected model on the entire training set."""
    regressor = candidate_models[selected_model_name]
    pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])
    pipeline.fit(x_train, y_train)
    return pipeline


def evaluate_final_model(model, x_train, y_train, x_test, y_test):
    """Evaluate final trained model on training set and held-out test set."""
    train_preds = model.predict(x_train)
    test_preds = model.predict(x_test)

    train_metrics = {
        "mae": float(mean_absolute_error(y_train, train_preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_train, train_preds))),
        "r2": float(r2_score(y_train, train_preds)),
    }
    test_metrics = {
        "mae": float(mean_absolute_error(y_test, test_preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, test_preds))),
        "r2": float(r2_score(y_test, test_preds)),
    }
    return train_metrics, test_metrics, test_preds


def export_tree_node(tree):
    """Convert scikit-learn Tree object into a JSON-serializable dictionary."""
    tree_stat = tree.tree_
    value_array = (
        tree_stat.value.squeeze(axis=(1, 2)).tolist()
        if tree_stat.value.ndim == 3
        else tree_stat.value.squeeze().tolist()
    )
    return {
        "childrenLeft": [int(x) for x in tree_stat.children_left],
        "childrenRight": [int(x) for x in tree_stat.children_right],
        "feature": [int(x) for x in tree_stat.feature],
        "threshold": [float(x) for x in tree_stat.threshold],
        "value": [float(x) for x in value_array] if isinstance(value_array, list) else [float(value_array)],
    }


def serialize_model_structure(regressor):
    """Serialize the trained regressor into JSON format for browser inference."""
    if hasattr(regressor, "coef_"):
        # Linear, Ridge, Lasso, ElasticNet
        return {
            "type": "linear",
            "intercept": float(regressor.intercept_),
            "coefficients": [float(val) for val in regressor.coef_],
        }
    elif hasattr(regressor, "tree_"):
        # Single Decision Tree
        return {
            "type": "tree",
            "tree": export_tree_node(regressor),
        }
    elif isinstance(regressor, (RandomForestRegressor, ExtraTreesRegressor)):
        # Forest ensemble
        return {
            "type": "forest",
            "trees": [export_tree_node(est) for est in regressor.estimators_],
        }
    elif isinstance(regressor, GradientBoostingRegressor):
        # Gradient Boosting ensemble
        init_val = float(regressor.init_.constant_[0][0]) if hasattr(regressor.init_, "constant_") else float(regressor.init_.mean_)
        return {
            "type": "gradient_boosting",
            "initValue": init_val,
            "learningRate": float(regressor.learning_rate),
            "trees": [export_tree_node(est[0]) for est in regressor.estimators_],
        }
    else:
        raise NotImplementedError(f"Export strategy for regressor {type(regressor)} is not implemented.")


def export_model(
    model,
    selected_model_name,
    numeric_features,
    categorical_features,
    x_train,
    x_test,
    y_train,
    y_test,
    cv_results,
    train_metrics,
    test_metrics,
    leakage_info,
    output_path,
):
    """Export trained model, preprocessing metadata, and metrics to model.json."""
    fitted_preprocessor = model.named_steps["preprocessor"]
    fitted_numeric = fitted_preprocessor.named_transformers_["numeric"]
    fitted_categorical = fitted_preprocessor.named_transformers_["categorical"]

    numeric_imputer = fitted_numeric.named_steps["imputer"]
    scaler = fitted_numeric.named_steps["scaler"]
    categorical_imputer = fitted_categorical.named_steps["imputer"]
    encoder = fitted_categorical.named_steps["encoder"]
    regressor = model.named_steps["regressor"]

    numeric_metadata = []
    for index, name in enumerate(numeric_features):
        series = x_train[name]
        numeric_metadata.append(
            {
                "name": name,
                "median": float(numeric_imputer.statistics_[index]),
                "mean": float(scaler.mean_[index]),
                "scale": float(scaler.scale_[index]),
                "minimum": float(series.min()),
                "maximum": float(series.max()),
            }
        )

    categorical_metadata = []
    for index, name in enumerate(categorical_features):
        categories = [str(value) for value in encoder.categories_[index]]
        categorical_metadata.append(
            {
                "name": name,
                "imputeValue": str(categorical_imputer.statistics_[index]),
                "categories": categories,
            }
        )

    example = {
        "year": 2015,
        "road": "nh",
        "seasonalimpact": "yes",
        "demand": "high",
        "wttons": 10.0,
        "distance": 1500.0,
        "fuelcost": 25000.0,
    }
    feature_cols = numeric_features + categorical_features
    example_frame = pd.DataFrame([example], columns=x_train.columns)
    python_prediction = float(model.predict(example_frame)[0])

    serialized_model = serialize_model_structure(regressor)
    serialized_model["name"] = selected_model_name

    selected_cv = cv_results[selected_model_name]

    artifact = {
        "model": serialized_model,
        "features": {
            "order": x_train.columns.tolist(),
            "numeric": numeric_metadata,
            "categorical": categorical_metadata,
        },
        "metrics": {
            "trainingRows": int(len(x_train)),
            "testRows": int(len(x_test)),
            "crossValidation": {
                "mae": float(selected_cv["cv_mae_mean"]),
                "rmse": float(selected_cv["cv_rmse_mean"]),
                "rmseStd": float(selected_cv["cv_rmse_std"]),
                "r2": float(selected_cv["cv_r2_mean"]),
                "trainR2": float(selected_cv["train_r2_mean"]),
            },
            "trainingSet": train_metrics,
            "heldOutTest": test_metrics,
        },
        "targetLeakage": leakage_info,
        "example": {
            "input": example,
            "prediction": python_prediction,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def main():
    print("=" * 80)
    print("FREIGHT TARIFF RATE PREDICTION MODEL TRAINING & EXPORT")
    print("=" * 80)

    # 1 & 2. Load dataset & Target Leakage Analysis
    features, target, leakage_info = load_data(DATA_PATH)
    print("\n[1] TARGET LEAKAGE INVESTIGATION:")
    if leakage_info["targetLeakageFound"]:
        print(f"    WARNING: Target leakage detected in raw data!")
        print(f"    Formula: {leakage_info['formula']}")
        print(f"    Action: Excluded direct additive features (labour, toll, miscellaneous).")
        print(f"    Modeling features used: {features.columns.tolist()}")

    # 3. Train/Test Split
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    print(f"\n[2] DATA SPLIT: Train rows = {len(x_train)}, Held-out Test rows = {len(x_test)}")

    # 4. Feature Identification
    numeric_features = x_train.select_dtypes(include="number").columns.tolist()
    categorical_features = [col for col in x_train.columns if col not in numeric_features]

    # 5. Preprocessor
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # 6. Evaluate Candidate Models
    candidate_models = get_candidate_models(random_state=42)
    print("\n[3] CROSS-VALIDATION EVALUATION (5-Fold CV on Training Set):")
    cv_results = evaluate_models(candidate_models, preprocessor, x_train, y_train, cv_splits=5)

    header = f"{'Model':28s} | {'CV RMSE (Mean)':15s} | {'CV RMSE (Std)':13s} | {'CV MAE':10s} | {'CV R2':8s} | {'Train R2':8s}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    sorted_cv = sorted(cv_results.items(), key=lambda item: item[1]["cv_rmse_mean"])
    for name, res in sorted_cv:
        print(
            f"{name:28s} | {res['cv_rmse_mean']:15.2f} | {res['cv_rmse_std']:13.2f} | "
            f"{res['cv_mae_mean']:10.2f} | {res['cv_r2_mean']:8.4f} | {res['train_r2_mean']:8.4f}"
        )
    print("-" * len(header))

    # 7 & 8. Select Best Model
    selected_model_name = select_best_model(cv_results)
    print(f"\n[4] MODEL SELECTION: Selected '{selected_model_name}' based on lowest CV RMSE.")

    # 9. Train Final Model
    final_model = train_final_model(
        candidate_models, selected_model_name, preprocessor, x_train, y_train
    )

    # 10 & 11. Evaluate Final Model
    train_metrics, test_metrics, test_preds = evaluate_final_model(
        final_model, x_train, y_train, x_test, y_test
    )
    print("\n[5] FINAL MODEL PERFORMANCE:")
    print(f"    Selected Model : {selected_model_name}")
    print(f"    CV RMSE        : {cv_results[selected_model_name]['cv_rmse_mean']:.2f}")
    print(f"    CV MAE         : {cv_results[selected_model_name]['cv_mae_mean']:.2f}")
    print(f"    CV R2          : {cv_results[selected_model_name]['cv_r2_mean']:.4f}")
    print(f"    Test RMSE      : {test_metrics['rmse']:.2f}")
    print(f"    Test MAE       : {test_metrics['mae']:.2f}")
    print(f"    Test R2        : {test_metrics['r2']:.4f}")

    # 12-17. Export Artifact
    artifact = export_model(
        final_model,
        selected_model_name,
        numeric_features,
        categorical_features,
        x_train,
        x_test,
        y_train,
        y_test,
        cv_results,
        train_metrics,
        test_metrics,
        leakage_info,
        OUTPUT_PATH,
    )

    print(f"\n[6] EXPORT SUCCESSFUL: Model exported to {OUTPUT_PATH}")
    print(f"    Example input prediction: {artifact['example']['prediction']:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
