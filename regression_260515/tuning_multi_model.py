import itertools
import pandas as pd
import numpy as np
import os
import random
import wandb
import matplotlib.pyplot as plt
import pickle
import datetime
import json
import hashlib
import shutil
import lightgbm as lgb
from types import SimpleNamespace
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from filelock import FileLock


# ================================================
# Common utilities
# ================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_cfg(config, key, default=None):
    if hasattr(config, key):
        value = getattr(config, key)
        return default if value is None else value
    try:
        value = config.get(key)
        return default if value is None else value
    except Exception:
        return default


def to_flat_array(y):
    return np.array(y).flatten()


def get_model_file_extension(model_type):
    return ".txt" if model_type == "lightgbm" else ".pkl"


def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def is_metric_improved(metric_name, current_value, best_values, key):
    if current_value is None:
        return False

    if key not in best_values or best_values[key] is None:
        return True

    if metric_name == "R2":
        return current_value > best_values[key]

    return current_value < best_values[key]


def make_sweep_id_key(sweep_name, sweep_config):
    config_json = json.dumps(to_jsonable(sweep_config), sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:12]
    return f"sweep_id_{sweep_name}_{config_hash}"


# ================================================
# Data class
# ================================================
class Data:
    def __init__(self):
        self.raw_data = None
        self.X = None
        self.Y = None
        self.scaler = None
        self.train_X = None
        self.train_Y = None
        self.val_X = None
        self.val_Y = None
        self.test_X = None
        self.test_Y = None

    def load_data(self, path):
        self.raw_data = pd.read_csv(path)
        self.raw_data.dropna(inplace=True)
        print("Data loaded successfully!")
        print(f"Data Shape: {self.raw_data.shape}")
        print(f"Columns: {self.raw_data.columns.tolist()}")

    def remove_outliers(self, columns, weight=1.5):
        outlier_idx = set()

        def get_outlier(df=None, column=None, weight=1.5):
            data = df[column]
            quantile_25 = np.percentile(data.values, 25)
            quantile_75 = np.percentile(data.values, 75)
            iqr = quantile_75 - quantile_25
            iqr_weight = iqr * weight
            lowest_val = quantile_25 - iqr_weight
            highest_val = quantile_75 + iqr_weight
            outlier_index = data[(data < lowest_val) | (data > highest_val)].index
            return outlier_index

        for col in columns:
            idx = get_outlier(df=self.raw_data, column=col, weight=weight)
            outlier_idx.update(idx)

        if outlier_idx:
            self.raw_data.drop(index=list(outlier_idx), inplace=True)
            print(f"Removed {len(outlier_idx)} outlier rows based on columns: {columns}")
        else:
            print("No outliers found.")

    def split_data(self, input_cols, output_cols):
        self.X = self.raw_data[input_cols]
        self.Y = self.raw_data[output_cols]

    def normalize_data(self):
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

        base_dir = globals().get("PATH", globals().get("BASE_PATH", "."))
        artifact_nm = globals().get("ARTIFACT_NM", "multi_model")
        model_dir = os.path.join(base_dir, "saved_models", artifact_nm)
        os.makedirs(model_dir, exist_ok=True)
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {scaler_path}")

    def split_train_val_test(self, test_size=0.2, val_size=0.2, random_state=42):
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=random_state
        )
        relative_val_size = val_size / (1 - test_size)
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train_val, Y_train_val, test_size=relative_val_size, random_state=random_state
        )

        self.train_X = X_train
        self.train_Y = Y_train
        self.val_X = X_val
        self.val_Y = Y_val
        self.test_X = X_test
        self.test_Y = Y_test
        print(f"Data split into: Train {self.train_X.shape}, Val {self.val_X.shape}, Test {self.test_X.shape}")


# ================================================
# Multi-model wrapper
# ================================================
class ModelWrapper(Data):
    def __init__(self, model_type="lightgbm"):
        super().__init__()
        self.model_type = model_type
        self.model = None
        self.Y_col = None
        self.best_val_loss = None

    def _build_sklearn_model(self, config):
        if self.model_type == "random_forest":
            max_depth = get_cfg(config, "max_depth", 20)
            max_leaf_nodes = get_cfg(config, "max_leaf_nodes", -1)
            return RandomForestRegressor(
                n_estimators=int(get_cfg(config, "n_estimators", 500)),
                criterion=get_cfg(config, "criterion", "squared_error"),
                max_depth=int(max_depth) if max_depth != -1 else None,
                min_samples_split=int(get_cfg(config, "min_samples_split", 2)),
                min_samples_leaf=int(get_cfg(config, "min_samples_leaf", 1)),
                max_features=get_cfg(config, "max_features", "sqrt"),
                max_leaf_nodes=int(max_leaf_nodes) if max_leaf_nodes != -1 else None,
                bootstrap=bool(get_cfg(config, "bootstrap", True)),
                ccp_alpha=float(get_cfg(config, "ccp_alpha", 0.0)),
                random_state=42,
                n_jobs=-1,
            )

        if self.model_type == "extra_trees":
            max_depth = get_cfg(config, "max_depth", 20)
            max_leaf_nodes = get_cfg(config, "max_leaf_nodes", -1)
            return ExtraTreesRegressor(
                n_estimators=int(get_cfg(config, "n_estimators", 500)),
                criterion=get_cfg(config, "criterion", "squared_error"),
                max_depth=int(max_depth) if max_depth != -1 else None,
                min_samples_split=int(get_cfg(config, "min_samples_split", 2)),
                min_samples_leaf=int(get_cfg(config, "min_samples_leaf", 1)),
                max_features=get_cfg(config, "max_features", "sqrt"),
                max_leaf_nodes=int(max_leaf_nodes) if max_leaf_nodes != -1 else None,
                bootstrap=bool(get_cfg(config, "bootstrap", False)),
                ccp_alpha=float(get_cfg(config, "ccp_alpha", 0.0)),
                random_state=42,
                n_jobs=-1,
            )

        if self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                loss=get_cfg(config, "loss", "squared_error"),
                n_estimators=int(get_cfg(config, "n_estimators", 500)),
                learning_rate=float(get_cfg(config, "lr", 0.03)),
                max_depth=int(get_cfg(config, "max_depth", 5)),
                subsample=float(get_cfg(config, "subsample", 0.8)),
                criterion=get_cfg(config, "criterion", "friedman_mse"),
                min_samples_split=int(get_cfg(config, "min_samples_split", 2)),
                min_samples_leaf=int(get_cfg(config, "min_samples_leaf", 2)),
                max_features=get_cfg(config, "max_features", None),
                alpha=float(get_cfg(config, "alpha", 0.9)),
                ccp_alpha=float(get_cfg(config, "ccp_alpha", 0.0)),
                random_state=42,
            )

        raise ValueError(f"Unsupported model_type for sklearn model: {self.model_type}")

    def train_model(self, config):
        y_train = to_flat_array(self.train_Y)
        y_val = to_flat_array(self.val_Y)

        if self.model_type == "lightgbm":
            params = {
                "objective": "regression",
                "metric": "l2",
                "learning_rate": float(get_cfg(config, "lr", 1e-3)),
                "num_leaves": int(get_cfg(config, "num_leaves", 31)),
                "max_depth": int(get_cfg(config, "max_depth", -1)),
                "min_child_samples": int(get_cfg(config, "min_child_samples", 20)),
                "subsample": float(get_cfg(config, "subsample", 0.8)),
                "colsample_bytree": float(get_cfg(config, "colsample_bytree", 0.8)),
                "reg_alpha": float(get_cfg(config, "reg_alpha", 0.0)),
                "reg_lambda": float(get_cfg(config, "reg_lambda", 0.0)),
                "verbosity": -1,
                "seed": 42,
            }

            train_data = lgb.Dataset(self.train_X, label=y_train)
            valid_data = lgb.Dataset(self.val_X, label=y_val, reference=train_data)

            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=int(get_cfg(config, "n_estimators", 500)),
                valid_sets=[valid_data],
                valid_names=["valid"],
                callbacks=[
                    lgb.early_stopping(int(get_cfg(config, "patience", 20))),
                    lgb.log_evaluation(0),
                ],
            )
            score_dict = self.model.best_score.get("valid", {})
            self.best_val_loss = float(score_dict.get("l2")) if "l2" in score_dict else None
            return self.model

        self.model = self._build_sklearn_model(config)
        self.model.fit(self.train_X, y_train)

        y_val_pred = self.model.predict(self.val_X)
        self.best_val_loss = float(mean_squared_error(y_val, y_val_pred))
        return self.model

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not trained yet.")

        if self.model_type == "lightgbm":
            best_iteration = getattr(self.model, "best_iteration", None)
            return self.model.predict(X, num_iteration=best_iteration)

        return self.model.predict(X)

    def evaluate_split(self, X, y):
        y_pred = self.predict(X)
        y_true = to_flat_array(y)

        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        return {
            "MAE": mae,
            "MAPE": mape,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
        }, y_pred

    def save_model(self, model_path):
        if self.model_type == "lightgbm":
            self.model.save_model(model_path)
        else:
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)

    def plot_scatter(self, output_col, X_data, y_data, save_path=None, metrics=None):
        y_pred = self.predict(X_data)
        y_true = to_flat_array(y_data)

        if metrics is None:
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            metrics = {
                "R2": r2,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "MAPE": mape,
            }

        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", label="Data points")

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal Fit")

        slope, intercept = np.polyfit(y_true, y_pred, 1)
        reg_line = slope * np.array([min_val, max_val]) + intercept
        plt.plot([min_val, max_val], reg_line, "b-", lw=2, label="Regression Line")

        plt.grid(True, linestyle="--", linewidth=1)
        plt.xlabel("Actual Values", fontsize=24)
        plt.ylabel("Predicted Values", fontsize=24)
        plt.title(f"Scatter Plot for {output_col} ({self.model_type})", fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)

        ordered_keys = ["R2", "MAE", "MSE", "RMSE", "MAPE"]
        lines = []
        for key in ordered_keys:
            value = metrics.get(key, None)
            if value is None:
                line = f"{key}: N/A"
            elif key == "R2":
                line = f"R2: {value:.4f}"
            elif key == "MAPE":
                line = f"MAPE: {value * 100:.2f}%"
            else:
                line = f"{key}: {value:.4f}"
            lines.append(line)

        metrics_text = "\n".join(lines)
        plt.gca().text(
            0.05,
            0.95,
            metrics_text,
            transform=plt.gca().transAxes,
            fontsize=20,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Scatter plot saved to {save_path}")
        plt.close()


# ================================================
# Robust ensemble helper (outlier mitigation)
# ================================================
def robust_ensemble_predict(predictions_by_model, method="median"):
    if not predictions_by_model:
        raise ValueError("predictions_by_model is empty")

    stacked = np.vstack(predictions_by_model)
    if method == "median":
        return np.median(stacked, axis=0)

    if method == "trimmed_mean":
        sorted_preds = np.sort(stacked, axis=0)
        if sorted_preds.shape[0] <= 2:
            return np.mean(sorted_preds, axis=0)
        return np.mean(sorted_preds[1:-1, :], axis=0)

    raise ValueError(f"Unsupported ensemble method: {method}")


class RobustEnsembleModel:
    def __init__(self, members, method="median", bundle_dir=None):
        self.members = members
        self.method = method
        self.bundle_dir = bundle_dir
        self._loaded_models = {}

    def _load_member_model(self, member):
        model_type = member["model_type"]
        model_path = member["model_path"]
        if not os.path.isabs(model_path) and self.bundle_dir:
            model_path = os.path.join(self.bundle_dir, model_path)
        cache_key = (model_type, model_path)

        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        if model_type == "lightgbm":
            model = lgb.Booster(model_file=model_path)
        else:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

        self._loaded_models[cache_key] = model
        return model

    def _predict_single(self, model_type, model, X):
        if model_type == "lightgbm":
            best_iteration = getattr(model, "best_iteration", None)
            return model.predict(X, num_iteration=best_iteration)
        return model.predict(X)

    def predict(self, X):
        predictions = []
        for member in self.members:
            model = self._load_member_model(member)
            pred = self._predict_single(member["model_type"], model, X)
            predictions.append(pred)
        return robust_ensemble_predict(predictions, method=self.method)


def save_ensemble_bundle(
    base_path,
    target_config,
    seed,
    ensemble_method,
    wrappers_by_model,
    input_cols,
    output_cols,
    model_metrics,
    ensemble_metrics,
    member_train_configs=None,
    member_config_sources=None,
    run_tag="baseline",
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_root = os.path.join(
        base_path,
        "best_model",
        f"{target_config['wandb_project']}_{target_config['target_name']}_ensemble_{ensemble_method}_{run_tag}",
    )
    bundle_dir = os.path.join(bundle_root, f"seed_{seed}_{timestamp}")
    os.makedirs(bundle_dir, exist_ok=True)

    members = []
    for model_type, wrapper in wrappers_by_model.items():
        extension = get_model_file_extension(model_type)
        model_filename = f"member_{model_type}{extension}"
        model_path = os.path.join(bundle_dir, model_filename)
        wrapper.save_model(model_path)
        members.append(
            {
                "model_type": model_type,
                "model_path": model_path,
                "model_filename": model_filename,
            }
        )

    ensemble_model = RobustEnsembleModel(
        members=[{"model_type": m["model_type"], "model_path": m["model_filename"]} for m in members],
        method=ensemble_method,
        bundle_dir=bundle_dir,
    )
    ensemble_model_path = os.path.join(bundle_dir, "ensemble_model.pkl")
    with open(ensemble_model_path, "wb") as f:
        pickle.dump(ensemble_model, f)

    ensemble_config = {
        "target_name": target_config["target_name"],
        "source_file": target_config["file"],
        "wandb_project": target_config["wandb_project"],
        "seed": seed,
        "ensemble_method": ensemble_method,
        "input_columns": list(input_cols),
        "output_columns": list(output_cols),
        "members": members,
        "model_test_metrics": to_jsonable(model_metrics),
        "ensemble_test_metrics": to_jsonable(ensemble_metrics),
        "member_train_configs": to_jsonable(member_train_configs or {}),
        "member_config_sources": to_jsonable(member_config_sources or {}),
        "ensemble_model_path": ensemble_model_path,
    }
    ensemble_config_path = os.path.join(bundle_dir, "ensemble_config.json")
    with open(ensemble_config_path, "w") as f:
        json.dump(to_jsonable(ensemble_config), f, indent=2)

    return {
        "bundle_root": bundle_root,
        "bundle_dir": bundle_dir,
        "ensemble_model_path": ensemble_model_path,
        "ensemble_config_path": ensemble_config_path,
    }


# ================================================
# Best model artifact update
# ================================================
def update_best_model_artifact(splits, model_file, config_file, scatter_files=None):
    best_model_dir = os.path.join(PATH, "best_model", ARTIFACT_NM)
    os.makedirs(best_model_dir, exist_ok=True)
    best_values_file = os.path.join(best_model_dir, "best_values.json")

    if os.path.exists(best_values_file):
        with open(best_values_file, "r") as f:
            best_values = json.load(f)
    else:
        best_values = {}

    improved_any = False
    improvements_description = []

    key = "best_val_loss"
    current_val = splits.get("best_val_loss")
    improved = False
    if key not in best_values:
        improved = True
    else:
        old_val = best_values[key]
        if old_val is None and current_val is not None:
            improved = True
        elif current_val is not None and current_val < old_val:
            improved = True

    if improved:
        formatted_old = "N/A" if key not in best_values or best_values[key] is None else f"{float(best_values[key]):.4f}"
        formatted_current = "None" if current_val is None else f"{current_val:.4f}"

        improvements_description.append(f"{key} improved from {formatted_old} to {formatted_current}")
        best_values[key] = current_val
        improved_any = True

        key_dir = os.path.join(best_model_dir, key)
        os.makedirs(key_dir, exist_ok=True)

        model_ext = os.path.splitext(model_file)[1] if os.path.splitext(model_file)[1] else ".bin"
        model_dest = os.path.join(key_dir, f"model{model_ext}")
        config_dest = os.path.join(key_dir, "config.json")

        shutil.copyfile(model_file, model_dest)
        shutil.copyfile(config_file, config_dest)
        print(f"Updated best {key} model with value: {formatted_current}")

        if scatter_files:
            fixed_names = ["train_scatter.png", "val_scatter.png", "test_scatter.png"]
            for scatter_path, fixed_name in zip(scatter_files, fixed_names):
                if os.path.exists(scatter_path):
                    dest = os.path.join(key_dir, fixed_name)
                    shutil.copyfile(scatter_path, dest)
                    print(f"Updated scatter plot for {key} in {dest}")

    for split, metrics in splits.items():
        if split == "best_val_loss":
            continue

        for metric_name, current_value in metrics.items():
            key = f"{split}_{metric_name}"
            improved = False

            if key not in best_values:
                improved = True
            else:
                best_value = best_values[key]
                if best_value is None and current_value is not None:
                    improved = True
                elif current_value is not None:
                    if metric_name == "R2" and current_value > best_value:
                        improved = True
                    if metric_name != "R2" and current_value < best_value:
                        improved = True

            if improved:
                formatted_old = "N/A" if key not in best_values or best_values[key] is None else f"{float(best_values[key]):.4f}"
                formatted_current = "None" if current_value is None else f"{current_value:.4f}"

                improvements_description.append(f"{key} improved from {formatted_old} to {formatted_current}")
                best_values[key] = current_value
                improved_any = True

                key_dir = os.path.join(best_model_dir, key)
                os.makedirs(key_dir, exist_ok=True)

                model_ext = os.path.splitext(model_file)[1] if os.path.splitext(model_file)[1] else ".bin"
                model_dest = os.path.join(key_dir, f"model{model_ext}")
                config_dest = os.path.join(key_dir, "config.json")

                shutil.copyfile(model_file, model_dest)
                shutil.copyfile(config_file, config_dest)
                print(f"Updated best {key} model with value: {formatted_current}")

                if scatter_files:
                    fixed_names = ["train_scatter.png", "val_scatter.png", "test_scatter.png"]
                    for scatter_path, fixed_name in zip(scatter_files, fixed_names):
                        if os.path.exists(scatter_path):
                            dest = os.path.join(key_dir, fixed_name)
                            shutil.copyfile(scatter_path, dest)
                            print(f"Updated scatter plot for {key} in {dest}")

    with open(best_values_file, "w") as f:
        json.dump(best_values, f)

    description_str = " ; ".join(improvements_description) if improvements_description else "No improvements this run."

    if improved_any:
        run = wandb.run
        art = wandb.Artifact(f"{ARTIFACT_NM}", type="model", description=description_str, metadata=best_values)
        art.add_dir(best_model_dir, name="best_model")
        run.log_artifact(art)
        print(f"Logged best_model artifact with description: {description_str}")
    else:
        print("No improvements this run. Best model artifact not updated.")

    return improved_any


# ================================================
# W&B sweep training
# ================================================
def sweep_train():
    wandb.init(project=WANDB_PR, entity="schwalbe-university-of-seoul")
    config = wandb.config

    global model_instance
    wrapper = model_instance

    y_train = wrapper.train_Y[wrapper.Y_col[0]] if isinstance(wrapper.train_Y, pd.DataFrame) else wrapper.train_Y
    y_val = wrapper.val_Y[wrapper.Y_col[0]] if isinstance(wrapper.val_Y, pd.DataFrame) else wrapper.val_Y
    y_test = wrapper.test_Y[wrapper.Y_col[0]] if isinstance(wrapper.test_Y, pd.DataFrame) else wrapper.test_Y

    X_train = wrapper.train_X
    X_val = wrapper.val_X
    X_test = wrapper.test_X

    wrapper.train_model(config)

    metrics_train, _ = wrapper.evaluate_split(X_train, y_train)
    metrics_val, _ = wrapper.evaluate_split(X_val, y_val)
    metrics_test, _ = wrapper.evaluate_split(X_test, y_test)

    best_val_loss = wrapper.best_val_loss

    wandb.log(
        {
            "model_type": wrapper.model_type,
            "final_train_metrics": metrics_train,
            "final_val_metrics": metrics_val,
            "final_test_metrics": metrics_test,
            "best_val_loss": best_val_loss,
        }
    )

    splits = {"train": metrics_train, "val": metrics_val, "test": metrics_test, "best_val_loss": best_val_loss}

    base_best_values_file = os.path.join(PATH, "best_model", ARTIFACT_NM, "best_values.json")
    if os.path.exists(base_best_values_file):
        with open(base_best_values_file, "r") as f:
            base_best_values = json.load(f)
    else:
        base_best_values = {}

    improved_flag = False

    key = "best_val_loss"
    current_val = splits.get("best_val_loss")
    if is_metric_improved(key, current_val, base_best_values, key):
        improved_flag = True

    for split, metrics in splits.items():
        if split == "best_val_loss":
            continue

        for metric_name, current_value in metrics.items():
            key = f"{split}_{metric_name}"
            if is_metric_improved(metric_name, current_value, base_best_values, key):
                improved_flag = True

    if improved_flag:
        best_model_dir = os.path.join(PATH, "best_model", ARTIFACT_NM)
        os.makedirs(best_model_dir, exist_ok=True)

        model_extension = ".txt" if wrapper.model_type == "lightgbm" else ".pkl"
        model_file = os.path.join(best_model_dir, f"best_model{model_extension}")
        config_file = os.path.join(best_model_dir, "best_config.json")

        wrapper.save_model(model_file)
        with open(config_file, "w") as f:
            config_dict = dict(config)
            config_dict["model_type"] = wrapper.model_type
            json.dump(config_dict, f)

        scatter_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        train_scatter = os.path.join(best_model_dir, f"{wrapper.Y_col[0]}_train_scatter_{scatter_timestamp}.png")
        val_scatter = os.path.join(best_model_dir, f"{wrapper.Y_col[0]}_val_scatter_{scatter_timestamp}.png")
        test_scatter = os.path.join(best_model_dir, f"{wrapper.Y_col[0]}_test_scatter_{scatter_timestamp}.png")

        wrapper.plot_scatter(wrapper.Y_col[0], X_train, y_train, save_path=train_scatter, metrics=metrics_train)
        wrapper.plot_scatter(wrapper.Y_col[0], X_val, y_val, save_path=val_scatter, metrics=metrics_val)
        wrapper.plot_scatter(wrapper.Y_col[0], X_test, y_test, save_path=test_scatter, metrics=metrics_test)

        update_best_model_artifact(splits, model_file, config_file, scatter_files=[train_scatter, val_scatter, test_scatter])

    wandb.finish()


# ================================================
# Sweep ID helpers
# ================================================
def get_or_create_sweep_id(sweep_config, project):
    lock_path = SWEEP_ID_PATH + ".lock"
    with FileLock(lock_path):
        if os.path.exists(SWEEP_ID_PATH):
            with open(SWEEP_ID_PATH, "r") as f:
                try:
                    sweep_data = json.load(f)
                except json.decoder.JSONDecodeError:
                    sweep_data = {}
        else:
            sweep_data = {}

        key = make_sweep_id_key(SWEEP_NM, sweep_config)
        if key in sweep_data:
            sweep_id = sweep_data[key]
            try:
                _ = wandb.Api().sweep(f"schwalbe-university-of-seoul/{project}/{sweep_id}")
                print(f"Loaded existing sweep_id for current config: {sweep_id}")
                return sweep_id
            except Exception as e:
                print(f"Existing sweep_id {sweep_id} is unavailable. Recreating sweep. Error: {e}")
                sweep_data.pop(key, None)

        sweep_id = wandb.sweep(sweep_config, project=project, entity="schwalbe-university-of-seoul")
        sweep_data[key] = sweep_id

        os.makedirs(os.path.dirname(SWEEP_ID_PATH), exist_ok=True)
        with open(SWEEP_ID_PATH, "w") as f:
            json.dump(sweep_data, f)

        print(f"Created new sweep_id: {sweep_id}")
        return sweep_id


# ================================================
# Config
# ================================================
NUM_WORKER = 1
NORMALIZE = False
BASE_PATH = "/gpfs/home1/r1jae262/jupyter/MFT_1MW/MFT_1MW_2026/regression_260515"
RUN_MODE = "ensemble_best"  # "sweep", "ensemble_baseline", or "ensemble_best"
SWEEP_COUNT = 10
OUTER_REPEAT = 100
ENSEMBLE_METHOD = "median"  # "median" or "trimmed_mean"
STRICT_BEST_CONFIG = True  # if True, raise error when best sweep config is missing

MODEL_TYPES = ["lightgbm", "random_forest", "extra_trees", "gradient_boosting"]

BASE_TARGET_CONFIGS = [
    {"target_name": "Lmt", "file": "data_Lmt.csv", "wandb_project": "MFT_1MW_260518"},
    {"target_name": "Llt", "file": "data_Llt.csv", "wandb_project": "MFT_1MW_260518"},
    {"target_name": "Tx_loss", "file": "data_Tx_loss.csv", "wandb_project": "MFT_1MW_260518"},
    {"target_name": "Rx_loss", "file": "data_Rx_loss.csv", "wandb_project": "MFT_1MW_260518"},
    {"target_name": "P_main_winding_inner", "file": "data_P_Tx_main_winding_inner.csv", "wandb_project": "MFT_1MW_260518"},
    {"target_name": "P_main_winding_outer", "file": "data_P_Tx_main_winding_outer.csv", "wandb_project": "MFT_1MW_260518"},
    {"target_name": "P_side_winding_inner", "file": "data_P_Tx_side_winding_inner.csv", "wandb_project": "MFT_1MW_260518"},
    {"target_name": "P_side_winding_outer", "file": "data_P_Tx_side_winding_outer.csv", "wandb_project": "MFT_1MW_260518"},
    {"target_name": "time", "file": "data_time.csv", "wandb_project": "MFT_1MW_260518"},
]

MODEL_CONFIGS = []
for target_cfg in BASE_TARGET_CONFIGS:
    for model_type in MODEL_TYPES:
        pretty_model_nm = "LightGBM" if model_type == "lightgbm" else model_type
        MODEL_CONFIGS.append(
            {
                "name": f"{target_cfg['target_name']}_{pretty_model_nm}_260515",
                "target_name": target_cfg["target_name"],
                "file": target_cfg["file"],
                "wandb_project": target_cfg["wandb_project"],
                "model_type": model_type,
            }
        )


# ================================================
# Runtime
# ================================================
def run_pipeline(seed, model_config):
    global PATH, SWEEP_NM, FILE_NAME, WANDB_PR, SWEEP_ID_PATH, ARTIFACT_NM

    PATH = BASE_PATH
    SWEEP_NM = model_config["name"]
    FILE_NAME = model_config["file"]
    WANDB_PR = model_config["wandb_project"]
    SWEEP_ID_PATH = f"{PATH}/wandb_id/sweep_id.json"
    ARTIFACT_NM = f"{WANDB_PR}_{SWEEP_NM}"

    print(f"\n=== Running with seed {seed} for model {SWEEP_NM} ({model_config['model_type']}) ===")
    set_seed(seed)

    wrapper = ModelWrapper(model_type=model_config["model_type"])
    data_path = f"{PATH}/{FILE_NAME}"
    wrapper.load_data(data_path)

    input_cols = wrapper.raw_data.columns[:-1]
    output_cols = wrapper.raw_data.columns[-1:].tolist()

    wrapper.split_data(input_cols, output_cols)
    wrapper.Y_col = output_cols

    if NORMALIZE:
        wrapper.normalize_data()

    wrapper.split_train_val_test(test_size=0.2, val_size=0.2, random_state=seed)

    global model_instance
    model_instance = wrapper


def get_sweep_config(model_type):
    if model_type == "lightgbm":
        return {
            "name": SWEEP_NM,
            "method": "bayes",
            "metric": {"name": "best_val_loss", "goal": "minimize"},
            "parameters": {
                "lr": {
                    "values": [
                        1e-6, 2e-6, 3e-6, 5e-6,
                        1e-5, 2e-5, 3e-5, 5e-5,
                        1e-4, 2e-4, 3e-4, 5e-4,
                        1e-3, 2e-3, 3e-3, 5e-3,
                        1e-2, 2e-2, 3e-2, 5e-2,
                    ]
                },
                "num_leaves": {"values": [7, 15, 31, 63, 127, 255]},
                "max_depth": {"values": [-1, 3, 5, 7, 10, 15, 20]},
                "n_estimators": {"values": [50, 100, 150, 200, 300, 500, 1000, 2000]},
                "patience": {"values": [10, 15, 20, 25, 30, 40]},
                "min_child_samples": {"values": [5, 10, 20, 30, 50]},
                "subsample": {"values": [0.6, 0.7, 0.8, 0.9, 1.0]},
                "colsample_bytree": {"values": [0.6, 0.7, 0.8, 0.9, 1.0]},
                "reg_alpha": {"values": [0, 1e-3, 1e-2, 1e-1, 1]},
                "reg_lambda": {"values": [0, 1e-3, 1e-2, 1e-1, 1]},
            },
        }

    if model_type in {"random_forest", "extra_trees"}:
        return {
            "name": SWEEP_NM,
            "method": "bayes",
            "metric": {"name": "best_val_loss", "goal": "minimize"},
            "parameters": {
                "n_estimators": {"values": [100, 200, 300, 500, 800, 1200, 1600, 2000]},
                "criterion": {"values": ["squared_error", "friedman_mse"]},
                "max_depth": {"values": [-1, 5, 8, 12, 16, 24, 32, 48, 64]},
                "min_samples_split": {"values": [2, 4, 8, 16, 32]},
                "min_samples_leaf": {"values": [1, 2, 4, 8, 16, 32]},
                "max_features": {"values": ["sqrt", "log2", 0.25, 0.4, 0.6, 0.8, 1.0]},
                "max_leaf_nodes": {"values": [-1, 64, 128, 256, 512, 1024, 2048]},
                "bootstrap": {"values": [True, False]},
                "ccp_alpha": {"values": [0.0, 1e-6, 1e-5, 1e-4, 1e-3]},
            },
        }

    if model_type == "gradient_boosting":
        return {
            "name": SWEEP_NM,
            "method": "bayes",
            "metric": {"name": "best_val_loss", "goal": "minimize"},
            "parameters": {
                "loss": {"values": ["squared_error", "absolute_error", "huber"]},
                "n_estimators": {"values": [100, 200, 300, 500, 800, 1200, 1600, 2000]},
                "lr": {"values": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 1e-1]},
                "max_depth": {"values": [1, 2, 3, 4, 5, 7, 10]},
                "subsample": {"values": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
                "criterion": {"values": ["friedman_mse", "squared_error"]},
                "min_samples_split": {"values": [2, 4, 8, 16, 32]},
                "min_samples_leaf": {"values": [1, 2, 4, 8, 16, 32]},
                "max_features": {"values": ["sqrt", "log2", 0.25, 0.4, 0.6, 0.8, 1.0]},
                "alpha": {"values": [0.75, 0.85, 0.9, 0.95, 0.99]},
                "ccp_alpha": {"values": [0.0, 1e-6, 1e-5, 1e-4, 1e-3]},
            },
        }

    raise ValueError(f"Unsupported model_type: {model_type}")


def get_default_model_params(model_type):
    if model_type == "lightgbm":
        return {
            "lr": 1e-3,
            "num_leaves": 63,
            "max_depth": -1,
            "n_estimators": 500,
            "patience": 30,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
        }

    if model_type in {"random_forest", "extra_trees"}:
        return {
            "n_estimators": 800,
            "criterion": "squared_error",
            "max_depth": 24,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "max_leaf_nodes": -1,
            "bootstrap": True if model_type == "random_forest" else False,
            "ccp_alpha": 0.0,
        }

    if model_type == "gradient_boosting":
        return {
            "loss": "squared_error",
            "n_estimators": 500,
            "lr": 0.03,
            "max_depth": 5,
            "subsample": 0.8,
            "criterion": "friedman_mse",
            "min_samples_split": 2,
            "min_samples_leaf": 2,
            "max_features": None,
            "alpha": 0.9,
            "ccp_alpha": 0.0,
        }

    raise ValueError(f"Unsupported model_type: {model_type}")


MODEL_HPARAM_KEYS = {
    "lightgbm": {
        "lr",
        "num_leaves",
        "max_depth",
        "n_estimators",
        "patience",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    },
    "random_forest": {
        "n_estimators",
        "criterion",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "max_leaf_nodes",
        "bootstrap",
        "ccp_alpha",
    },
    "extra_trees": {
        "n_estimators",
        "criterion",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "max_leaf_nodes",
        "bootstrap",
        "ccp_alpha",
    },
    "gradient_boosting": {
        "loss",
        "n_estimators",
        "lr",
        "max_depth",
        "subsample",
        "criterion",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "alpha",
        "ccp_alpha",
    },
}


def filter_hparams_for_model(model_type, config_dict):
    keys = MODEL_HPARAM_KEYS.get(model_type)
    if keys is None:
        raise ValueError(f"Unsupported model_type for hyperparameter filtering: {model_type}")
    return {k: config_dict[k] for k in keys if k in config_dict}


def find_model_config_for_target(target_config, model_type):
    matches = [
        cfg
        for cfg in MODEL_CONFIGS
        if cfg["model_type"] == model_type
        and cfg["file"] == target_config["file"]
        and cfg["wandb_project"] == target_config["wandb_project"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one MODEL_CONFIG for target={target_config['target_name']} and model={model_type}, "
            f"but found {len(matches)}"
        )
    return matches[0]


def resolve_best_config_file(base_path, model_config):
    artifact_nm = f"{model_config['wandb_project']}_{model_config['name']}"
    artifact_dir = os.path.join(base_path, "best_model", artifact_nm)

    if not os.path.isdir(artifact_dir):
        return None, artifact_dir

    prioritized_candidates = [
        os.path.join(artifact_dir, "best_val_loss", "config.json"),
        os.path.join(artifact_dir, "best_config.json"),
    ]
    for path in prioritized_candidates:
        if os.path.exists(path):
            return path, artifact_dir

    recursive_candidates = []
    for root, _, files in os.walk(artifact_dir):
        if "config.json" in files:
            recursive_candidates.append(os.path.join(root, "config.json"))

    if recursive_candidates:
        recursive_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return recursive_candidates[0], artifact_dir

    return None, artifact_dir


def load_training_config_for_model(base_path, target_config, model_type, strict=True):
    model_config = find_model_config_for_target(target_config, model_type)
    best_config_path, artifact_dir = resolve_best_config_file(base_path, model_config)

    default_cfg = get_default_model_params(model_type).copy()

    if best_config_path is None:
        if strict:
            raise FileNotFoundError(
                f"Best config not found for model={model_type}, target={target_config['target_name']}. "
                f"Expected under: {artifact_dir}. Run RUN_MODE='sweep' first for this model."
            )
        merged_cfg = default_cfg
        cfg_source = "default_fallback"
    else:
        with open(best_config_path, "r") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"Best config file is not a dict: {best_config_path}")

        tuned_hparams = filter_hparams_for_model(model_type, loaded)
        merged_cfg = default_cfg
        merged_cfg.update(tuned_hparams)
        cfg_source = best_config_path

    merged_cfg["model_type"] = model_type
    return SimpleNamespace(**merged_cfg), merged_cfg, cfg_source


def run_sweep(model_type):
    sweep_config = get_sweep_config(model_type)
    sweep_id = get_or_create_sweep_id(sweep_config, project=WANDB_PR)

    print(f"Using sweep id: {sweep_id}")
    wandb.agent(
        sweep_id,
        function=sweep_train,
        count=SWEEP_COUNT,
        project=WANDB_PR,
        entity="schwalbe-university-of-seoul",
    )


def run_ensemble_baseline(seed, target_config, ensemble_method="median"):
    set_seed(seed)

    dataset_path = os.path.join(BASE_PATH, target_config["file"])
    base_data = Data()
    base_data.load_data(dataset_path)

    input_cols = base_data.raw_data.columns[:-1]
    output_cols = base_data.raw_data.columns[-1:].tolist()
    base_data.split_data(input_cols, output_cols)
    if NORMALIZE:
        base_data.normalize_data()
    base_data.split_train_val_test(test_size=0.2, val_size=0.2, random_state=seed)

    y_test = base_data.test_Y[output_cols[0]] if isinstance(base_data.test_Y, pd.DataFrame) else base_data.test_Y
    y_test = to_flat_array(y_test)

    predictions = []
    model_metrics = {}
    wrappers_by_model = {}
    member_train_configs = {}
    member_config_sources = {}

    print(f"\n=== Ensemble baseline: {target_config['target_name']} / seed={seed} / method={ensemble_method} ===")
    for model_type in MODEL_TYPES:
        wrapper = ModelWrapper(model_type=model_type)
        wrapper.Y_col = output_cols
        wrapper.train_X = base_data.train_X
        wrapper.train_Y = base_data.train_Y
        wrapper.val_X = base_data.val_X
        wrapper.val_Y = base_data.val_Y
        wrapper.test_X = base_data.test_X
        wrapper.test_Y = base_data.test_Y

        cfg_dict = get_default_model_params(model_type).copy()
        cfg = SimpleNamespace(**cfg_dict)
        wrapper.train_model(cfg)
        metrics_test, pred_test = wrapper.evaluate_split(wrapper.test_X, wrapper.test_Y)

        model_metrics[model_type] = metrics_test
        wrappers_by_model[model_type] = wrapper
        member_train_configs[model_type] = cfg_dict
        member_config_sources[model_type] = "default_baseline"
        predictions.append(pred_test)
        print(f"[{model_type}] test MAE={metrics_test['MAE']:.6f}, RMSE={metrics_test['RMSE']:.6f}, R2={metrics_test['R2']:.6f}")

    ensemble_pred = robust_ensemble_predict(predictions, method=ensemble_method)
    ensemble_metrics = {
        "MAE": float(mean_absolute_error(y_test, ensemble_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_test, ensemble_pred)),
        "MSE": float(mean_squared_error(y_test, ensemble_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, ensemble_pred))),
        "R2": float(r2_score(y_test, ensemble_pred)),
    }

    print(
        f"[ensemble:{ensemble_method}] test MAE={ensemble_metrics['MAE']:.6f}, "
        f"RMSE={ensemble_metrics['RMSE']:.6f}, R2={ensemble_metrics['R2']:.6f}"
    )

    bundle_info = save_ensemble_bundle(
        base_path=BASE_PATH,
        target_config=target_config,
        seed=seed,
        ensemble_method=ensemble_method,
        wrappers_by_model=wrappers_by_model,
        input_cols=input_cols,
        output_cols=output_cols,
        model_metrics=model_metrics,
        ensemble_metrics=ensemble_metrics,
        member_train_configs=member_train_configs,
        member_config_sources=member_config_sources,
        run_tag="baseline",
    )
    print(f"Saved ensemble bundle to {bundle_info['bundle_dir']}")

    wandb.init(
        project=target_config["wandb_project"],
        entity="schwalbe-university-of-seoul",
        name=f"{target_config['target_name']}_ensemble_{ensemble_method}_{seed}",
        config={
            "mode": "ensemble_baseline",
            "target_name": target_config["target_name"],
            "file": target_config["file"],
            "ensemble_method": ensemble_method,
            "seed": seed,
            "model_types": MODEL_TYPES,
        },
    )
    wandb.log(
        {
            "ensemble_test_metrics": to_jsonable(ensemble_metrics),
            "model_test_metrics": to_jsonable(model_metrics),
            "member_train_configs": to_jsonable(member_train_configs),
            "member_config_sources": to_jsonable(member_config_sources),
            "ensemble_bundle_path": bundle_info["bundle_dir"],
        }
    )

    ensemble_artifact = wandb.Artifact(
        name=f"{target_config['target_name']}_ensemble_{ensemble_method}",
        type="ensemble_model",
        description=f"Robust ensemble bundle ({ensemble_method})",
        metadata={
            "target_name": target_config["target_name"],
            "ensemble_method": ensemble_method,
            "seed": seed,
            "model_types": MODEL_TYPES,
        },
    )
    ensemble_artifact.add_dir(bundle_info["bundle_dir"], name="ensemble_bundle")
    wandb.log_artifact(ensemble_artifact)
    wandb.finish()


def run_ensemble_best(seed, target_config, ensemble_method="median", strict_best_config=True):
    set_seed(seed)

    dataset_path = os.path.join(BASE_PATH, target_config["file"])
    base_data = Data()
    base_data.load_data(dataset_path)

    input_cols = base_data.raw_data.columns[:-1]
    output_cols = base_data.raw_data.columns[-1:].tolist()
    base_data.split_data(input_cols, output_cols)
    if NORMALIZE:
        base_data.normalize_data()
    base_data.split_train_val_test(test_size=0.2, val_size=0.2, random_state=seed)

    y_test = base_data.test_Y[output_cols[0]] if isinstance(base_data.test_Y, pd.DataFrame) else base_data.test_Y
    y_test = to_flat_array(y_test)

    predictions = []
    model_metrics = {}
    wrappers_by_model = {}
    member_train_configs = {}
    member_config_sources = {}

    print(f"\n=== Ensemble best-config: {target_config['target_name']} / seed={seed} / method={ensemble_method} ===")
    for model_type in MODEL_TYPES:
        wrapper = ModelWrapper(model_type=model_type)
        wrapper.Y_col = output_cols
        wrapper.train_X = base_data.train_X
        wrapper.train_Y = base_data.train_Y
        wrapper.val_X = base_data.val_X
        wrapper.val_Y = base_data.val_Y
        wrapper.test_X = base_data.test_X
        wrapper.test_Y = base_data.test_Y

        cfg, cfg_dict, cfg_source = load_training_config_for_model(
            base_path=BASE_PATH,
            target_config=target_config,
            model_type=model_type,
            strict=strict_best_config,
        )
        wrapper.train_model(cfg)
        metrics_test, pred_test = wrapper.evaluate_split(wrapper.test_X, wrapper.test_Y)

        model_metrics[model_type] = metrics_test
        wrappers_by_model[model_type] = wrapper
        member_train_configs[model_type] = cfg_dict
        member_config_sources[model_type] = cfg_source
        predictions.append(pred_test)
        print(
            f"[{model_type}] source={cfg_source} | "
            f"test MAE={metrics_test['MAE']:.6f}, RMSE={metrics_test['RMSE']:.6f}, R2={metrics_test['R2']:.6f}"
        )

    ensemble_pred = robust_ensemble_predict(predictions, method=ensemble_method)
    ensemble_metrics = {
        "MAE": float(mean_absolute_error(y_test, ensemble_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_test, ensemble_pred)),
        "MSE": float(mean_squared_error(y_test, ensemble_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, ensemble_pred))),
        "R2": float(r2_score(y_test, ensemble_pred)),
    }

    print(
        f"[ensemble_best:{ensemble_method}] test MAE={ensemble_metrics['MAE']:.6f}, "
        f"RMSE={ensemble_metrics['RMSE']:.6f}, R2={ensemble_metrics['R2']:.6f}"
    )

    bundle_info = save_ensemble_bundle(
        base_path=BASE_PATH,
        target_config=target_config,
        seed=seed,
        ensemble_method=ensemble_method,
        wrappers_by_model=wrappers_by_model,
        input_cols=input_cols,
        output_cols=output_cols,
        model_metrics=model_metrics,
        ensemble_metrics=ensemble_metrics,
        member_train_configs=member_train_configs,
        member_config_sources=member_config_sources,
        run_tag="best",
    )
    print(f"Saved best-config ensemble bundle to {bundle_info['bundle_dir']}")

    wandb.init(
        project=target_config["wandb_project"],
        entity="schwalbe-university-of-seoul",
        name=f"{target_config['target_name']}_ensemble_best_{ensemble_method}_{seed}",
        config={
            "mode": "ensemble_best",
            "target_name": target_config["target_name"],
            "file": target_config["file"],
            "ensemble_method": ensemble_method,
            "seed": seed,
            "model_types": MODEL_TYPES,
            "strict_best_config": strict_best_config,
        },
    )
    wandb.log(
        {
            "ensemble_test_metrics": to_jsonable(ensemble_metrics),
            "model_test_metrics": to_jsonable(model_metrics),
            "member_train_configs": to_jsonable(member_train_configs),
            "member_config_sources": to_jsonable(member_config_sources),
            "ensemble_bundle_path": bundle_info["bundle_dir"],
        }
    )

    ensemble_artifact = wandb.Artifact(
        name=f"{target_config['target_name']}_ensemble_best_{ensemble_method}",
        type="ensemble_model",
        description=f"Best-config robust ensemble bundle ({ensemble_method})",
        metadata={
            "target_name": target_config["target_name"],
            "ensemble_method": ensemble_method,
            "seed": seed,
            "model_types": MODEL_TYPES,
            "strict_best_config": strict_best_config,
        },
    )
    ensemble_artifact.add_dir(bundle_info["bundle_dir"], name="ensemble_bundle")
    wandb.log_artifact(ensemble_artifact)
    wandb.finish()


def main():
    if RUN_MODE == "sweep":
        for _ in range(OUTER_REPEAT):
            seed = random.randint(1, 10000)
            for model_config in MODEL_CONFIGS:
                run_pipeline(seed, model_config)
                run_sweep(model_config["model_type"])
        return

    if RUN_MODE == "ensemble_baseline":
        seed = random.randint(1, 10000)
        for target_cfg in BASE_TARGET_CONFIGS:
            run_ensemble_baseline(seed, target_cfg, ensemble_method=ENSEMBLE_METHOD)
        return

    if RUN_MODE == "ensemble_best":
        seed = random.randint(1, 10000)
        for target_cfg in BASE_TARGET_CONFIGS:
            run_ensemble_best(
                seed,
                target_cfg,
                ensemble_method=ENSEMBLE_METHOD,
                strict_best_config=STRICT_BEST_CONFIG,
            )
        return

    raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE}")


if __name__ == "__main__":
    main()
