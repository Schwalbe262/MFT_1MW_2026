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
import shutil  # 파일 복사용
import uuid
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
import platform
from filelock import FileLock  # 파일 동시 접근 제어


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ================================================
# Data 클래스
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
        """입력된 컬럼 리스트에 대해 이상치를 탐지하고, 해당 행들을 raw_data에서 제거합니다."""
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
        # 출력 데이터를 스케일링 처리 (예: 1/1000로 변경하고 싶으면 SCALING 값 조정)
        self.Y = self.raw_data[output_cols]

    def normalize_data(self):
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        # scaler 저장
        model_dir = os.path.join(PATH, "saved_models", ARTIFACT_NM)
        os.makedirs(model_dir, exist_ok=True)
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {scaler_path}")
    
    def split_train_val_test(self, test_size=0.2, val_size=0.2, random_state=42):
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(self.X, self.Y, test_size=test_size, random_state=random_state)
        relative_val_size = val_size / (1 - test_size)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=relative_val_size, random_state=random_state)
        self.train_X = X_train
        self.train_Y = Y_train
        self.val_X = X_val
        self.val_Y = Y_val
        self.test_X = X_test
        self.test_Y = Y_test
        print(f"Data split into: Train {self.train_X.shape}, Val {self.val_X.shape}, Test {self.test_X.shape}")

# ================================================
# LightGBMWrapper 클래스 (Data 상속)
# ================================================
class LGBMWrapper(Data):
    def __init__(self):
        super().__init__()
        self.model = None
        self.Y_col = None  # 출력 변수 이름 (리스트)

    def train_model(self, config):
        params = {
            'objective': 'regression',
            'metric': 'l2',
            'learning_rate': config.lr,
            'num_leaves': config.num_leaves,
            'max_depth': config.max_depth,
            'verbosity': -1,
            'seed': 42
        }
        train_data = lgb.Dataset(self.train_X, label=np.array(self.train_Y).flatten())
        valid_data = lgb.Dataset(self.val_X, label=np.array(self.val_Y).flatten(), reference=train_data)
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=config.n_estimators,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(config.patience), lgb.log_evaluation(0)]
        )
        return self.model

    def evaluate_split(self, model, X, y):
        y_pred = model.predict(X, num_iteration=model.best_iteration)
        y_true = np.array(y).flatten()
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {"MAE": mae, "MAPE": mape, "MSE": mse, "RMSE": rmse, "R2": r2}, y_pred

    def plot_scatter(self, output_col, X_data, y_data, save_path=None, metrics=None):
        y_pred = self.model.predict(X_data, num_iteration=self.model.best_iteration)
        y_true = np.array(y_data).flatten()
        
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
                "MAPE": mape
            }
        
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k', label="Data points")
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal Fit")
        slope, intercept = np.polyfit(y_true, y_pred, 1)
        reg_line = slope * np.array([min_val, max_val]) + intercept
        plt.plot([min_val, max_val], reg_line, 'b-', lw=2, label="Regression Line")
        
        # 그리드 표시
        plt.grid(True, linestyle='--', linewidth=1)
        
        plt.xlabel("Actual Values", fontsize=24)
        plt.ylabel("Predicted Values", fontsize=24)
        plt.title(f"Scatter Plot for {output_col}", fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        
        # 출력 순서: R², MAE, MSE, RMSE, MAPE
        ordered_keys = ["R2", "MAE", "MSE", "RMSE", "MAPE"]
        lines = []
        for key in ordered_keys:
            value = metrics.get(key, None)
            if value is None:
                line = f"{key}: N/A"
            elif key == "R2":
                # R²로 표시
                line = f"R²: {value:.4f}"
            elif key == "MAPE":
                # MAPE를 퍼센트로 표시 (소수점 2째자리)
                line = f"MAPE: {value * 100:.2f}%"
            else:
                line = f"{key}: {value:.4f}"
            lines.append(line)
        metrics_text = "\n".join(lines)
        
        plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                    fontsize=20, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Scatter plot saved to {save_path}")
        plt.close()

# ================================================
# Best Model Artifact 업데이트 함수
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
    
    # best_val_loss 처리
    key = "best_val_loss"
    current_val = splits.get("best_val_loss")
    improved = False
    if key not in best_values:
        improved = True
    else:
        old_val = best_values[key]
        if old_val is None and current_val is not None:
            improved = True
        elif current_val is None:
            improved = False
        else:
            if current_val < old_val:
                improved = True
    if improved:
        formatted_old = "N/A" if key not in best_values or best_values[key] is None else f"{float(best_values[key]):.4f}"
        formatted_current = "None" if current_val is None else f"{current_val:.4f}"
        improvements_description.append(f"{key} improved from {formatted_old} to {formatted_current}")
        best_values[key] = current_val
        improved_any = True
        key_dir = os.path.join(best_model_dir, key)
        os.makedirs(key_dir, exist_ok=True)
        model_dest = os.path.join(key_dir, "model.txt")
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
    
    # 나머지 메트릭 처리 (예: train_MAE, train_R2, 등)
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
                elif current_value is None:
                    improved = False
                else:
                    if metric_name == "R2":
                        if current_value > best_value:
                            improved = True
                    else:
                        if current_value < best_value:
                            improved = True
            if improved:
                formatted_old = "N/A" if key not in best_values or best_values[key] is None else f"{float(best_values[key]):.4f}"
                formatted_current = "None" if current_value is None else f"{current_value:.4f}"
                improvements_description.append(f"{key} improved from {formatted_old} to {formatted_current}")
                best_values[key] = current_value
                improved_any = True
                key_dir = os.path.join(best_model_dir, key)
                os.makedirs(key_dir, exist_ok=True)
                model_dest = os.path.join(key_dir, "model.txt")
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
# Sweep용 단일 실험 함수 (wandb.init 내부에서 실행)
# ================================================
def sweep_train():
    wandb.init(project=WANDB_PR, entity="schwalbe-university-of-seoul")
    config = wandb.config
    global lgb_model_instance
    lgb_wrapper = lgb_model_instance

    # 데이터 준비
    y_train = lgb_wrapper.train_Y[lgb_wrapper.Y_col[0]] if isinstance(lgb_wrapper.train_Y, pd.DataFrame) else lgb_wrapper.train_Y
    y_val   = lgb_wrapper.val_Y[lgb_wrapper.Y_col[0]] if isinstance(lgb_wrapper.val_Y, pd.DataFrame) else lgb_wrapper.val_Y
    y_test  = lgb_wrapper.test_Y[lgb_wrapper.Y_col[0]] if isinstance(lgb_wrapper.test_Y, pd.DataFrame) else lgb_wrapper.test_Y
    X_train = lgb_wrapper.train_X
    X_val   = lgb_wrapper.val_X
    X_test  = lgb_wrapper.test_X

    model = lgb_wrapper.train_model(config)
    
    metrics_train, _ = lgb_wrapper.evaluate_split(model, X_train, y_train)
    metrics_val, _   = lgb_wrapper.evaluate_split(model, X_val, y_val)
    metrics_test, _  = lgb_wrapper.evaluate_split(model, X_test, y_test)
    
    best_val_loss = model.best_score['valid_0']['l2'] if 'valid_0' in model.best_score else np.inf
    if best_val_loss == float('inf'):
        best_val_loss = None

    wandb.log({
        "final_train_metrics": metrics_train,
        "final_val_metrics": metrics_val,
        "final_test_metrics": metrics_test,
        "best_val_loss": best_val_loss
    })
    
    splits = {"train": metrics_train, "val": metrics_val, "test": metrics_test}
    splits["best_val_loss"] = best_val_loss

    base_best_values_file = os.path.join(PATH, "best_model", ARTIFACT_NM, "best_values.json")
    if os.path.exists(base_best_values_file):
        with open(base_best_values_file, "r") as f:
            base_best_values = json.load(f)
    else:
        base_best_values = {}

    improved_flag = False
    key = "best_val_loss"
    current_val = splits.get("best_val_loss")
    
    # best_val_loss 개선 여부 확인
    if key not in base_best_values or (current_val is not None and current_val < base_best_values[key]):
        improved_flag = True
    
    # 다른 지표 개선 여부 확인
    for split, metrics in splits.items():
        if split == "best_val_loss":
            continue
        for metric_name, current_value in metrics.items():
            key = f"{split}_{metric_name}"
            if key not in base_best_values or (metric_name == "R2" and current_value > base_best_values[key]) or \
                (metric_name != "R2" and current_value < base_best_values[key]):
                improved_flag = True

    scatter_files = None
    if improved_flag:
        # best_model 폴더에 저장
        best_model_dir = os.path.join(PATH, "best_model", ARTIFACT_NM)
        os.makedirs(best_model_dir, exist_ok=True)

        model_file = os.path.join(best_model_dir, "best_model.txt")
        config_file = os.path.join(best_model_dir, "best_config.json")

        model.save_model(model_file)
        with open(config_file, "w") as f:
            json.dump(dict(config), f)

        # Scatter Plot 저장
        scatter_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        train_scatter = os.path.join(best_model_dir, f"{lgb_wrapper.Y_col[0]}_train_scatter_{scatter_timestamp}.png")
        val_scatter   = os.path.join(best_model_dir, f"{lgb_wrapper.Y_col[0]}_val_scatter_{scatter_timestamp}.png")
        test_scatter  = os.path.join(best_model_dir, f"{lgb_wrapper.Y_col[0]}_test_scatter_{scatter_timestamp}.png")

        lgb_wrapper.plot_scatter(lgb_wrapper.Y_col[0], X_train, y_train, save_path=train_scatter, metrics=metrics_train)
        lgb_wrapper.plot_scatter(lgb_wrapper.Y_col[0], X_val, y_val, save_path=val_scatter, metrics=metrics_val)
        lgb_wrapper.plot_scatter(lgb_wrapper.Y_col[0], X_test, y_test, save_path=test_scatter, metrics=metrics_test)
        scatter_files = [train_scatter, val_scatter, test_scatter]

        update_best_model_artifact(splits, model_file, config_file, scatter_files=scatter_files)

        # 모델 업데이트
        lgb_wrapper.model = model

    wandb.finish()


# ================================================
# Sweep ID 저장 및 불러오기 함수 (파일 잠금 적용)
# ================================================
def get_or_create_sweep_id(sweep_config, project):
    # 파일 잠금 적용: sweep_id.json에 대한 동시 접근 방지
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
        
        key = f"sweep_id_{SWEEP_NM}"
        if key in sweep_data:
            sweep_id = sweep_data[key]
            try:
                _ = wandb.Api().sweep(f"schwalbe-university-of-seoul/{project}/{sweep_id}")
                print(f"Loaded existing sweep_id: {sweep_id}")
                return sweep_id
            except Exception as e:
                print(f"기존 sweep_id {sweep_id}를 찾을 수 없습니다. 새로운 sweep을 생성합니다. 에러: {e}")
                sweep_data.pop(key, None)
        
        sweep_id = wandb.sweep(sweep_config, project=project, entity="schwalbe-university-of-seoul")
        sweep_data[key] = sweep_id
        with open(SWEEP_ID_PATH, "w") as f:
            json.dump(sweep_data, f)
        print(f"Created new sweep_id: {sweep_id}")
        return sweep_id

# ================================================
# 메인 실행부
# ================================================
NUM_WORKER = 1
NORMALIZE = False
BASE_PATH = "/gpfs/home1/r1jae262/jupyter/MFT_1MW/MFT_1MW_2026/regression_260515"



# 모델 설정 리스트
MODEL_CONFIGS = [
    {
        "name": "Lmt_LightGBM_260515",
        "file": "data_Lmt.csv", 
        "wandb_project": "MFT_1MW_260515"
    },
    {
        "name": "Llt_LightGBM_260515",
        "file": "data_Llt.csv",
        "wandb_project": "MFT_1MW_260515"
    },
    {
        "name": "Tx_loss_LightGBM_260515",
        "file": "data_Tx_loss.csv",
        "wandb_project": "MFT_1MW_260515"
    },
    {
        "name": "Rx_loss_LightGBM_260515", 
        "file": "data_Rx_loss.csv",
        "wandb_project": "MFT_1MW_260515"
    },
    {
        "name": "P_main_winding_inner_LightGBM_260515", 
        "file": "data_P_Tx_main_winding_inner.csv",
        "wandb_project": "MFT_1MW_260515"
    },
    {
        "name": "P_main_winding_outer_LightGBM_260515", 
        "file": "data_P_Tx_main_winding_outer.csv",
        "wandb_project": "MFT_1MW_260515"
    },
    {
        "name": "P_side_winding_inner_LightGBM_260515", 
        "file": "data_P_Tx_side_winding_inner.csv",
        "wandb_project": "MFT_1MW_260515"
    },
    {
        "name": "P_side_winding_outer_LightGBM_260515", 
        "file": "data_P_Tx_side_winding_outer.csv",    
        "wandb_project": "MFT_1MW_260515"
    },
    {
        "name": "time_LightGBM_260515", 
        "file": "data_time.csv",    
        "wandb_project": "MFT_1MW_260515"
    },
]



def run_pipeline(seed, model_config):
    global PATH, SWEEP_NM, FILE_NAME, WANDB_PR, SWEEP_ID_PATH, ARTIFACT_NM
    
    PATH = BASE_PATH
    SWEEP_NM = model_config["name"]
    FILE_NAME = model_config["file"]
    WANDB_PR = model_config["wandb_project"]
    SWEEP_ID_PATH = f"{PATH}/wandb_id/sweep_id.json"
    ARTIFACT_NM = f"{WANDB_PR}_{SWEEP_NM}"

    print(f"\n=== Running with seed {seed} for model {SWEEP_NM} ===")
    set_seed(seed)

    lgb_wrapper = LGBMWrapper()
    data_path = f"{PATH}/{FILE_NAME}"
    lgb_wrapper.load_data(data_path)
    # 입력 변수: 컬럼 1~23, 출력 변수: 컬럼 31 (예시)
    input_cols = lgb_wrapper.raw_data.columns[:-1]  # 마지막 컬럼을 제외한 모든 컬럼을 입력 변수로
    output_cols = lgb_wrapper.raw_data.columns[-1:].tolist()  # 마지막 컬럼을 출력 변수로
    # lgb_wrapper.remove_outliers(columns=list(input_cols) + output_cols, weight=1.5)
    lgb_wrapper.split_data(input_cols, output_cols)
    lgb_wrapper.Y_col = output_cols  # 출력 컬럼 이름 저장
    if NORMALIZE:
        lgb_wrapper.normalize_data()
    lgb_wrapper.split_train_val_test(test_size=0.2, val_size=0.2, random_state=seed)
    global lgb_model_instance
    lgb_model_instance = lgb_wrapper

def run_sweep():
    sweep_config = {
        'name': SWEEP_NM,
        'method': 'bayes',  # Bayesian Optimization
        'metric': {'name': 'best_val_loss', 'goal': 'minimize'},
        'parameters': {
            # Learning rate (중복 제거 및 수정)
            'lr': {
                'values': [
                    1e-6, 2e-6, 3e-6, 5e-6,  # 초소형
                    1e-5, 2e-5, 3e-5, 5e-5,  # 소형
                    1e-4, 2e-4, 3e-4, 5e-4,  # 중형
                    1e-3, 2e-3, 3e-3, 5e-3,  # 대형
                    1e-2, 2e-2, 3e-2, 5e-2   # 초대형
                ]
            },
            # 트리 구조 관련 파라미터
            'num_leaves': {'values': [7, 15, 31, 63, 127, 255]},
            'max_depth': {'values': [-1, 3, 5, 7, 10, 15, 20]},  # 30, 40 제거

            # 부스팅 관련
            'n_estimators': {'values': [50, 100, 150, 200, 300, 500, 1000, 2000]},  # 10000 제거
            'patience': {'values': [10, 15, 20, 25, 30, 40]},

            # 과적합 방지 관련
            'min_child_samples': {'values': [5, 10, 20, 30, 50]},
            'subsample': {'values': [0.6, 0.7, 0.8, 0.9, 1.0]},  # 0.65, 0.75, 0.85, 0.95 제거
            'colsample_bytree': {'values': [0.6, 0.7, 0.8, 0.9, 1.0]},

            # 정규화 (L1, L2 페널티)
            'reg_alpha': {'values': [0, 1e-3, 1e-2, 1e-1, 1]},
            'reg_lambda': {'values': [0, 1e-3, 1e-2, 1e-1, 1]}
        }
    }

    
    sweep_id = get_or_create_sweep_id(sweep_config, project=WANDB_PR)
    print(f"Using sweep id: {sweep_id}")
    wandb.agent(sweep_id, function=sweep_train, count=10, project=WANDB_PR, entity="schwalbe-university-of-seoul")

def main():
    # 한 번만 sweep id를 생성하고 재사용하도록 함
    run_pipeline(seed=42)
    run_sweep()

if __name__ == '__main__':
    seed = random.randint(1, 10000)
    for i in range(100):
        for model_config in MODEL_CONFIGS:
            run_pipeline(seed, model_config)
            run_sweep()