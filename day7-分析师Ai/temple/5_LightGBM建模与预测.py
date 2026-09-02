#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手车价格预测 - LightGBM 建模与预测
包含：数据预处理、特征工程、模型训练、MAE评估、预测生成提交文件
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============ 中文字体配置 ============
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STHeiti"]
matplotlib.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

# ============ 路径配置 ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(SCRIPT_DIR, "used_car_train_20200313.csv")
TEST_FILE = os.path.join(SCRIPT_DIR, "used_car_testB_20200421.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "model_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("【二手车价格预测 - LightGBM 建模与预测】")
print("=" * 70)


# ============================================================
# 第一部分：数据预处理与特征工程（训练和预测共用）
# ============================================================
def preprocess_data(df, train_df=None, is_train=True):
    """数据预处理：日期特征、目标编码等"""
    df = df.copy()

    # --- 处理 notRepairedDamage ---
    df["notRepairedDamage"] = df["notRepairedDamage"].replace({"-": -1, "0.0": 0, "1.0": 1}).astype(float)

    # --- 日期特征工程 ---
    def process_date_feature(df, col, prefix):
        df[col] = df[col].astype(int).astype(str).str.zfill(8)
        df[f"{prefix}_year"] = df[col].str[:4].astype(int)
        df[f"{prefix}_month"] = df[col].str[4:6].astype(int)
        df[f"{prefix}_day"] = df[col].str[6:8].astype(int)
        df[f"{prefix}_date"] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
        return df

    df = process_date_feature(df, "regDate", "reg")
    df = process_date_feature(df, "creatDate", "create")

    # --- 日期衍生特征 ---
    df["car_age"] = (df["create_date"] - df["reg_date"]).dt.days / 365.25
    df["car_age"] = df["car_age"].clip(0, 100)

    # 以训练集的最大 creatDate 作为参考日期
    if train_df is not None and not is_train:
        reference_date = pd.to_datetime(
            train_df["creatDate"].astype(int).astype(str).str.zfill(8),
            format="%Y%m%d", errors="coerce"
        ).max()
    else:
        reference_date = df["create_date"].max()

    df["car_age_now"] = (reference_date - df["reg_date"]).dt.days / 365.25
    df["car_age_now"] = df["car_age_now"].clip(0, 100)
    df["listing_days"] = (reference_date - df["create_date"]).dt.days.clip(0)
    df["is_new_car"] = (df["car_age"] <= 1).astype(int)

    # 删除原始日期列
    df = df.drop(columns=["regDate", "creatDate", "reg_date", "create_date"], errors="ignore")

    # --- 缺失值处理 ---
    if is_train:
        # 训练集：直接用中位数填充
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
    else:
        # 测试集：用训练集中位数填充
        train_numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        for col in train_numeric_cols:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(train_df[col].median())

    # --- 目标编码 ---
    cat_cols = ["brand", "bodyType", "fuelType", "gearbox", "model", "regionCode"]

    if is_train:
        # 训练集：计算目标编码并保存映射
        encodings = {}
        global_mean = df["price"].mean()
        for col in cat_cols:
            encodings[col] = df.groupby(col)["price"].mean()
            df[col] = df[col].map(encodings[col]).fillna(global_mean)
        # 保存编码映射供预测使用
        df.attrs["encodings"] = encodings
        df.attrs["global_mean"] = global_mean
    else:
        # 测试集：使用训练集的编码映射
        encodings = train_df.attrs.get("encodings", {})
        global_mean = train_df.attrs.get("global_mean", 0)
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].map(encodings.get(col, {})).fillna(global_mean)

    return df


# ============================================================
# 第二部分：模型训练
# ============================================================
print("\n【1. 加载数据】")
train_df_raw = pd.read_csv(TRAIN_FILE, sep=" ")
test_df_raw = pd.read_csv(TEST_FILE, sep=" ")
print(f"  训练集: {train_df_raw.shape[0]} 行 × {train_df_raw.shape[1]} 列")
print(f"  测试集: {test_df_raw.shape[0]} 行")

print("\n【2. 数据预处理与特征工程】")
train_df = preprocess_data(train_df_raw, is_train=True)
print(f"  预处理完成: {train_df.shape}")

print("\n【3. 准备特征矩阵】")
exclude_cols = ["SaleID", "name", "price", "notRepairedDamage"]
feature_cols = [col for col in train_df.columns if col not in exclude_cols]
X = train_df[feature_cols].copy()
y = train_df["price"].copy()
print(f"  特征数量: {X.shape[1]}")
print(f"  特征列表: {feature_cols}")

print("\n【4. 数据划分】")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  训练集: {X_train.shape[0]} 样本")
print(f"  验证集: {X_test.shape[0]} 样本")

print("\n【5. 训练 LightGBM 模型】")
from lightgbm import LGBMRegressor

params = {
    "objective": "mae",           # 直接优化 MAE
    "metric": "mae",              # 验证指标
    "n_estimators": 2000,          # 树的数量
    "learning_rate": 0.02,        # 学习率（小学习率 + 多树 = 更好的泛化）
    "num_leaves": 63,             # 叶子节点数
    "max_depth": -1,              # 不限制深度
    "min_child_samples": 20,      # 叶子最小样本数
    "subsample": 0.8,             # 行采样比例
    "colsample_bytree": 0.8,      # 列采样比例
    "reg_alpha": 0.1,             # L1 正则
    "reg_lambda": 1.0,            # L2 正则
    "random_state": 42,
    "verbosity": -1,              # 静默
    "n_jobs": -1,                 # 使用所有 CPU
    "early_stopping_round": 100,   # 早停轮数
}

print(f"  模型参数:")
for k, v in params.items():
    print(f"    {k}: {v}")
print("  开始训练...")

model = LGBMRegressor(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="mae",
    callbacks=[],
)

# 获取最佳迭代轮数
best_iteration = model.best_iteration_
print(f"  训练完成！最佳迭代轮数: {best_iteration}")

print("\n【6. 模型评估】")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


def evaluate_model(y_true, y_pred, dataset_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100

    print(f"\n  {dataset_name}:")
    print(f"    MAE  (平均绝对误差): {mae:,.2f}")
    print(f"    RMSE (均方根误差):   {rmse:,.2f}")
    print(f"    R²   (决定系数):     {r2:.4f}")
    print(f"    MAPE (平均百分比误差): {mape:.2f}%")
    return mae, rmse, r2, mape


train_metrics = evaluate_model(y_train, y_train_pred, "训练集")
test_metrics = evaluate_model(y_test, y_test_pred, "验证集")


# ============================================================
# 第三部分：MAE 详细分析
# ============================================================
print("\n【7. MAE 详细分析（按价格区间）】")

price_bins = [0, 1000, 3000, 5000, 10000, 20000, 50000, float("inf")]
price_labels = ["0-1k", "1k-3k", "3k-5k", "5k-10k", "10k-20k", "20k-50k", "50k+"]

print(f"\n  验证集按价格区间的 MAE:")
print(f"  {'价格区间':<10s} {'样本数':>8s} {'MAE':>10s} {'RMSE':>10s} {'MAPE':>10s}")
print("  " + "-" * 50)

for i in range(len(price_bins) - 1):
    mask = (y_test >= price_bins[i]) & (y_test < price_bins[i + 1])
    if mask.sum() > 0:
        bin_mae = mean_absolute_error(y_test[mask], y_test_pred[mask])
        bin_rmse = np.sqrt(mean_squared_error(y_test[mask], y_test_pred[mask]))
        bin_mape = np.mean(np.abs((y_test[mask] - y_test_pred[mask]) / (y_test[mask] + 1))) * 100
        print(f"  {price_labels[i]:<10s} {mask.sum():>8d} {bin_mae:>10.2f} {bin_rmse:>10.2f} {bin_mape:>9.2f}%")

absolute_errors = np.abs(y_test - y_test_pred)
print(f"\n  MAE 分布统计:")
print(f"    - 最小绝对误差: {absolute_errors.min():.2f}")
print(f"    - 最大绝对误差: {absolute_errors.max():.2f}")
print(f"    - 平均绝对误差 (MAE): {absolute_errors.mean():.2f}")
print(f"    - 中位数绝对误差: {np.median(absolute_errors):.2f}")
print(f"    - 90%分位数绝对误差: {np.percentile(absolute_errors, 90):.2f}")
print(f"    - 95%分位数绝对误差: {np.percentile(absolute_errors, 95):.2f}")
print(f"    - 99%分位数绝对误差: {np.percentile(absolute_errors, 99):.2f}")

error_ranges = [0, 100, 500, 1000, 2000, 5000, float("inf")]
error_labels = ["<100", "100-500", "500-1k", "1k-2k", "2k-5k", ">5k"]
print(f"\n  误差区间占比:")
for i in range(len(error_ranges) - 1):
    pct = ((absolute_errors >= error_ranges[i]) & (absolute_errors < error_ranges[i + 1])).sum() / len(absolute_errors) * 100
    print(f"    误差 {error_labels[i]:<10s}: {pct:.2f}%")


# ============================================================
# 第四部分：可视化
# ============================================================
print("\n【8. 模型可视化】")

# 8.1 特征重要性
fig, ax = plt.subplots(figsize=(12, 8))
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]
top_n = min(20, len(feature_cols))
top_features = [feature_cols[i] for i in indices[:top_n]]
top_importance = importance[indices[:top_n]]

ax.barh(range(top_n), top_importance, color="#00b4d8", alpha=0.8)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features)
ax.set_xlabel("特征重要性 (Gain)")
ax.set_title(f"LightGBM Top {top_n} 特征重要性")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_lgbm_feature_importance.png"), dpi=150)
plt.close()
print("  01_lgbm_feature_importance.png 已保存")

# 8.2 预测值 vs 实际值
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].scatter(y_train, y_train_pred, alpha=0.3, s=5, color="#00b4d8")
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--", lw=2, label="理想预测线")
axes[0].set_xlabel("实际价格")
axes[0].set_ylabel("预测价格")
axes[0].set_title(f"训练集: R²={train_metrics[2]:.4f}")
axes[0].legend()
axes[1].scatter(y_test, y_test_pred, alpha=0.3, s=5, color="#6bcf7f")
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="理想预测线")
axes[1].set_xlabel("实际价格")
axes[1].set_ylabel("预测价格")
axes[1].set_title(f"验证集: R²={test_metrics[2]:.4f}")
axes[1].legend()
plt.suptitle("LightGBM 预测值 vs 实际值")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_lgbm_prediction_vs_actual.png"), dpi=150)
plt.close()
print("  02_lgbm_prediction_vs_actual.png 已保存")

# 8.3 残差分析
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
residuals = y_test - y_test_pred
axes[0].hist(residuals, bins=50, color="#ff6b6b", alpha=0.7, edgecolor="white")
axes[0].axvline(0, color="black", linestyle="--", label="0基准线")
axes[0].set_xlabel("残差 (实际 - 预测)")
axes[0].set_ylabel("频数")
axes[0].set_title("验证集残差分布")
axes[0].legend()
axes[1].scatter(y_test_pred, residuals, alpha=0.3, s=5, color="#00b4d8")
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_xlabel("预测价格")
axes[1].set_ylabel("残差")
axes[1].set_title("残差 vs 预测值")
plt.suptitle("LightGBM 残差分析")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_lgbm_residual_analysis.png"), dpi=150)
plt.close()
print("  03_lgbm_residual_analysis.png 已保存")

# 8.4 MAE 分析可视化
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].hist(absolute_errors, bins=50, color="#ff6b6b", alpha=0.7, edgecolor="white")
axes[0].axvline(absolute_errors.mean(), color="blue", linestyle="--", label=f"MAE: {absolute_errors.mean():.0f}")
axes[0].axvline(np.median(absolute_errors), color="green", linestyle="--", label=f"中位数: {np.median(absolute_errors):.0f}")
axes[0].set_xlabel("绝对误差 |实际-预测|")
axes[0].set_ylabel("频数")
axes[0].set_title("验证集 MAE 分布")
axes[0].legend()

mae_by_bin = []
for i in range(len(price_bins) - 1):
    mask = (y_test >= price_bins[i]) & (y_test < price_bins[i + 1])
    if mask.sum() > 0:
        mae_by_bin.append({"bin": price_labels[i], "mae": mean_absolute_error(y_test[mask], y_test_pred[mask])})
    else:
        mae_by_bin.append({"bin": price_labels[i], "mae": 0})

bins = [x["bin"] for x in mae_by_bin]
maes = [x["mae"] for x in mae_by_bin]
axes[1].bar(bins, maes, color="#00b4d8", alpha=0.8)
axes[1].set_xlabel("价格区间")
axes[1].set_ylabel("MAE")
axes[1].set_title("按价格区间的 MAE")
axes[1].tick_params(axis="x", rotation=45)
plt.suptitle("LightGBM MAE 详细分析")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_lgbm_mae_analysis.png"), dpi=150)
plt.close()
print("  04_lgbm_mae_analysis.png 已保存")

# 8.5 训练过程
try:
    evals_result = model.evals_result_
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in evals_result:
        if "mae" in evals_result[key]:
            ax.plot(evals_result[key]["mae"], label=f"{key} MAE")
    ax.axvline(best_iteration, color="red", linestyle="--", label=f"最佳迭代: {best_iteration}")
    ax.set_xlabel("迭代轮数")
    ax.set_ylabel("MAE")
    ax.set_title("LightGBM 训练过程 MAE 变化")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_lgbm_training_process.png"), dpi=150)
    plt.close()
    print("  05_lgbm_training_process.png 已保存")
except Exception as e:
    print(f"  跳过训练过程图: {e}")


# ============================================================
# 第五部分：保存模型与预测
# ============================================================
print("\n【9. 保存模型】")
model_path = os.path.join(OUTPUT_DIR, "lightgbm_model.pkl")
import joblib
joblib.dump(model, model_path)
print(f"  模型已保存: {model_path}")

# 保存特征重要性
importance_df = pd.DataFrame({"feature": feature_cols, "importance": importance}).sort_values("importance", ascending=False)
importance_df.to_csv(os.path.join(OUTPUT_DIR, "lgbm_feature_importance.csv"), index=False)
print("  lgbm_feature_importance.csv 已保存")

# 保存编码映射（供预测脚本使用）
encodings_path = os.path.join(OUTPUT_DIR, "lgbm_encodings.pkl")
encodings_data = {
    "encodings": train_df.attrs.get("encodings", {}),
    "global_mean": train_df.attrs.get("global_mean", 0),
    "feature_cols": feature_cols,
}
joblib.dump(encodings_data, encodings_path)
print(f"  编码映射已保存: {encodings_path}")

# ---- 预测 ----
print("\n【10. 测试集预测】")
print("  加载测试集并预处理...")

# 加载已保存的编码映射
encodings_data = joblib.load(encodings_path)
encodings = encodings_data["encodings"]
global_mean = encodings_data["global_mean"]

# 手动预处理测试集（参考训练集的统计量）
test_df = test_df_raw.copy()

# 处理 notRepairedDamage
test_df["notRepairedDamage"] = test_df["notRepairedDamage"].replace({"-": -1, "0.0": 0, "1.0": 1}).astype(float)

# 日期特征
test_df["regDate"] = test_df["regDate"].astype(int).astype(str).str.zfill(8)
test_df["creatDate"] = test_df["creatDate"].astype(int).astype(str).str.zfill(8)

test_df["reg_year"] = test_df["regDate"].str[:4].astype(int)
test_df["reg_month"] = test_df["regDate"].str[4:6].astype(int)
test_df["reg_day"] = test_df["regDate"].str[6:8].astype(int)
test_df["reg_date"] = pd.to_datetime(test_df["regDate"], format="%Y%m%d", errors="coerce")

test_df["create_year"] = test_df["creatDate"].str[:4].astype(int)
test_df["create_month"] = test_df["creatDate"].str[4:6].astype(int)
test_df["create_day"] = test_df["creatDate"].str[6:8].astype(int)
test_df["create_date"] = pd.to_datetime(test_df["creatDate"], format="%Y%m%d", errors="coerce")

# 使用原始训练集的最大 creatDate 作为参考日期
reference_date = pd.to_datetime(
    train_df_raw["creatDate"].astype(int).astype(str).str.zfill(8),
    format="%Y%m%d", errors="coerce"
).max()

# 衍生特征
test_df["car_age"] = (test_df["create_date"] - test_df["reg_date"]).dt.days / 365.25
test_df["car_age"] = test_df["car_age"].clip(0, 100)
test_df["car_age_now"] = (reference_date - test_df["reg_date"]).dt.days / 365.25
test_df["car_age_now"] = test_df["car_age_now"].clip(0, 100)
test_df["listing_days"] = (reference_date - test_df["create_date"]).dt.days.clip(0)
test_df["is_new_car"] = (test_df["car_age"] <= 1).astype(int)

# 删除原始日期列
test_df = test_df.drop(columns=["regDate", "creatDate", "reg_date", "create_date"], errors="ignore")

# 缺失值处理（用训练集中位数）
train_numeric_cols = train_df.select_dtypes(include=[np.number]).columns
for col in train_numeric_cols:
    if col in test_df.columns and test_df[col].isnull().any():
        test_df[col] = test_df[col].fillna(train_df[col].median())

# 目标编码（使用训练集保存的映射）
cat_cols = ["brand", "bodyType", "fuelType", "gearbox", "model", "regionCode"]
for col in cat_cols:
    if col in test_df.columns:
        test_df[col] = test_df[col].map(encodings.get(col, {})).fillna(global_mean)

print(f"  测试集预处理完成: {test_df.shape}")

# 准备特征
exclude_cols_test = ["SaleID", "name", "price", "notRepairedDamage"]
feature_cols_test = [col for col in test_df.columns if col not in exclude_cols_test]

# 对齐特征列
for col in feature_cols:
    if col not in test_df.columns:
        test_df[col] = 0

X_test_final = test_df[feature_cols].copy()
print(f"  测试特征矩阵: {X_test_final.shape[0]} × {X_test_final.shape[1]}")

# 预测
print("  开始预测...")
predictions = model.predict(X_test_final)
predictions = np.maximum(predictions, 0)  # 确保价格非负
print(f"  预测完成，共 {len(predictions)} 条")

# 生成提交文件
submit = pd.DataFrame({
    "SaleID": test_df_raw["SaleID"],
    "price": predictions,
})
output_file = os.path.join(SCRIPT_DIR, "used_car_submit_lgbm.csv")
submit.to_csv(output_file, index=False)

print(f"\n【11. 提交文件统计】")
print(f"  提交文件: {output_file}")
print(f"  行数: {len(submit)}")
print(f"  SaleID 范围: {submit['SaleID'].min()} ~ {submit['SaleID'].max()}")
print(f"  价格统计: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}, median={np.median(predictions):.2f}")

# 预览
print(f"\n  预览前10行:")
print(submit.head(10).to_string(index=False))


# ============================================================
# 第六部分：总结报告
# ============================================================
print("\n" + "=" * 70)
print("【LightGBM 建模总结报告】")
print("=" * 70)
print(f"""
数据概况:
  - 训练集: {len(train_df):,} 样本
  - 验证集: {len(y_test):,} 样本
  - 特征数量: {len(feature_cols)}

模型参数:
  - objective: mae
  - n_estimators: {params['n_estimators']}
  - learning_rate: {params['learning_rate']}
  - num_leaves: {params['num_leaves']}
  - best_iteration: {best_iteration}

模型性能:
  训练集 MAE:  {train_metrics[0]:,.2f}
  验证集 MAE:  {test_metrics[0]:,.2f}
  训练集 R²:   {train_metrics[2]:.4f}
  验证集 R²:   {test_metrics[2]:.4f}
  训练集 RMSE: {train_metrics[1]:,.2f}
  验证集 RMSE: {test_metrics[1]:,.2f}

Top 5 重要特征:
""")

for i, (feat, imp) in enumerate(zip(top_features[:5], top_importance[:5]), 1):
    print(f"  {i}. {feat}: {imp:.4f}")

print(f"""
输出文件:
  模型文件: lightgbm_model.pkl
  编码映射: lgbm_encodings.pkl
  特征重要性: lgbm_feature_importance.csv
  提交文件: used_car_submit_lgbm.csv
  可视化图表: 01~05_lgbm_*.png
""")

print("=" * 70)
print("【LightGBM 建模完成】")
print("=" * 70)
