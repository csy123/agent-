#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手车价格预测 - CatBoost 建模与预测
CatBoost 原生支持类别特征，无需目标编码，自带有序目标编码防止数据泄露
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
print("【二手车价格预测 - CatBoost 建模与预测】")
print("=" * 70)


# ============================================================
# 第一部分：数据预处理（训练和预测共用逻辑）
# ============================================================
def build_preprocessed(df, reference_date=None, is_train=True):
    """
    数据预处理：日期特征工程 + 缺失值处理
    返回：(处理后的DataFrame, 训练集中位数dict)
    """
    df = df.copy()

    # --- 处理 notRepairedDamage ---
    df["notRepairedDamage"] = df["notRepairedDamage"].replace({"-": -1, "0.0": 0, "1.0": 1}).astype(float)

    # --- 日期特征工程 ---
    # regDate
    df["regDate_str"] = df["regDate"].astype(int).astype(str).str.zfill(8)
    df["reg_year"] = df["regDate_str"].str[:4].astype(int)
    df["reg_month"] = df["regDate_str"].str[4:6].astype(int)
    df["reg_day"] = df["regDate_str"].str[6:8].astype(int)
    df["reg_date"] = pd.to_datetime(df["regDate_str"], format="%Y%m%d", errors="coerce")

    # creatDate
    df["creatDate_str"] = df["creatDate"].astype(int).astype(str).str.zfill(8)
    df["create_year"] = df["creatDate_str"].str[:4].astype(int)
    df["create_month"] = df["creatDate_str"].str[4:6].astype(int)
    df["create_day"] = df["creatDate_str"].str[6:8].astype(int)
    df["create_date"] = pd.to_datetime(df["creatDate_str"], format="%Y%m%d", errors="coerce")

    # --- 日期衍生特征 ---
    df["car_age"] = (df["create_date"] - df["reg_date"]).dt.days / 365.25
    df["car_age"] = df["car_age"].clip(0, 100)

    # 参考日期
    if reference_date is None:
        reference_date = df["create_date"].max()

    df["car_age_now"] = (reference_date - df["reg_date"]).dt.days / 365.25
    df["car_age_now"] = df["car_age_now"].clip(0, 100)
    df["listing_days"] = (reference_date - df["create_date"]).dt.days.clip(0)
    df["is_new_car"] = (df["car_age"] <= 1).astype(int)

    # 删除中间列
    df = df.drop(columns=["regDate", "creatDate", "regDate_str", "creatDate_str",
                          "reg_date", "create_date"], errors="ignore")

    # --- 缺失值处理 ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if is_train:
        # 训练集：保存中位数用于测试集
        medians = {}
        for col in numeric_cols:
            if df[col].isnull().any():
                med = df[col].median()
                df[col] = df[col].fillna(med)
                medians[col] = med
    else:
        # 测试集：使用传入的中位数
        medians = {}
        for col in numeric_cols:
            if df[col].isnull().any():
                # 跳过 price（测试集没有）
                if col == "price":
                    continue
                df[col] = df[col].fillna(reference_date[col] if isinstance(reference_date, dict) and col in reference_date else 0)

    return df, medians


# ============================================================
# 第二部分：加载和预处理数据
# ============================================================
print("\n【1. 加载数据】")
train_df_raw = pd.read_csv(TRAIN_FILE, sep=" ")
test_df_raw = pd.read_csv(TEST_FILE, sep=" ")
print(f"  训练集: {train_df_raw.shape[0]} 行 × {train_df_raw.shape[1]} 列")
print(f"  测试集: {test_df_raw.shape[0]} 行")

print("\n【2. 数据预处理】")

# 先用原始训练集计算参考日期
reference_date = pd.to_datetime(
    train_df_raw["creatDate"].astype(int).astype(str).str.zfill(8),
    format="%Y%m%d", errors="coerce"
).max()
print(f"  参考日期（最大 creatDate）: {reference_date.strftime('%Y-%m-%d')}")

# 训练集预处理
train_df, medians = build_preprocessed(train_df_raw, reference_date=reference_date, is_train=True)
print(f"  训练集预处理完成: {train_df.shape}")

# CatBoost 需要将类别特征转为字符串类型（内部自动处理）
cat_cols = ["brand", "bodyType", "fuelType", "gearbox", "model", "regionCode"]
print(f"\n  CatBoost 类别特征: {cat_cols}")
print("  CatBoost 原生支持类别特征，无需目标编码（使用有序目标编码）")

# 将类别特征转为字符串（CatBoost 要求）
for col in cat_cols:
    train_df[col] = train_df[col].astype(str)

print("\n【3. 准备特征矩阵】")
exclude_cols = ["SaleID", "name", "price", "notRepairedDamage"]
feature_cols = [col for col in train_df.columns if col not in exclude_cols]

X = train_df[feature_cols].copy()
y = train_df["price"].copy()

# 确定哪些是类别特征（在 feature_cols 中的部分）
cat_features_in_data = [col for col in cat_cols if col in feature_cols]
print(f"  特征数量: {X.shape[1]}")
print(f"  特征列表: {feature_cols}")
print(f"  类别特征（CatBoost 自动处理）: {cat_features_in_data}")

print("\n【4. 数据划分】")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  训练集: {X_train.shape[0]} 样本")
print(f"  验证集: {X_test.shape[0]} 样本")

# ============================================================
# 第三部分：训练 CatBoost 模型
# ============================================================
print("\n【5. 训练 CatBoost 模型】")
from catboost import CatBoostRegressor, Pool

params = {
    "loss_function": "MAE",         # 直接优化 MAE
    "eval_metric": "MAE",           # 验证指标
    "iterations": 3000,             # 迭代轮数（树的数量）
    "learning_rate": 0.05,          # 学习率
    "depth": 8,                     # 树深度
    "l2_leaf_reg": 3,               # L2 正则
    "random_seed": 42,
    "verbose": 500,                 # 每500轮打印一次
    "early_stopping_rounds": 200,   # 早停
    "use_best_model": True,         # 使用最佳模型
    "task_type": "CPU",             # 使用 CPU
    "allow_writing_files": False,   # 不写临时文件
}

print(f"  模型参数:")
for k, v in params.items():
    if k != "verbose":
        print(f"    {k}: {v}")
print("  开始训练...")

# 创建 Pool 对象（CatBoost 推荐方式，可指定类别特征）
train_pool = Pool(X_train, y_train, cat_features=cat_features_in_data)
eval_pool = Pool(X_test, y_test, cat_features=cat_features_in_data)

model = CatBoostRegressor(**params)
model.fit(train_pool, eval_set=eval_pool, use_best_model=True)

best_iteration = model.get_best_iteration()
print(f"\n  训练完成！最佳迭代轮数: {best_iteration}")


# ============================================================
# 第四部分：模型评估
# ============================================================
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
# 第五部分：MAE 详细分析
# ============================================================
print("\n【7. MAE 详细分析（按价格区间）】")

price_bins = [0, 1000, 3000, 5000, 10000, 20000, 50000, float("inf")]
price_labels = ["0-1k", "1k-3k", "3k-5k", "5k-10k", "10k-20k", "20k-50k", "50k+"]

print(f"\n  验证集按价格区间的 MAE:")
print(f"  {'价格区间':<10s} {'样本数':>8s} {'MAE':>10s} {'RMSE':>10s} {'MAPE':>10s}")
print("  " + "-" * 50)

mae_by_bins = []
for i in range(len(price_bins) - 1):
    mask = (y_test >= price_bins[i]) & (y_test < price_bins[i + 1])
    if mask.sum() > 0:
        bin_mae = mean_absolute_error(y_test[mask], y_test_pred[mask])
        bin_rmse = np.sqrt(mean_squared_error(y_test[mask], y_test_pred[mask]))
        bin_mape = np.mean(np.abs((y_test[mask] - y_test_pred[mask]) / (y_test[mask] + 1))) * 100
        mae_by_bins.append({"bin": price_labels[i], "mae": bin_mae})
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
# 第六部分：可视化
# ============================================================
print("\n【8. 模型可视化】")

# 8.1 特征重要性
fig, ax = plt.subplots(figsize=(12, 8))
feature_importance = model.get_feature_importance()
importance = np.array(feature_importance)
indices = np.argsort(importance)[::-1]
top_n = min(20, len(feature_cols))
top_features = [feature_cols[i] for i in indices[:top_n]]
top_importance = importance[indices[:top_n]]

ax.barh(range(top_n), top_importance, color="#00b4d8", alpha=0.8)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features)
ax.set_xlabel("特征重要性")
ax.set_title(f"CatBoost Top {top_n} 特征重要性")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_catboost_feature_importance.png"), dpi=150)
plt.close()
print("  01_catboost_feature_importance.png 已保存")

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
plt.suptitle("CatBoost 预测值 vs 实际值")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_catboost_prediction_vs_actual.png"), dpi=150)
plt.close()
print("  02_catboost_prediction_vs_actual.png 已保存")

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
plt.suptitle("CatBoost 残差分析")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_catboost_residual_analysis.png"), dpi=150)
plt.close()
print("  03_catboost_residual_analysis.png 已保存")

# 8.4 MAE 分析可视化
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].hist(absolute_errors, bins=50, color="#ff6b6b", alpha=0.7, edgecolor="white")
axes[0].axvline(absolute_errors.mean(), color="blue", linestyle="--", label=f"MAE: {absolute_errors.mean():.0f}")
axes[0].axvline(np.median(absolute_errors), color="green", linestyle="--", label=f"中位数: {np.median(absolute_errors):.0f}")
axes[0].set_xlabel("绝对误差 |实际-预测|")
axes[0].set_ylabel("频数")
axes[0].set_title("验证集 MAE 分布")
axes[0].legend()

bins_names = [x["bin"] for x in mae_by_bins]
maes_vals = [x["mae"] for x in mae_by_bins]
axes[1].bar(bins_names, maes_vals, color="#00b4d8", alpha=0.8)
axes[1].set_xlabel("价格区间")
axes[1].set_ylabel("MAE")
axes[1].set_title("按价格区间的 MAE")
axes[1].tick_params(axis="x", rotation=45)
plt.suptitle("CatBoost MAE 详细分析")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_catboost_mae_analysis.png"), dpi=150)
plt.close()
print("  04_catboost_mae_analysis.png 已保存")

# 8.5 训练过程
try:
    eval_result = model.get_evals_result()
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in eval_result:
        if "MAE" in eval_result[key]:
            ax.plot(eval_result[key]["MAE"], label=f"{key} MAE")
    ax.axvline(best_iteration, color="red", linestyle="--", label=f"最佳迭代: {best_iteration}")
    ax.set_xlabel("迭代轮数")
    ax.set_ylabel("MAE")
    ax.set_title("CatBoost 训练过程 MAE 变化")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_catboost_training_process.png"), dpi=150)
    plt.close()
    print("  05_catboost_training_process.png 已保存")
except Exception as e:
    print(f"  跳过训练过程图: {e}")


# ============================================================
# 第七部分：保存模型
# ============================================================
print("\n【9. 保存模型】")
model_path = os.path.join(OUTPUT_DIR, "catboost_model.cbm")
model.save_model(model_path)
print(f"  模型已保存: {model_path}")

# 保存特征重要性
importance_df = pd.DataFrame({"feature": feature_cols, "importance": importance}).sort_values("importance", ascending=False)
importance_df.to_csv(os.path.join(OUTPUT_DIR, "catboost_feature_importance.csv"), index=False)
print("  catboost_feature_importance.csv 已保存")

# 保存预处理信息
import json
preprocess_info = {
    "medians": {k: float(v) for k, v in medians.items()},
    "feature_cols": feature_cols,
    "cat_features": cat_features_in_data,
    "reference_date": str(reference_date),
}
info_path = os.path.join(OUTPUT_DIR, "catboost_preprocess_info.json")
with open(info_path, "w") as f:
    json.dump(preprocess_info, f, indent=2)
print(f"  预处理信息已保存: {info_path}")


# ============================================================
# 第八部分：测试集预测
# ============================================================
print("\n【10. 测试集预测】")
print("  加载测试集并预处理...")

# 测试集预处理
test_df = test_df_raw.copy()
test_df["notRepairedDamage"] = test_df["notRepairedDamage"].replace({"-": -1, "0.0": 0, "1.0": 1}).astype(float)

# 日期特征
test_df["regDate_str"] = test_df["regDate"].astype(int).astype(str).str.zfill(8)
test_df["creatDate_str"] = test_df["creatDate"].astype(int).astype(str).str.zfill(8)

test_df["reg_year"] = test_df["regDate_str"].str[:4].astype(int)
test_df["reg_month"] = test_df["regDate_str"].str[4:6].astype(int)
test_df["reg_day"] = test_df["regDate_str"].str[6:8].astype(int)
test_df["reg_date"] = pd.to_datetime(test_df["regDate_str"], format="%Y%m%d", errors="coerce")

test_df["create_year"] = test_df["creatDate_str"].str[:4].astype(int)
test_df["create_month"] = test_df["creatDate_str"].str[4:6].astype(int)
test_df["create_day"] = test_df["creatDate_str"].str[6:8].astype(int)
test_df["create_date"] = pd.to_datetime(test_df["creatDate_str"], format="%Y%m%d", errors="coerce")

# 衍生特征
test_df["car_age"] = (test_df["create_date"] - test_df["reg_date"]).dt.days / 365.25
test_df["car_age"] = test_df["car_age"].clip(0, 100)
test_df["car_age_now"] = (reference_date - test_df["reg_date"]).dt.days / 365.25
test_df["car_age_now"] = test_df["car_age_now"].clip(0, 100)
test_df["listing_days"] = (reference_date - test_df["create_date"]).dt.days.clip(0)
test_df["is_new_car"] = (test_df["car_age"] <= 1).astype(int)

# 删除中间列
test_df = test_df.drop(columns=["regDate", "creatDate", "regDate_str", "creatDate_str",
                                "reg_date", "create_date"], errors="ignore")

# 缺失值处理（用训练集中位数）
for col, med_val in medians.items():
    if col in test_df.columns and test_df[col].isnull().any():
        test_df[col] = test_df[col].fillna(med_val)

# 类别特征转字符串
for col in cat_cols:
    if col in test_df.columns:
        test_df[col] = test_df[col].astype(str)

print(f"  测试集预处理完成: {test_df.shape}")

# 准备特征矩阵
X_test_final = test_df[feature_cols].copy()
print(f"  测试特征矩阵: {X_test_final.shape[0]} × {X_test_final.shape[1]}")

# 预测
print("  开始预测...")
predictions = model.predict(X_test_final)
predictions = np.maximum(predictions, 0)
print(f"  预测完成，共 {len(predictions)} 条")

# 生成提交文件
submit = pd.DataFrame({
    "SaleID": test_df_raw["SaleID"],
    "price": predictions,
})
output_file = os.path.join(SCRIPT_DIR, "used_car_submit_catboost.csv")
submit.to_csv(output_file, index=False)

print(f"\n【11. 提交文件统计】")
print(f"  提交文件: {output_file}")
print(f"  行数: {len(submit)}")
print(f"  SaleID 范围: {submit['SaleID'].min()} ~ {submit['SaleID'].max()}")
print(f"  价格统计: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}, median={np.median(predictions):.2f}")

print(f"\n  预览前10行:")
print(submit.head(10).to_string(index=False))


# ============================================================
# 第九部分：总结报告
# ============================================================
print("\n" + "=" * 70)
print("【CatBoost 建模总结报告】")
print("=" * 70)
print(f"""
数据概况:
  - 训练集: {len(train_df):,} 样本
  - 验证集: {len(y_test):,} 样本
  - 特征数量: {len(feature_cols)}

模型参数:
  - loss_function: MAE
  - iterations: {params['iterations']}
  - learning_rate: {params['learning_rate']}
  - depth: {params['depth']}
  - l2_leaf_reg: {params['l2_leaf_reg']}
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
  模型文件: catboost_model.cbm
  预处理信息: catboost_preprocess_info.json
  特征重要性: catboost_feature_importance.csv
  提交文件: used_car_submit_catboost.csv
  可视化图表: 01~05_catboost_*.png
""")

print("=" * 70)
print("【CatBoost 建模完成】")
print("=" * 70)
