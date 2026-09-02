#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手车价格预测 - XGBoost 建模
包含：数据预处理、特征工程（日期处理）、模型训练、评估、可视化
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============ 中文字体配置 ============
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# ============ 路径配置 ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "used_car_train_20200313.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "model_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("【二手车价格预测 - XGBoost 建模】")
print("=" * 70)

# ============ 1. 数据加载 ============
print("\n【1. 数据加载】")
df = pd.read_csv(DATA_FILE, sep=" ")
print(f"原始数据: {df.shape[0]} 行 × {df.shape[1]} 列")

# ============ 2. 数据预处理 ============
print("\n【2. 数据预处理】")

# 2.1 处理 notRepairedDamage 字段（"-"值处理）
df["notRepairedDamage"] = df["notRepairedDamage"].replace({"-": -1, "0.0": 0, "1.0": 1}).astype(float)
print("  notRepairedDamage: 处理完成（- → -1, 0.0 → 0, 1.0 → 1）")

# 2.2 处理日期特征 - 核心特征工程
print("\n  --- 日期特征工程 ---")

def process_date_feature(df, col, prefix):
    """将格式为 20040402 的日期字符串转为多个数值特征"""
    # 转为字符串，补全前导零
    df[col] = df[col].astype(int).astype(str).str.zfill(8)
    
    # 提取年、月、日
    df[f"{prefix}_year"] = df[col].str[:4].astype(int)
    df[f"{prefix}_month"] = df[col].str[4:6].astype(int)
    df[f"{prefix}_day"] = df[col].str[6:8].astype(int)
    
    # 转换为 datetime 对象（处理异常日期）
    df[f"{prefix}_date"] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
    
    # 计算距今的年数（以 creatDate 的最大值为参考点）
    return df

# 处理 regDate（注册日期）
df = process_date_feature(df, "regDate", "reg")
print("  regDate → reg_year, reg_month, reg_day, reg_date")

# 处理 creatDate（上架日期）
df = process_date_feature(df, "creatDate", "create")
print("  creatDate → create_year, create_month, create_day, create_date")

# 2.3 基于日期计算衍生特征
print("\n  --- 日期衍生特征 ---")

# 车龄（上架时的车龄，单位：年）
df["car_age"] = (df["create_date"] - df["reg_date"]).dt.days / 365.25
# 处理异常值（负数可能是数据错误）
df["car_age"] = df["car_age"].clip(0, 100)
print("  car_age: 车龄（上架时）")

# 使用 creatDate 的最大值作为当前时间参考
reference_date = df["create_date"].max()
print(f"  参考日期（最大 creatDate）: {reference_date.strftime('%Y-%m-%d')}")

# 车龄（截至参考日期）
df["car_age_now"] = (reference_date - df["reg_date"]).dt.days / 365.25
df["car_age_now"] = df["car_age_now"].clip(0, 100)
print("  car_age_now: 车龄（截至参考日期）")

# 上架时长（单位：天）
df["listing_days"] = (reference_date - df["create_date"]).dt.days.clip(0)
print("  listing_days: 上架时长（天）")

# 是否为新年份
df["is_new_car"] = (df["car_age"] <= 1).astype(int)
print("  is_new_car: 是否新车（车龄≤1年）")

# 2.4 删除原始日期字符串列
df = df.drop(columns=["regDate", "creatDate", "reg_date", "create_date"], errors="ignore")

# ============ 3. 缺失值处理 ============
print("\n【3. 缺失值处理】")

# 数值列用中位数填充
numeric_cols_with_missing = df.select_dtypes(include=[np.number]).columns[
    df.select_dtypes(include=[np.number]).isnull().any()
]

for col in numeric_cols_with_missing:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"  {col}: 用中位数 {median_val:.2f} 填充")

# ============ 4. 特征编码 ============
print("\n【4. 特征编码】")

# 识别类别特征（数值型但实际是编码的类别）
cat_cols = ["brand", "bodyType", "fuelType", "gearbox", "model", "regionCode"]
print(f"  类别特征: {cat_cols}")

# 使用目标编码（Target Encoding）处理高频类别特征
from sklearn.model_selection import KFold

def target_encode(df, cols, target):
    """目标编码"""
    df_encoded = df.copy()
    for col in cols:
        # 计算每个类别的目标均值
        target_mean = df.groupby(col)[target].mean()
        global_mean = df[target].mean()
        # 用全局均值替换低频类别
        df_encoded[col] = df[col].map(target_mean).fillna(global_mean)
    return df_encoded

df = target_encode(df, cat_cols, "price")
print("  已使用目标编码处理类别特征")

# ============ 5. 准备特征矩阵 ============
print("\n【5. 准备特征矩阵】")

# 定义特征列
exclude_cols = ["SaleID", "name", "price", "notRepairedDamage"]
feature_cols = [col for col in df.columns if col not in exclude_cols]

# 分离特征和目标
X = df[feature_cols].copy()
y = df["price"].copy()

print(f"  特征数量: {X.shape[1]}")
print(f"  样本数量: {X.shape[0]}")
print(f"  特征列表: {feature_cols}")

# ============ 6. 数据划分 ============
print("\n【6. 数据划分】")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  训练集: {X_train.shape[0]} 样本")
print(f"  测试集: {X_test.shape[0]} 样本")

# ============ 7. 训练 XGBoost 模型 ============
print("\n【7. 训练 XGBoost 模型】")

from xgboost import XGBRegressor

# 模型参数
params = {
    "n_estimators": 1000,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": 0,
    "tree_method": "hist",  # 使用直方图优化，训练更快
    "eval_metric": "mae",  # 使用 MAE 作为验证指标
}

print(f"  模型参数: {params}")
print("  开始训练...")

model = XGBRegressor(**params)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

print("  训练完成！")

# ============ 8. 模型评估 ============
print("\n【8. 模型评估】")

# 预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 计算指标
def evaluate_model(y_true, y_pred, dataset_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100  # +1 避免除零
    
    print(f"\n  {dataset_name}:")
    print(f"    MAE  (平均绝对误差): {mae:,.2f}")
    print(f"    RMSE (均方根误差):   {rmse:,.2f}")
    print(f"    R²   (决定系数):     {r2:.4f}")
    print(f"    MAPE (平均百分比误差): {mape:.2f}%")
    return mae, rmse, r2, mape

train_metrics = evaluate_model(y_train, y_train_pred, "训练集")
test_metrics = evaluate_model(y_test, y_test_pred, "测试集")

# ============ 8.1 详细 MAE 分析 ============
print("\n【8.1 MAE 详细分析（按价格区间）】")

# 按价格区间统计 MAE
price_bins = [0, 1000, 3000, 5000, 10000, 20000, 50000, float("inf")]
price_labels = ["0-1k", "1k-3k", "3k-5k", "5k-10k", "10k-20k", "20k-50k", "50k+"]

print("\n  测试集按价格区间的 MAE:")
print(f"  {'价格区间':<10s} {'样本数':>8s} {'MAE':>10s} {'RMSE':>10s} {'MAPE':>10s}")
print("  " + "-" * 50)

for i in range(len(price_bins) - 1):
    mask = (y_test >= price_bins[i]) & (y_test < price_bins[i + 1])
    if mask.sum() > 0:
        bin_mae = mean_absolute_error(y_test[mask], y_test_pred[mask])
        bin_rmse = np.sqrt(mean_squared_error(y_test[mask], y_test_pred[mask]))
        bin_mape = np.mean(np.abs((y_test[mask] - y_test_pred[mask]) / (y_test[mask] + 1))) * 100
        print(f"  {price_labels[i]:<10s} {mask.sum():>8d} {bin_mae:>10.2f} {bin_rmse:>10.2f} {bin_mape:>9.2f}%")

# MAE 分布统计
absolute_errors = np.abs(y_test - y_test_pred)
print(f"\n  MAE 分布统计:")
print(f"    - 最小绝对误差: {absolute_errors.min():.2f}")
print(f"    - 最大绝对误差: {absolute_errors.max():.2f}")
print(f"    - 平均绝对误差 (MAE): {absolute_errors.mean():.2f}")
print(f"    - 中位数绝对误差: {np.median(absolute_errors):.2f}")
print(f"    - 90%分位数绝对误差: {np.percentile(absolute_errors, 90):.2f}")
print(f"    - 95%分位数绝对误差: {np.percentile(absolute_errors, 95):.2f}")
print(f"    - 99%分位数绝对误差: {np.percentile(absolute_errors, 99):.2f}")

# MAE 在各区间的占比
print(f"\n  误差区间占比:")
error_ranges = [0, 100, 500, 1000, 2000, 5000, float("inf")]
error_labels = ["<100", "100-500", "500-1k", "1k-2k", "2k-5k", ">5k"]
for i in range(len(error_ranges) - 1):
    pct = ((absolute_errors >= error_ranges[i]) & (absolute_errors < error_ranges[i + 1])).sum() / len(absolute_errors) * 100
    print(f"    误差 {error_labels[i]:<10s}: {pct:.2f}%")

# 可视化 MAE 分布
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 绝对误差分布直方图
axes[0].hist(absolute_errors, bins=50, color="#ff6b6b", alpha=0.7, edgecolor="white")
axes[0].axvline(absolute_errors.mean(), color="blue", linestyle="--", 
                label=f"MAE: {absolute_errors.mean():.0f}")
axes[0].axvline(np.median(absolute_errors), color="green", linestyle="--", 
                label=f"中位数: {np.median(absolute_errors):.0f}")
axes[0].set_xlabel("绝对误差 |实际-预测|")
axes[0].set_ylabel("频数")
axes[0].set_title("测试集 MAE 分布")
axes[0].legend()

# 按价格区间的 MAE 柱状图
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

plt.suptitle("MAE 详细分析")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_mae_analysis.png"), dpi=150)
plt.close()
print("\n  08_mae_analysis.png 已保存")

# ============ 9. 模型可视化 ============
print("\n【9. 模型可视化】")

# 9.1 特征重要性
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
ax.set_title(f"XGBoost Top {top_n} 特征重要性")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_feature_importance.png"), dpi=150)
plt.close()
print("  01_feature_importance.png 已保存")

# 9.2 预测值 vs 实际值
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 训练集
axes[0].scatter(y_train, y_train_pred, alpha=0.3, s=5, color="#00b4d8")
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
             "r--", lw=2, label="理想预测线")
axes[0].set_xlabel("实际价格")
axes[0].set_ylabel("预测价格")
axes[0].set_title(f"训练集: R²={train_metrics[2]:.4f}")
axes[0].legend()

# 测试集
axes[1].scatter(y_test, y_test_pred, alpha=0.3, s=5, color="#6bcf7f")
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             "r--", lw=2, label="理想预测线")
axes[1].set_xlabel("实际价格")
axes[1].set_ylabel("预测价格")
axes[1].set_title(f"测试集: R²={test_metrics[2]:.4f}")
axes[1].legend()

plt.suptitle("预测值 vs 实际值")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_prediction_vs_actual.png"), dpi=150)
plt.close()
print("  02_prediction_vs_actual.png 已保存")

# 9.3 残差分析
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 残差分布
residuals = y_test - y_test_pred
axes[0].hist(residuals, bins=50, color="#ff6b6b", alpha=0.7, edgecolor="white")
axes[0].axvline(0, color="black", linestyle="--", label="0基准线")
axes[0].set_xlabel("残差 (实际 - 预测)")
axes[0].set_ylabel("频数")
axes[0].set_title("测试集残差分布")
axes[0].legend()

# 残差 vs 预测值
axes[1].scatter(y_test_pred, residuals, alpha=0.3, s=5, color="#00b4d8")
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_xlabel("预测价格")
axes[1].set_ylabel("残差")
axes[1].set_title("残差 vs 预测值")

plt.suptitle("模型残差分析")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_residual_analysis.png"), dpi=150)
plt.close()
print("  03_residual_analysis.png 已保存")

# 9.4 训练过程（如果有记录）
try:
    # 使用 XGBoost 的内置评估结果
    eval_results = model.evals_result()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in eval_results:
        # 获取 MAE 指标
        if "mae" in eval_results[key]:
            ax.plot(eval_results[key]["mae"], label=f"{key} MAE")
        elif "validation_0" in eval_results and "mae" in eval_results["validation_0"]:
            # 尝试不同的 key 格式
            pass
    
    ax.set_xlabel("迭代轮数")
    ax.set_ylabel("MAE")
    ax.set_title("训练过程 MAE 变化")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_training_process.png"), dpi=150)
    plt.close()
    print("  04_training_process.png 已保存")
except:
    print("  跳过训练过程图（无评估结果）")

# ============ 10. 模型对比与保存 ============
print("\n【10. 模型保存】")

# 保存模型
model_path = os.path.join(OUTPUT_DIR, "xgboost_model.json")
model.save_model(model_path)
print(f"  模型已保存: {model_path}")

# 保存特征重要性表
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importance
}).sort_values("importance", ascending=False)
importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
print(f"  特征重要性表已保存")

# ============ 11. 总结报告 ============
print("\n" + "=" * 70)
print("【建模总结报告】")
print("=" * 70)
print(f"""
数据概况:
  - 总样本数: {len(df):,}
  - 特征数量: {len(feature_cols)}
  - 训练集:   {X_train.shape[0]:,} 样本
  - 测试集:   {X_test.shape[0]:,} 样本

日期特征工程:
  - regDate (20040402) → reg_year, reg_month, reg_day
  - creatDate (20160404) → create_year, create_month, create_day
  - 衍生特征: car_age, car_age_now, listing_days, is_new_car

模型性能:
  训练集 R²:  {train_metrics[2]:.4f}
  测试集 R²:  {test_metrics[2]:.4f}
  训练集 RMSE: {train_metrics[1]:,.2f}
  测试集 RMSE: {test_metrics[1]:,.2f}
  测试集 MAE:  {test_metrics[0]:,.2f}

Top 5 重要特征:
""")

for i, (feat, imp) in enumerate(zip(top_features[:5], top_importance[:5]), 1):
    print(f"  {i}. {feat}: {imp:.4f}")

print(f"""
输出文件:
  模型文件: xgboost_model.json
  特征重要性: feature_importance.csv
  可视化图表: 01~04_*.png
""")

print("=" * 70)
print("【建模完成】")
print("=" * 70)
