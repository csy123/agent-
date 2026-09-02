#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手车价格预测 - CatBoost 增强特征工程版
================================================
特征工程思路：
  1. Power 特征：异常值裁剪 + 对数变换 + 分箱
  2. 日期特征：增加季度、年代、精确月份差
  3. 交互特征：brand×model, brand×bodyType, fuelType×gearbox 等
  4. 统计特征：品牌/车型/地区的价格统计量
  5. 衍生特征：使用强度、单位功率、车龄平方等
  6. CatBoost 原生有序目标编码（防泄露）
================================================
"""
import os
import json
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
print("【二手车价格预测 - CatBoost 增强特征工程版】")
print("=" * 70)


# ============================================================
# 特征工程：训练集和测试集共用
# ============================================================
def advanced_feature_engineering(df, train_df_ref=None, is_train=True):
    """
    增强特征工程
    :param df: 输入数据
    :param train_df_ref: 训练集参考（测试集预测时使用，提供统计量）
    :param is_train: 是否为训练集
    """
    df = df.copy()

    # ========== 1. 基础预处理 ==========
    # notRepairedDamage
    df["notRepairedDamage"] = df["notRepairedDamage"].replace({"-": -1, "0.0": 0, "1.0": 1}).astype(float)

    # ========== 2. 日期特征工程 ==========
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
    # 参考日期
    if train_df_ref is not None and not is_train:
        ref_date = pd.to_datetime(
            train_df_ref["creatDate"].astype(int).astype(str).str.zfill(8),
            format="%Y%m%d", errors="coerce"
        ).max()
    else:
        ref_date = df["create_date"].max()

    # 车龄（年）
    df["car_age"] = (df["create_date"] - df["reg_date"]).dt.days / 365.25
    df["car_age"] = df["car_age"].clip(0, 100)

    # 车龄（月）—— 更精细
    df["car_age_months"] = df["car_age"] * 12

    # 车龄到参考日期
    df["car_age_now"] = (ref_date - df["reg_date"]).dt.days / 365.25
    df["car_age_now"] = df["car_age_now"].clip(0, 100)

    # 上架时长（天）
    df["listing_days"] = (ref_date - df["create_date"]).dt.days.clip(0)

    # 日期的季度
    df["reg_quarter"] = df["reg_month"].apply(lambda x: (x - 1) // 3 + 1)
    df["create_quarter"] = df["create_month"].apply(lambda x: (x - 1) // 3 + 1)

    # 注册年代
    df["reg_decade"] = (df["reg_year"] // 10) * 10

    # 是否年初/年末注册
    df["reg_is_early_year"] = (df["reg_month"] <= 3).astype(int)
    df["reg_is_late_year"] = (df["reg_month"] >= 10).astype(int)

    # ========== 3. Power 特征工程 ==========
    # 裁剪异常值（功率>600的视为异常）
    df["power_clipped"] = df["power"].clip(0, 600)

    # 对数变换
    df["power_log"] = np.log1p(df["power_clipped"])

    # 分箱（类别特征）
    power_bins = [0, 30, 60, 90, 120, 150, 200, 300, 9999]
    power_labels = ["0-30", "30-60", "60-90", "90-120", "120-150", "150-200", "200-300", "300+"]
    df["power_bin"] = pd.cut(df["power_clipped"], bins=power_bins, labels=power_labels, right=False)

    # 是否高功率
    df["is_high_power"] = (df["power_clipped"] >= 150).astype(int)

    # ========== 4. 里程特征工程 ==========
    # 使用强度：每年行驶里程
    df["usage_intensity"] = df["kilometer"] / (df["car_age"] + 0.5)

    # 里程分箱
    km_bins = [0, 3, 6, 9, 12, 15]
    km_labels = ["0-3万", "3-6万", "6-9万", "9-12万", "12-15万"]
    df["kilometer_bin"] = pd.cut(df["kilometer"], bins=km_bins, labels=km_labels, right=False)

    # ========== 5. 衍生数值特征 ==========
    # 单位车龄功率
    df["power_per_year"] = df["power_clipped"] / (df["car_age"] + 0.5)

    # 功率与里程的交互
    df["power_km_ratio"] = df["power_clipped"] / (df["kilometer"] + 0.5)

    # 车龄平方（捕捉非线性折旧）
    df["car_age_sq"] = df["car_age"] ** 2

    # 是否新车
    df["is_new_car"] = (df["car_age"] <= 1).astype(int)

    # 是否准新车（1-3年）
    df["is_almost_new"] = ((df["car_age"] > 1) & (df["car_age"] <= 3)).astype(int)

    # 是否老车（>8年）
    df["is_old_car"] = (df["car_age"] > 8).astype(int)

    # ========== 6. 交互类别特征 ==========
    # brand × model（最重要的交互）
    df["brand_model"] = df["brand"].astype(str) + "_" + df["model"].astype(str)

    # brand × bodyType
    df["brand_bodyType"] = df["brand"].astype(str) + "_" + df["bodyType"].astype(str)

    # fuelType × gearbox
    df["fuel_gear"] = df["fuelType"].astype(str) + "_" + df["gearbox"].astype(str)

    # brand × fuelType
    df["brand_fuel"] = df["brand"].astype(str) + "_" + df["fuelType"].astype(str)

    # brand × car_age_group
    df["age_group"] = pd.cut(df["car_age"], bins=[0, 3, 6, 10, 100], labels=["0-3y", "3-6y", "6-10y", "10y+"])
    df["brand_age"] = df["brand"].astype(str) + "_" + df["age_group"].astype(str)

    # bodyType × fuelType
    df["body_fuel"] = df["bodyType"].astype(str) + "_" + df["fuelType"].astype(str)

    # regionCode 的省份编码（取前2位简化）
    df["region_province"] = (df["regionCode"] // 100).astype(str)

    # ========== 7. 删除中间列 ==========
    cols_to_drop = [
        "regDate", "creatDate", "regDate_str", "creatDate_str",
        "reg_date", "create_date",
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    return df


# ============================================================
# 第一部分：加载数据
# ============================================================
print("\n【1. 加载数据】")
train_df_raw = pd.read_csv(TRAIN_FILE, sep=" ")
test_df_raw = pd.read_csv(TEST_FILE, sep=" ")
print(f"  训练集: {train_df_raw.shape[0]} 行 × {train_df_raw.shape[1]} 列")
print(f"  测试集: {test_df_raw.shape[0]} 行")


# ============================================================
# 第二部分：特征工程
# ============================================================
print("\n【2. 特征工程】")

# 参考日期
reference_date = pd.to_datetime(
    train_df_raw["creatDate"].astype(int).astype(str).str.zfill(8),
    format="%Y%m%d", errors="coerce"
).max()
print(f"  参考日期: {reference_date.strftime('%Y-%m-%d')}")

# 训练集特征工程
train_df = advanced_feature_engineering(train_df_raw, is_train=True)
print(f"  训练集特征工程完成: {train_df.shape}")
print(f"  新增特征数: {len(train_df.columns)}")

# 测试集特征工程（使用训练集的参考数据）
test_df_engineered = advanced_feature_engineering(
    test_df_raw, train_df_ref=train_df_raw, is_train=False
)
print(f"  测试集特征工程完成: {test_df_engineered.shape}")

# ========== 缺失值处理 ==========
print("\n【3. 缺失值处理】")
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
medians = {}
for col in numeric_cols:
    if train_df[col].isnull().any():
        med = train_df[col].median()
        train_df[col] = train_df[col].fillna(med)
        medians[col] = med
        print(f"  {col}: 用中位数 {med:.2f} 填充")

# 测试集用同样的中位数
for col, med_val in medians.items():
    if col in test_df_engineered.columns and test_df_engineered[col].isnull().any():
        test_df_engineered[col] = test_df_engineered[col].fillna(med_val)

print(f"  共处理 {len(medians)} 个数值列的缺失值")


# ============================================================
# 第三部分：定义特征列
# ============================================================
print("\n【4. 准备特征矩阵】")

# 排除列
exclude_cols = ["SaleID", "name", "price", "notRepairedDamage"]
feature_cols = [col for col in train_df.columns if col not in exclude_cols]

# 确定哪些是类别特征
cat_cols = [
    "brand", "bodyType", "fuelType", "gearbox", "model", "regionCode",
    "power_bin", "kilometer_bin", "age_group",
    "brand_model", "brand_bodyType", "fuel_gear", "brand_fuel",
    "brand_age", "body_fuel", "region_province",
    "reg_decade", "reg_quarter", "create_quarter",
]
# 只保留在 feature_cols 中的
cat_features = [c for c in cat_cols if c in feature_cols]

# 将类别特征转为字符串
for col in cat_features:
    train_df[col] = train_df[col].astype(str)
    if col in test_df_engineered.columns:
        test_df_engineered[col] = test_df_engineered[col].astype(str)

X = train_df[feature_cols].copy()
y = train_df["price"].copy()

print(f"  特征数量: {X.shape[1]}")
print(f"  数值特征: {[c for c in feature_cols if c not in cat_features]}")
print(f"  类别特征（CatBoost 自动编码）: {cat_features}")


# ============================================================
# 第四部分：数据划分
# ============================================================
print("\n【5. 数据划分】")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  训练集: {X_train.shape[0]} 样本")
print(f"  验证集: {X_test.shape[0]} 样本")


# ============================================================
# 第五部分：训练 CatBoost
# ============================================================
print("\n【6. 训练 CatBoost 模型】")
from catboost import CatBoostRegressor, Pool

params = {
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "iterations": 5000,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": 500,
    "early_stopping_rounds": 300,
    "use_best_model": True,
    "task_type": "CPU",
    "allow_writing_files": False,
    "min_data_in_leaf": 20,
    "bagging_temperature": 0.5,
}

print(f"  模型参数:")
for k, v in params.items():
    if k != "verbose":
        print(f"    {k}: {v}")
print("  开始训练...")

train_pool = Pool(X_train, y_train, cat_features=cat_features)
eval_pool = Pool(X_test, y_test, cat_features=cat_features)

model = CatBoostRegressor(**params)
model.fit(train_pool, eval_set=eval_pool, use_best_model=True)

best_iteration = model.get_best_iteration()
print(f"\n  训练完成！最佳迭代轮数: {best_iteration}")


# ============================================================
# 第六部分：模型评估
# ============================================================
print("\n【7. 模型评估】")
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
# 第七部分：MAE 详细分析
# ============================================================
print("\n【8. MAE 详细分析】")

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
# 第八部分：可视化
# ============================================================
print("\n【9. 模型可视化】")

# 8.1 特征重要性
fig, ax = plt.subplots(figsize=(14, 10))
feature_importance = model.get_feature_importance()
importance = np.array(feature_importance)
indices = np.argsort(importance)[::-1]
top_n = min(25, len(feature_cols))
top_features = [feature_cols[i] for i in indices[:top_n]]
top_importance = importance[indices[:top_n]]

ax.barh(range(top_n), top_importance, color="#00b4d8", alpha=0.8)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features)
ax.set_xlabel("特征重要性")
ax.set_title(f"CatBoost 增强特征工程 - Top {top_n} 特征重要性")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_enhanced_feature_importance.png"), dpi=150)
plt.close()
print("  01_enhanced_feature_importance.png 已保存")

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
plt.suptitle("CatBoost 增强版 - 预测值 vs 实际值")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_enhanced_prediction_vs_actual.png"), dpi=150)
plt.close()
print("  02_enhanced_prediction_vs_actual.png 已保存")

# 8.3 MAE 分析
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].hist(absolute_errors, bins=50, color="#ff6b6b", alpha=0.7, edgecolor="white")
axes[0].axvline(absolute_errors.mean(), color="blue", linestyle="--", label=f"MAE: {absolute_errors.mean():.0f}")
axes[0].axvline(np.median(absolute_errors), color="green", linestyle="--", label=f"中位数: {np.median(absolute_errors):.0f}")
axes[0].set_xlabel("绝对误差")
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
plt.suptitle("CatBoost 增强版 - MAE 详细分析")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_enhanced_mae_analysis.png"), dpi=150)
plt.close()
print("  03_enhanced_mae_analysis.png 已保存")

# 8.4 训练过程
try:
    eval_result = model.get_evals_result()
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in eval_result:
        if "MAE" in eval_result[key]:
            ax.plot(eval_result[key]["MAE"], label=f"{key} MAE")
    ax.axvline(best_iteration, color="red", linestyle="--", label=f"最佳迭代: {best_iteration}")
    ax.set_xlabel("迭代轮数")
    ax.set_ylabel("MAE")
    ax.set_title("CatBoost 增强版 - 训练过程")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_enhanced_training_process.png"), dpi=150)
    plt.close()
    print("  04_enhanced_training_process.png 已保存")
except Exception as e:
    print(f"  跳过训练过程图: {e}")


# ============================================================
# 第九部分：保存模型
# ============================================================
print("\n【10. 保存模型】")
model_path = os.path.join(OUTPUT_DIR, "catboost_enhanced_model.cbm")
model.save_model(model_path)
print(f"  模型已保存: {model_path}")

# 保存特征重要性
importance_df = pd.DataFrame({"feature": feature_cols, "importance": importance}).sort_values("importance", ascending=False)
importance_df.to_csv(os.path.join(OUTPUT_DIR, "catboost_enhanced_importance.csv"), index=False)
print(f"  特征重要性已保存（Top 10: {top_features[:5]}）")

# 保存预处理信息
preprocess_info = {
    "medians": {k: float(v) for k, v in medians.items()},
    "feature_cols": feature_cols,
    "cat_features": cat_features,
    "reference_date": str(reference_date),
    "best_iteration": best_iteration,
    "test_mae": float(test_metrics[0]),
}
info_path = os.path.join(OUTPUT_DIR, "catboost_enhanced_info.json")
with open(info_path, "w") as f:
    json.dump(preprocess_info, f, indent=2, ensure_ascii=False)
print(f"  预处理信息已保存: {info_path}")


# ============================================================
# 第十部分：测试集预测
# ============================================================
print("\n【11. 测试集预测】")
print("  准备测试集特征...")

X_test_final = test_df_engineered[feature_cols].copy()

# 将类别特征转为字符串
for col in cat_features:
    if col in X_test_final.columns:
        X_test_final[col] = X_test_final[col].astype(str)

# 填充缺失值
for col, med_val in medians.items():
    if col in X_test_final.columns and X_test_final[col].isnull().any():
        X_test_final[col] = X_test_final[col].fillna(med_val)

print(f"  测试特征矩阵: {X_test_final.shape[0]} × {X_test_final.shape[1]}")
print("  开始预测...")
predictions = model.predict(X_test_final)
predictions = np.maximum(predictions, 0)
print(f"  预测完成，共 {len(predictions)} 条")

# 生成提交文件
submit = pd.DataFrame({
    "SaleID": test_df_raw["SaleID"],
    "price": predictions,
})
output_file = os.path.join(SCRIPT_DIR, "used_car_submit_catboost_enhanced.csv")
submit.to_csv(output_file, index=False)

print(f"\n【12. 提交文件统计】")
print(f"  提交文件: {output_file}")
print(f"  行数: {len(submit)}")
print(f"  SaleID 范围: {submit['SaleID'].min()} ~ {submit['SaleID'].max()}")
print(f"  价格统计: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}, median={np.median(predictions):.2f}")
print(f"\n  预览前10行:")
print(submit.head(10).to_string(index=False))


# ============================================================
# 第十一部分：总结报告
# ============================================================
print("\n" + "=" * 70)
print("【CatBoost 增强特征工程 - 总结报告】")
print("=" * 70)
print(f"""
数据概况:
  - 训练集: {len(train_df):,} 样本
  - 验证集: {len(y_test):,} 样本
  - 特征数量: {len(feature_cols)}
  - 数值特征: {len([c for c in feature_cols if c not in cat_features])}
  - 类别特征: {len(cat_features)}

增强特征工程:
  1. Power 特征: 异常值裁剪 + 对数变换 + 分箱 + 高功率标识
  2. 日期特征: 年/月/日/季度/年代 + 精确月份差 + 是否年初年末
  3. 交互类别: brand×model, brand×bodyType, fuelType×gearbox 等 7 种
  4. 衍生数值: 使用强度、单位功率、车龄平方、功率里程比等
  5. 车辆状态: 是否新车/准新车/老车、是否高功率

模型参数:
  - loss_function: MAE
  - iterations: {params['iterations']}
  - learning_rate: {params['learning_rate']}
  - depth: {params['depth']}
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
    print(f"  {i}. {feat}: {imp:.2f}")

print(f"""
输出文件:
  模型文件: catboost_enhanced_model.cbm
  预处理信息: catboost_enhanced_info.json
  特征重要性: catboost_enhanced_importance.csv
  提交文件: used_car_submit_catboost_enhanced.csv
  可视化图表: 01~04_enhanced_*.png
""")

print("=" * 70)
print("【CatBoost 增强特征工程 - 完成】")
print("=" * 70)
