#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手车价格预测 - Stacking 集成与全面优化
================================================
优化策略：
  1. 五折交叉目标编码（防止数据泄露）
  2. 统计聚合特征（brand/model/regionCode 的 price 统计量）
  3. v_ 特征交互（v_0*v_3, v_0+v_12 等）
  4. 三模型 Stacking 集成（CatBoost + LightGBM + XGBoost）
  5. log1p(price) 目标变换（改善高价区间）
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
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
print("【二手车价格预测 - Stacking 集成与全面优化】")
print("=" * 70)


# ============================================================
# 第一部分：基础特征工程（与增强版一致）
# ============================================================
def basic_feature_engineering(df, train_df_ref=None, is_train=True):
    """基础特征工程"""
    df = df.copy()

    # notRepairedDamage
    df["notRepairedDamage"] = df["notRepairedDamage"].replace({"-": -1, "0.0": 0, "1.0": 1}).astype(float)

    # 日期特征
    df["regDate_str"] = df["regDate"].astype(int).astype(str).str.zfill(8)
    df["reg_year"] = df["regDate_str"].str[:4].astype(int)
    df["reg_month"] = df["regDate_str"].str[4:6].astype(int)
    df["reg_day"] = df["regDate_str"].str[6:8].astype(int)
    df["reg_date"] = pd.to_datetime(df["regDate_str"], format="%Y%m%d", errors="coerce")

    df["creatDate_str"] = df["creatDate"].astype(int).astype(str).str.zfill(8)
    df["create_year"] = df["creatDate_str"].str[:4].astype(int)
    df["create_month"] = df["creatDate_str"].str[4:6].astype(int)
    df["create_day"] = df["creatDate_str"].str[6:8].astype(int)
    df["create_date"] = pd.to_datetime(df["creatDate_str"], format="%Y%m%d", errors="coerce")

    # 参考日期
    if train_df_ref is not None and not is_train:
        ref_date = pd.to_datetime(
            train_df_ref["creatDate"].astype(int).astype(str).str.zfill(8),
            format="%Y%m%d", errors="coerce"
        ).max()
    else:
        ref_date = df["create_date"].max()

    # 日期衍生特征
    df["car_age"] = (df["create_date"] - df["reg_date"]).dt.days / 365.25
    df["car_age"] = df["car_age"].clip(0, 100)
    df["car_age_months"] = df["car_age"] * 12
    df["car_age_now"] = (ref_date - df["reg_date"]).dt.days / 365.25
    df["car_age_now"] = df["car_age_now"].clip(0, 100)
    df["listing_days"] = (ref_date - df["create_date"]).dt.days.clip(0)
    df["reg_quarter"] = df["reg_month"].apply(lambda x: (x - 1) // 3 + 1)
    df["create_quarter"] = df["create_month"].apply(lambda x: (x - 1) // 3 + 1)
    df["reg_decade"] = (df["reg_year"] // 10) * 10
    df["reg_is_early_year"] = (df["reg_month"] <= 3).astype(int)
    df["reg_is_late_year"] = (df["reg_month"] >= 10).astype(int)

    # Power 特征
    df["power_clipped"] = df["power"].clip(0, 600)
    df["power_log"] = np.log1p(df["power_clipped"])
    power_bins = [0, 30, 60, 90, 120, 150, 200, 300, 9999]
    power_labels = ["0-30", "30-60", "60-90", "90-120", "120-150", "150-200", "200-300", "300+"]
    df["power_bin"] = pd.cut(df["power_clipped"], bins=power_bins, labels=power_labels, right=False)
    df["is_high_power"] = (df["power_clipped"] >= 150).astype(int)

    # 里程特征
    df["usage_intensity"] = df["kilometer"] / (df["car_age"] + 0.5)
    km_bins = [0, 3, 6, 9, 12, 15]
    km_labels = ["0-3万", "3-6万", "6-9万", "9-12万", "12-15万"]
    df["kilometer_bin"] = pd.cut(df["kilometer"], bins=km_bins, labels=km_labels, right=False)

    # 衍生数值特征
    df["power_per_year"] = df["power_clipped"] / (df["car_age"] + 0.5)
    df["power_km_ratio"] = df["power_clipped"] / (df["kilometer"] + 0.5)
    df["car_age_sq"] = df["car_age"] ** 2
    df["is_new_car"] = (df["car_age"] <= 1).astype(int)
    df["is_almost_new"] = ((df["car_age"] > 1) & (df["car_age"] <= 3)).astype(int)
    df["is_old_car"] = (df["car_age"] > 8).astype(int)

    # 交互类别特征
    df["brand_model"] = df["brand"].astype(str) + "_" + df["model"].astype(str)
    df["brand_bodyType"] = df["brand"].astype(str) + "_" + df["bodyType"].astype(str)
    df["fuel_gear"] = df["fuelType"].astype(str) + "_" + df["gearbox"].astype(str)
    df["brand_fuel"] = df["brand"].astype(str) + "_" + df["fuelType"].astype(str)
    df["age_group"] = pd.cut(df["car_age"], bins=[0, 3, 6, 10, 100], labels=["0-3y", "3-6y", "6-10y", "10y+"])
    df["brand_age"] = df["brand"].astype(str) + "_" + df["age_group"].astype(str)
    df["body_fuel"] = df["bodyType"].astype(str) + "_" + df["fuelType"].astype(str)
    df["region_province"] = (df["regionCode"] // 100).astype(str)

    # 删除中间列
    cols_to_drop = ["regDate", "creatDate", "regDate_str", "creatDate_str", "reg_date", "create_date"]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    return df


# ============================================================
# 第二部分：五折交叉目标编码（防泄露）
# ============================================================
def kfold_target_encode(train_df, test_df, cols, target="price", n_splits=5):
    """
    使用五折交叉进行目标编码，防止数据泄露
    返回: train_df, test_df (编码后的结果)
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    global_mean = train_df[target].mean()
    enc_stats = {}

    for col in cols:
        if col not in train_df.columns:
            continue

        # 计算全局均值
        global_mean_col = train_df[target].mean()

        # 五折交叉编码训练集
        train_encoded = np.zeros(len(train_df))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(train_df):
            train_mean = train_df.iloc[train_idx].groupby(col)[target].mean()
            train_encoded[val_idx] = train_df.iloc[val_idx][col].map(train_mean).fillna(global_mean_col)

        train_df[f"{col}_te"] = train_encoded

        # 用全量数据计算编码映射，应用到测试集
        full_mean = train_df.groupby(col)[target].mean()
        enc_stats[col] = full_mean

        test_df[f"{col}_te"] = test_df[col].map(full_mean).fillna(global_mean_col)

    return train_df, test_df, enc_stats


# ============================================================
# 第三部分：统计聚合特征
# ============================================================
def add_statistical_features(train_df, test_df, cat_cols, target="price"):
    """
    添加统计聚合特征（mean, median, std, min, max, skew）
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    stat_encodings = {}

    for col in cat_cols:
        if col not in train_df.columns:
            continue

        # 计算训练集统计量
        stats = train_df.groupby(col)[target].agg(["mean", "median", "std", "min", "max", "count"])
        stats.columns = [f"{col}_stat_{c}" for c in stats.columns]

        # 保存编码映射
        stat_encodings[col] = stats

        # 应用到训练集
        train_df = train_df.merge(
            stats, left_on=col, right_index=True, how="left"
        )

        # 应用到测试集
        test_df = test_df.merge(
            stats, left_on=col, right_index=True, how="left"
        )

        # 填充缺失值
        for stat_col in stats.columns:
            global_val = train_df[stat_col].mean()
            train_df[stat_col] = train_df[stat_col].fillna(global_val)
            test_df[stat_col] = test_df[stat_col].fillna(global_val)

    return train_df, test_df, stat_encodings


def add_statistical_features_single(df, cat_cols, stat_encodings):
    """为单个数据集添加统计特征（使用预计算的编码）"""
    df = df.copy()
    for col, stats in stat_encodings.items():
        if col not in df.columns:
            continue
        df = df.merge(stats, left_on=col, right_index=True, how="left")
        for stat_col in stats.columns:
            if stat_col in df.columns:
                df[stat_col] = df[stat_col].fillna(0)
    return df


# ============================================================
# 第四部分：v_ 特征交互
# ============================================================
def add_v_interactions(df):
    """添加 v_ 匿名特征的交互特征"""
    df = df.copy()
    v_cols = [c for c in df.columns if c.startswith("v_")]

    # 重要 v_ 特征的两两交互
    important_v = ["v_0", "v_3", "v_8", "v_10", "v_12"]

    for i, v1 in enumerate(important_v):
        if v1 not in df.columns:
            continue
        for j, v2 in enumerate(important_v):
            if j <= i or v2 not in df.columns:
                continue
            # 乘法交互
            df[f"{v1}_x_{v2}"] = df[v1] * df[v2]

    # 重要特征的平方
    for v in important_v:
        if v in df.columns:
            df[f"{v}_sq"] = df[v] ** 2

    # v_ 特征的统计聚合
    df["v_mean"] = df[v_cols].mean(axis=1)
    df["v_std"] = df[v_cols].std(axis=1)
    df["v_min"] = df[v_cols].min(axis=1)
    df["v_max"] = df[v_cols].max(axis=1)
    df["v_sum"] = df[v_cols].sum(axis=1)

    return df


# ============================================================
# 第五部分：加载和预处理数据
# ============================================================
print("\n【1. 加载数据】")
train_df_raw = pd.read_csv(TRAIN_FILE, sep=" ")
test_df_raw = pd.read_csv(TEST_FILE, sep=" ")
print(f"  训练集: {train_df_raw.shape[0]} 行")
print(f"  测试集: {test_df_raw.shape[0]} 行")

# 参考日期
reference_date = pd.to_datetime(
    train_df_raw["creatDate"].astype(int).astype(str).str.zfill(8),
    format="%Y%m%d", errors="coerce"
).max()
print(f"  参考日期: {reference_date.strftime('%Y-%m-%d')}")


# ============================================================
# 第六部分：特征工程（分步进行，方便后续编码）
# ============================================================
print("\n【2. 基础特征工程】")
train_df = basic_feature_engineering(train_df_raw, is_train=True)
print(f"  训练集: {train_df.shape}")

test_df = basic_feature_engineering(test_df_raw, train_df_ref=train_df_raw, is_train=False)
print(f"  测试集: {test_df.shape}")

# 缺失值处理
print("\n【3. 缺失值处理】")
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
medians = {}
for col in numeric_cols:
    if train_df[col].isnull().any():
        med = train_df[col].median()
        train_df[col] = train_df[col].fillna(med)
        medians[col] = med
        print(f"  {col}: {med:.2f}")

for col, med_val in medians.items():
    if col in test_df.columns and test_df[col].isnull().any():
        test_df[col] = test_df[col].fillna(med_val)


# ============================================================
# 第七部分：v_ 特征交互
# ============================================================
print("\n【4. v_ 特征交互】")
train_df = add_v_interactions(train_df)
test_df = add_v_interactions(test_df)
print(f"  训练集: {train_df.shape}")
print(f"  测试集: {test_df.shape}")


# ============================================================
# 第八部分：五折交叉目标编码
# ============================================================
print("\n【5. 五折交叉目标编码】")
cat_cols_for_te = ["brand", "model", "bodyType", "fuelType", "gearbox", "regionCode"]

train_df, test_df, te_encodings = kfold_target_encode(
    train_df, test_df, cat_cols_for_te, target="price", n_splits=5
)
print(f"  已对 {len(cat_cols_for_te)} 个特征进行五折目标编码")


# ============================================================
# 第九部分：统计聚合特征
# ============================================================
print("\n【6. 统计聚合特征】")
cat_cols_for_stat = ["brand", "model", "regionCode", "bodyType", "fuelType", "brand_model"]

train_df, test_df, stat_encodings = add_statistical_features(
    train_df, test_df, cat_cols_for_stat, target="price"
)
print(f"  已添加 {len(cat_cols_for_stat) * 6} 个统计特征")


# ============================================================
# 第十部分：准备特征矩阵
# ============================================================
print("\n【7. 准备特征矩阵】")

exclude_cols = ["SaleID", "name", "price", "notRepairedDamage", "regDate", "creatDate"]
feature_cols = [col for col in train_df.columns if col not in exclude_cols]

# 类别特征（用于 CatBoost）
cat_cols = ["brand", "bodyType", "fuelType", "gearbox", "model", "regionCode",
            "power_bin", "kilometer_bin", "age_group",
            "brand_model", "brand_bodyType", "fuel_gear", "brand_fuel",
            "brand_age", "body_fuel", "region_province"]
cat_features = [c for c in cat_cols if c in train_df.columns]

# 为 CatBoost 保留字符串类型的类别特征
X_catboost = train_df[feature_cols].copy()
for col in cat_features:
    X_catboost[col] = X_catboost[col].astype(str)

# 为 LightGBM/XGBoost 创建数值版本的特征矩阵（对类别特征进行标签编码）
X_other = train_df[feature_cols].copy()
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for col in cat_features:
    if col in X_other.columns:
        le = LabelEncoder()
        # 合并训练集和测试集的类别，确保编码一致
        combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)], axis=0)
        le.fit(combined)
        X_other[col] = le.transform(train_df[col].astype(str))
        label_encoders[col] = le

y = train_df["price"].copy()

print(f"  CatBoost 特征数: {X_catboost.shape[1]}")
print(f"  LightGBM/XGBoost 特征数: {X_other.shape[1]}")
print(f"  数值特征: {len([c for c in feature_cols if c not in cat_features])}")
print(f"  类别特征: {len(cat_features)}")


# ============================================================
# 第十一部分：数据划分
# ============================================================
print("\n【8. 数据划分】")
# 使用相同的索引划分，保证三个模型使用相同的数据
indices = np.arange(len(y))
train_idx, valid_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_train_catboost = X_catboost.iloc[train_idx]
X_valid_catboost = X_catboost.iloc[valid_idx]

X_train_other = X_other.iloc[train_idx]
X_valid_other = X_other.iloc[valid_idx]

y_train = y.iloc[train_idx]
y_valid = y.iloc[valid_idx]

print(f"  训练集: {X_train_catboost.shape[0]} 样本")
print(f"  验证集: {X_valid_catboost.shape[0]} 样本")


# ============================================================
# 第十二部分：模型 1 - CatBoost（log1p 变换）
# ============================================================
print("\n【9. 训练 CatBoost 模型】")
from catboost import CatBoostRegressor, Pool

y_train_log = np.log1p(y_train)
y_valid_log = np.log1p(y_valid)

cb_params = {
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": 500,
    "early_stopping_rounds": 200,
    "use_best_model": True,
    "allow_writing_files": False,
    "min_data_in_leaf": 20,
    "bagging_temperature": 0.5,
    "random_strength": 1.5,
}

train_pool = Pool(X_train_catboost, y_train_log, cat_features=cat_features)
valid_pool = Pool(X_valid_catboost, y_valid_log, cat_features=cat_features)

catboost_model = CatBoostRegressor(**cb_params)
catboost_model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

catboost_best_iter = catboost_model.get_best_iteration()
print(f"  CatBoost 最佳迭代: {catboost_best_iter}")


# ============================================================
# 第十三部分：模型 2 - LightGBM
# ============================================================
print("\n【10. 训练 LightGBM 模型】")
import lightgbm as lgb

lgb_params = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 127,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": 500,
    "num_threads": -1,
}

lgb_train = lgb.Dataset(X_train_other, y_train_log)
lgb_valid = lgb.Dataset(X_valid_other, y_valid_log, reference=lgb_train)

lightgbm_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=3000,
    valid_sets=[lgb_valid],
    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)],
)

lgb_best_iter = lightgbm_model.best_iteration
print(f"  LightGBM 最佳迭代: {lgb_best_iter}")


# ============================================================
# 第十四部分：模型 3 - XGBoost
# ============================================================
print("\n【11. 训练 XGBoost 模型】")
from xgboost import XGBRegressor

xgb_params = {
    "n_estimators": 3000,
    "max_depth": 8,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": 1,
    "tree_method": "hist",
    "eval_metric": "mae",
    "early_stopping_rounds": 200,
}

xgboost_model = XGBRegressor(**xgb_params)
xgboost_model.fit(
    X_train_other, y_train_log,
    eval_set=[(X_valid_other, y_valid_log)],
    verbose=500
)

xgb_best_iter = xgboost_model.best_iteration
print(f"  XGBoost 最佳迭代: {xgb_best_iter}")


# ============================================================
# 第十五部分：各模型评估
# ============================================================
print("\n【12. 各模型评估】")

# 预测（反变换）
y_pred_cb = np.expm1(catboost_model.predict(X_valid_catboost))
y_pred_lgb = np.expm1(lightgbm_model.predict(X_valid_other))
y_pred_xgb = np.expm1(xgboost_model.predict(X_valid_other))

# 计算 MAE
mae_cb = mean_absolute_error(y_valid, y_pred_cb)
mae_lgb = mean_absolute_error(y_valid, y_pred_lgb)
mae_xgb = mean_absolute_error(y_valid, y_pred_xgb)

print(f"  CatBoost MAE: {mae_cb:.2f}")
print(f"  LightGBM MAE: {mae_lgb:.2f}")
print(f"  XGBoost MAE: {mae_xgb:.2f}")

# 优化加权平均权重
print("\n  --- 优化融合权重 ---")
best_weighted_mae = float('inf')
best_weights = [0.4, 0.3, 0.3]
weight_combos = []
for w1 in np.arange(0.1, 0.7, 0.05):
    for w2 in np.arange(0.1, 0.7, 0.05):
        w3 = 1 - w1 - w2
        if w3 >= 0.05:
            weight_combos.append((w1, w2, w3))

for w1, w2, w3 in weight_combos:
    pred = w1 * y_pred_cb + w2 * y_pred_lgb + w3 * y_pred_xgb
    mae = mean_absolute_error(y_valid, pred)
    if mae < best_weighted_mae:
        best_weighted_mae = mae
        best_weights = [w1, w2, w3]

weights = best_weights
y_pred_weighted = weights[0] * y_pred_cb + weights[1] * y_pred_lgb + weights[2] * y_pred_xgb
mae_weighted = mean_absolute_error(y_valid, y_pred_weighted)
print(f"  最优权重: CatBoost={weights[0]:.2f}, LightGBM={weights[1]:.2f}, XGBoost={weights[2]:.2f}")
print(f"  加权平均 MAE: {mae_weighted:.2f}")


# ============================================================
# 第十六部分：Stacking 集成
# ============================================================
print("\n【13. Stacking 集成】")

# 使用 Ridge 作为元学习器
stack_X_train = np.column_stack([
    np.expm1(catboost_model.predict(X_train_catboost)),
    np.expm1(lightgbm_model.predict(X_train_other)),
    np.expm1(xgboost_model.predict(X_train_other)),
])

stack_X_valid = np.column_stack([
    y_pred_cb,
    y_pred_lgb,
    y_pred_xgb,
])

# 训练 Ridge 元学习器
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(stack_X_train, y_train)

# Stacking 预测
y_pred_stacked = ridge_model.predict(stack_X_valid)
mae_stacked = mean_absolute_error(y_valid, y_pred_stacked)
print(f"  Stacking MAE: {mae_stacked:.2f}")

# 选择最优方案
best_mae = min(mae_cb, mae_lgb, mae_xgb, mae_weighted, mae_stacked)
print(f"\n  最优方案 MAE: {best_mae:.2f}")

if best_mae == mae_stacked:
    best_method = "Stacking"
elif best_mae == mae_weighted:
    best_method = "加权平均"
elif best_mae == mae_cb:
    best_method = "CatBoost"
elif best_mae == mae_lgb:
    best_method = "LightGBM"
else:
    best_method = "XGBoost"

print(f"  最优方法: {best_method}")


# ============================================================
# 第十七部分：MAE 详细分析
# ============================================================
print("\n【14. MAE 详细分析】")

# 使用最优方法重新评估
if best_method == "Stacking":
    y_best_pred = y_pred_stacked
elif best_method == "加权平均":
    y_best_pred = y_pred_weighted
elif best_method == "CatBoost":
    y_best_pred = y_pred_cb
elif best_method == "LightGBM":
    y_best_pred = y_pred_lgb
else:
    y_best_pred = y_pred_xgb

price_bins = [0, 1000, 3000, 5000, 10000, 20000, 50000, float("inf")]
price_labels = ["0-1k", "1k-3k", "3k-5k", "5k-10k", "10k-20k", "20k-50k", "50k+"]

print(f"\n  价格区间 MAE 分析（{best_method}）:")
print(f"  {'价格区间':<10s} {'样本数':>8s} {'MAE':>10s}")
print("  " + "-" * 30)

for i in range(len(price_bins) - 1):
    mask = (y_valid >= price_bins[i]) & (y_valid < price_bins[i + 1])
    if mask.sum() > 0:
        bin_mae = mean_absolute_error(y_valid[mask], y_best_pred[mask])
        print(f"  {price_labels[i]:<10s} {mask.sum():>8d} {bin_mae:>10.2f}")


# ============================================================
# 第十八部分：可视化
# ============================================================
print("\n【15. 模型可视化】")

# 模型对比图
fig, ax = plt.subplots(figsize=(10, 6))
methods = ["CatBoost", "LightGBM", "XGBoost", "加权平均", "Stacking"]
maes = [mae_cb, mae_lgb, mae_xgb, mae_weighted, mae_stacked]
colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

bars = ax.bar(methods, maes, color=colors, alpha=0.8)
ax.axhline(y=400, color="red", linestyle="--", label="目标 MAE: 400")
ax.set_ylabel("MAE")
ax.set_title("各方案 MAE 对比")
ax.legend()

for bar, mae in zip(bars, maes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
            f"{mae:.1f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_ensemble_mae_comparison.png"), dpi=150)
plt.close()
print("  05_ensemble_mae_comparison.png 已保存")


# 预测 vs 实际
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].scatter(y_valid, y_best_pred, alpha=0.3, s=5, color="#00b4d8")
axes[0].plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], "r--", lw=2)
axes[0].set_xlabel("实际价格")
axes[0].set_ylabel("预测价格")
axes[0].set_title(f"{best_method} - 预测 vs 实际")

# 残差分布
residuals = y_valid - y_best_pred
axes[1].hist(residuals, bins=50, color="#ff6b6b", alpha=0.7)
axes[1].axvline(0, color="black", linestyle="--")
axes[1].set_xlabel("残差")
axes[1].set_ylabel("频数")
axes[1].set_title("残差分布")

plt.suptitle(f"最优方案: {best_method} (MAE: {best_mae:.2f})")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_best_model_analysis.png"), dpi=150)
plt.close()
print("  06_best_model_analysis.png 已保存")


# ============================================================
# 第十九部分：保存模型
# ============================================================
print("\n【16. 保存模型】")

# 保存 CatBoost 模型
catboost_model.save_model(os.path.join(OUTPUT_DIR, "catboost_v2_model.cbm"))

# 保存 LightGBM 模型
lightgbm_model.save_model(os.path.join(OUTPUT_DIR, "lightgbm_v2_model.txt"))

# 保存 XGBoost 模型
xgboost_model.save_model(os.path.join(OUTPUT_DIR, "xgboost_v2_model.json"))

# 保存 Ridge 元学习器
import joblib
joblib.dump(ridge_model, os.path.join(OUTPUT_DIR, "ridge_meta_model.pkl"))

# 保存预处理信息
preprocess_info = {
    "medians": {k: float(v) for k, v in medians.items()},
    "feature_cols": feature_cols,
    "cat_features": cat_features,
    "reference_date": str(reference_date),
    "best_method": best_method,
    "best_mae": float(best_mae),
    "weights": weights,
    "use_log_transform": True,
}
with open(os.path.join(OUTPUT_DIR, "ensemble_info.json"), "w") as f:
    json.dump(preprocess_info, f, indent=2, ensure_ascii=False)

# 保存编码信息
encodings_info = {
    "te_encodings": {k: v.to_dict() for k, v in te_encodings.items()},
}
with open(os.path.join(OUTPUT_DIR, "ensemble_encodings.pkl"), "wb") as f:
    import pickle
    pickle.dump({"te_encodings": te_encodings, "stat_encodings": stat_encodings}, f)

print("  所有模型和预处理信息已保存")


# ============================================================
# 第二十部分：测试集预测
# ============================================================
print("\n【17. 测试集预测】")

# 对测试集进行相同的特征处理
test_df_final = test_df.copy()

# 填充缺失值
for col, med_val in medians.items():
    if col in test_df_final.columns and test_df_final[col].isnull().any():
        test_df_final[col] = test_df_final[col].fillna(med_val)

# 为 CatBoost 准备特征矩阵（字符串类型）
X_test_catboost = test_df_final[feature_cols].copy()
for col in cat_features:
    if col in X_test_catboost.columns:
        X_test_catboost[col] = X_test_catboost[col].astype(str)

# 为 LightGBM/XGBoost 准备特征矩阵（数值类型）
X_test_other = test_df_final[feature_cols].copy()
for col in cat_features:
    if col in X_test_other.columns and col in label_encoders:
        # 使用保存的编码器进行转换（用字典映射加速）
        le = label_encoders[col]
        mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
        X_test_other[col] = X_test_other[col].astype(str).map(mapping).fillna(-1).astype(int)

print(f"  测试集特征矩阵准备完成")

# 使用最优方法预测
if best_method == "Stacking":
    pred_cb = np.expm1(catboost_model.predict(X_test_catboost))
    pred_lgb = np.expm1(lightgbm_model.predict(X_test_other))
    pred_xgb = np.expm1(xgboost_model.predict(X_test_other))
    stack_X_test = np.column_stack([pred_cb, pred_lgb, pred_xgb])
    predictions = ridge_model.predict(stack_X_test)
elif best_method == "加权平均":
    pred_cb = np.expm1(catboost_model.predict(X_test_catboost))
    pred_lgb = np.expm1(lightgbm_model.predict(X_test_other))
    pred_xgb = np.expm1(xgboost_model.predict(X_test_other))
    predictions = weights[0] * pred_cb + weights[1] * pred_lgb + weights[2] * pred_xgb
elif best_method == "CatBoost":
    predictions = np.expm1(catboost_model.predict(X_test_catboost))
elif best_method == "LightGBM":
    predictions = np.expm1(lightgbm_model.predict(X_test_other))
else:
    predictions = np.expm1(xgboost_model.predict(X_test_other))

predictions = np.maximum(predictions, 0)

# 生成提交文件
submit = pd.DataFrame({
    "SaleID": test_df_raw["SaleID"],
    "price": predictions,
})

output_file = os.path.join(SCRIPT_DIR, "used_car_submit_ensemble_v2.csv")
submit.to_csv(output_file, index=False)

print(f"  预测完成，共 {len(predictions)} 条")
print(f"  提交文件: {output_file}")
print(f"  价格统计: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")


# ============================================================
# 第二十一部分：总结报告
# ============================================================
print("\n" + "=" * 70)
print("【Stacking 集成优化 - 总结报告】")
print("=" * 70)
print(f"""
优化策略:
  1. 五折交叉目标编码（防止数据泄露）
  2. 统计聚合特征（brand/model/regionCode 的 price 统计量）
  3. v_ 特征交互（v_0*v_3, v_0+v_12 等）
  4. log1p(price) 目标变换（改善高价区间）
  5. 三模型 Stacking 集成（CatBoost + LightGBM + XGBoost）

特征工程:
  - 基础特征: {len(feature_cols)} 个
  - 统计特征: {len(cat_cols_for_stat) * 6} 个
  - v_交互特征: 多个（乘法/平方/统计聚合）

模型性能（验证集）:
  CatBoost MAE:  {mae_cb:.2f}
  LightGBM MAE:  {mae_lgb:.2f}
  XGBoost MAE:   {mae_xgb:.2f}
  加权平均 MAE:  {mae_weighted:.2f}
  Stacking MAE:  {mae_stacked:.2f}

最优方案:
  方法: {best_method}
  MAE:  {best_mae:.2f}
  目标: < 400 ({"已达成!" if best_mae < 400 else "未达成，继续优化中..."})

输出文件:
  模型文件: catboost_v2_model.cbm, lightgbm_v2_model.txt, xgboost_v2_model.json
  Stacking 元学习器: ridge_meta_model.pkl
  提交文件: used_car_submit_ensemble_v2.csv
  可视化图表: 05_ensemble_*.png, 06_best_model_*.png
""")

print("=" * 70)
print("【Stacking 集成优化 - 完成】")
print("=" * 70)
