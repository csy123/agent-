#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手车价格预测 - XGBoost 模型预测
使用训练好的模型对测试集进行预测，生成提交文件
"""
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============ 路径配置 ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(SCRIPT_DIR, "used_car_train_20200313.csv")
TEST_FILE = os.path.join(SCRIPT_DIR, "used_car_testB_20200421.csv")
SUBMIT_SAMPLE = os.path.join(SCRIPT_DIR, "used_car_sample_submit.csv")
MODEL_FILE = os.path.join(SCRIPT_DIR, "model_output", "xgboost_model.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "used_car_submit.csv")

print("=" * 70)
print("【二手车价格预测 - 模型预测】")
print("=" * 70)

# ============ 1. 加载数据 ============
print("\n【1. 加载数据】")
print("  加载训练数据（用于特征工程参考）...")
train_df = pd.read_csv(TRAIN_FILE, sep=" ")
print(f"  训练集: {train_df.shape[0]} 行")

print("  加载测试数据...")
test_df = pd.read_csv(TEST_FILE, sep=" ")
print(f"  测试集: {test_df.shape[0]} 行")
print(f"  测试集 SaleID 范围: {test_df['SaleID'].min()} ~ {test_df['SaleID'].max()}")

# ============ 1.1 分类特征唯一值统计 ============
print("\n【1.1 分类特征唯一值统计】")

cat_cols = ["brand", "bodyType", "fuelType", "gearbox", "model", "regionCode", 
            "notRepairedDamage", "seller", "offerType"]

print(f"\n{'特征名':<25s} {'唯一值个数':>10s} {'唯一值示例':<50s}")
print("-" * 90)

for col in cat_cols:
    # 合并训练集和测试集的唯一值（确保预测时覆盖完整）
    train_unique = set(train_df[col].dropna().unique())
    test_unique = set(test_df[col].dropna().unique())
    all_unique = train_unique | test_unique
    unique_count = len(all_unique)
    
    # 显示部分唯一值（转为原生类型以简化显示）
    sorted_values = sorted(all_unique, key=str)
    if len(sorted_values) > 20:
        values_str = str([str(v) for v in sorted_values[:15]]) + f" ... 共{len(sorted_values)}个"
    else:
        values_str = str([str(v) for v in sorted_values])
    
    print(f"{col:<25s} {unique_count:>10d} {values_str}")

# 详细分布
print("\n【详细分布统计】")
for col in cat_cols:
    print(f"\n--- {col} (共 {len(set(train_df[col].dropna().unique()) | set(test_df[col].dropna().unique()))} 种) ---")
    value_counts = train_df[col].value_counts()
    if len(value_counts) <= 15:
        for val, count in value_counts.items():
            pct = count / len(train_df) * 100
            print(f"  {str(val):<15s}: {count:>8d} ({pct:>5.2f}%)")
    else:
        # 只显示前10个
        for val, count in value_counts.head(10).items():
            pct = count / len(train_df) * 100
            print(f"  {str(val):<15s}: {count:>8d} ({pct:>5.2f}%)")
        print(f"  ... 还有 {len(value_counts) - 10} 种类别")

# 检查测试集中是否有训练集中未出现的类别
print("\n【异常检查 - 测试集中的未知类别】")
unknown_cols = []
for col in cat_cols:
    train_values = set(train_df[col].dropna().unique())
    test_values = set(test_df[col].dropna().unique())
    unknown = test_values - train_values
    if unknown:
        unknown_cols.append((col, len(unknown), sorted(list(unknown))[:10]))
        print(f"  ⚠️ {col}: 测试集中有 {len(unknown)} 个未知类别: {sorted(list(unknown))[:10]}")

if not unknown_cols:
    print("  ✅ 测试集所有类别在训练集中均存在")
else:
    print(f"\n  共 {len(unknown_cols)} 个特征存在未知类别，目标编码时将用全局均值填充")

# ============ 2. 数据预处理（与训练时保持一致）============
print("\n【2. 数据预处理】")

def preprocess_data(df, train_df=None, is_train=False):
    """数据预处理函数，训练和预测共用"""
    df = df.copy()
    
    # 2.1 处理 notRepairedDamage 字段
    df["notRepairedDamage"] = df["notRepairedDamage"].replace({"-": -1, "0.0": 0, "1.0": 1}).astype(float)
    print("  notRepairedDamage: 处理完成")
    
    # 2.2 处理日期特征
    def process_date_feature(df, col, prefix):
        df[col] = df[col].astype(int).astype(str).str.zfill(8)
        df[f"{prefix}_year"] = df[col].str[:4].astype(int)
        df[f"{prefix}_month"] = df[col].str[4:6].astype(int)
        df[f"{prefix}_day"] = df[col].str[6:8].astype(int)
        df[f"{prefix}_date"] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
        return df
    
    df = process_date_feature(df, "regDate", "reg")
    df = process_date_feature(df, "creatDate", "create")
    print("  日期特征提取完成")
    
    # 2.3 计算衍生特征
    # 车龄（上架时）
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
    
    # 车龄（截至参考日期）
    df["car_age_now"] = (reference_date - df["reg_date"]).dt.days / 365.25
    df["car_age_now"] = df["car_age_now"].clip(0, 100)
    
    # 上架时长
    df["listing_days"] = (reference_date - df["create_date"]).dt.days.clip(0)
    
    # 是否新车
    df["is_new_car"] = (df["car_age"] <= 1).astype(int)
    print("  衍生特征计算完成")
    
    # 2.4 删除中间日期列
    df = df.drop(columns=["regDate", "creatDate", "reg_date", "create_date"], errors="ignore")
    
    return df

# 预处理训练集和测试集
train_df_processed = preprocess_data(train_df, train_df=None, is_train=True)
test_df_processed = preprocess_data(test_df, train_df=train_df, is_train=False)

# ============ 3. 缺失值处理 ============
print("\n【3. 缺失值处理】")

# 获取训练集中位数（排除目标变量）
numeric_cols = train_df_processed.select_dtypes(include=[np.number]).columns
# 确保 price 不在 numeric_cols 中用于测试集
test_numeric_cols = [c for c in numeric_cols if c != "price"]
medians = train_df_processed[test_numeric_cols].median()

# 用训练集的中位数填充训练集
for col in test_numeric_cols:
    if train_df_processed[col].isnull().any():
        train_df_processed[col] = train_df_processed[col].fillna(medians[col])

# 用训练集的中位数填充测试集
for col in test_numeric_cols:
    if test_df_processed[col].isnull().any():
        test_df_processed[col] = test_df_processed[col].fillna(medians[col])

print(f"  使用训练集中位数填充缺失值")

# ============ 4. 特征编码（目标编码）============
print("\n【4. 特征编码】")

cat_cols = ["brand", "bodyType", "fuelType", "gearbox", "model", "regionCode"]

# 使用训练集计算目标编码
def target_encode_train(train_df, cat_cols, target):
    """从训练集计算目标编码映射"""
    encodings = {}
    global_mean = train_df[target].mean()
    for col in cat_cols:
        encodings[col] = train_df.groupby(col)[target].mean()
    return encodings, global_mean

encodings, global_mean = target_encode_train(train_df_processed, cat_cols, "price")

# 应用编码到训练集和测试集
for col in cat_cols:
    train_df_processed[col] = train_df_processed[col].map(encodings[col]).fillna(global_mean)
    test_df_processed[col] = test_df_processed[col].map(encodings[col]).fillna(global_mean)

print("  目标编码完成")

# ============ 5. 准备特征矩阵 ============
print("\n【5. 准备特征矩阵】")

# 定义特征列（与训练时一致）
exclude_cols = ["SaleID", "name", "price", "notRepairedDamage"]
feature_cols = [col for col in train_df_processed.columns if col not in exclude_cols]

# 准备测试集特征
X_test = test_df_processed[feature_cols].copy()
print(f"  测试特征矩阵: {X_test.shape[0]} × {X_test.shape[1]}")
print(f"  特征列表: {feature_cols}")

# ============ 6. 加载模型并预测 ============
print("\n【6. 加载模型并预测】")

from xgboost import XGBRegressor

# 加载模型
model = XGBRegressor()
model.load_model(MODEL_FILE)
print(f"  模型加载成功: {MODEL_FILE}")

# 预测
print("  开始预测...")
predictions = model.predict(X_test)
print(f"  预测完成，共 {len(predictions)} 条")

# 预测值后处理：确保价格为正数
predictions = np.maximum(predictions, 0)
print(f"  负值修正完成（min={predictions.min():.2f}, max={predictions.max():.2f}）")

# ============ 7. 生成提交文件 ============
print("\n【7. 生成提交文件】")

# 使用测试集的 SaleID 和预测结果生成提交文件
submit = pd.DataFrame({
    "SaleID": test_df["SaleID"],
    "price": predictions
})
print(f"  提交文件格式: SaleID, price")
print(f"  SaleID 范围: {submit['SaleID'].min()} ~ {submit['SaleID'].max()}")
print(f"  共 {len(submit)} 行")

# 保存结果
submit.to_csv(OUTPUT_FILE, index=False)
print(f"  提交文件已保存: {OUTPUT_FILE}")

# ============ 8. 预测结果统计 ============
print("\n【8. 预测结果统计】")
print(f"""
  预测价格统计:
    - 最小值:   {predictions.min():,.2f}
    - 最大值:   {predictions.max():,.2f}
    - 均值:     {predictions.mean():,.2f}
    - 中位数:   {np.median(predictions):,.2f}
    - 标准差:   {predictions.std():,.2f}
""")

# ============ 9. 预览结果 ============
print("\n【9. 提交文件预览】")
print(f"{'SaleID':<10s} {'price':>12s}")
print("-" * 25)
for i in range(min(10, len(submit))):
    print(f"{submit['SaleID'].iloc[i]:<10d} {submit['price'].iloc[i]:>12.2f}")
if len(submit) > 10:
    print(f"{'...':<10s} {'...':>12s}")
    print(f"{submit['SaleID'].iloc[-1]:<10d} {submit['price'].iloc[-1]:>12.2f}")

print("\n" + "=" * 70)
print("【预测完成】")
print("=" * 70)
print(f"\n提交文件: {OUTPUT_FILE}")
print(f"文件大小: {os.path.getsize(OUTPUT_FILE) / 1024:.2f} KB")
