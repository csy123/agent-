#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手车价格预测 - 数据探索
读取 used_car_train_20200313.csv，查看字段及前5行数据
"""
import os
import pandas as pd

# ============ 1. 读取数据 ============
# 使用脚本所在目录作为基准，避免相对路径问题
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "used_car_train_20200313.csv")
print(f"数据文件路径: {DATA_FILE}")
df = pd.read_csv(DATA_FILE, sep=" ")

# ============ 2. 查看基本信息 ============
print("=" * 70)
print("【数据基本信息】")
print("=" * 70)
print(f"数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"文件大小: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

# ============ 3. 查看字段说明 ============
print("\n" + "=" * 70)
print("【字段名与类型】")
print("=" * 70)

# 区分特征类型
business_cols = [
    "SaleID", "name", "regDate", "model", "brand",
    "bodyType", "fuelType", "gearbox", "power", "kilometer",
    "notRepairedDamage", "regionCode", "seller", "offerType",
    "creatDate", "price"
]
anonymous_cols = [f"v_{i}" for i in range(15)]

print("\n--- 业务特征 ---")
for col in business_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        dtype = df[col].dtype
        print(f"  {col:<20s} ({str(dtype):>10s})  非空值: {non_null:>8d} / {len(df)}")

print("\n--- 匿名特征（v_0 ~ v_14）---")
for col in anonymous_cols:
    if col in df.columns:
        print(f"  {col:<20s} ({str(df[col].dtype):>10s})  范围: [{df[col].min():.2f}, {df[col].max():.2f}]")

# ============ 4. 前5行数据 ============
print("\n" + "=" * 70)
print("【前5行数据】")
print("=" * 70)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)
pd.set_option("display.max_colwidth", 20)
print(df.head(5).to_string())

# ============ 5. 目标变量 price 统计 ============
print("\n" + "=" * 70)
print("【目标变量 price 统计】")
print("=" * 70)
print(f"  最小值:   {df['price'].min():,}")
print(f"  最大值:   {df['price'].max():,}")
print(f"  均值:     {df['price'].mean():,.2f}")
print(f"  中位数:   {df['price'].median():,.2f}")
print(f"  标准差:   {df['price'].std():,.2f}")
print(f"  偏度:     {df['price'].skew():.4f}")
print(f"  峰度:     {df['price'].kurt():.4f}")

# ============ 6. 缺失值检查 ============
print("\n" + "=" * 70)
print("【缺失值统计】")
print("=" * 70)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"缺失数": missing, "缺失率%": missing_pct})
missing_df = missing_df[missing_df["缺失数"] > 0]
if len(missing_df) > 0:
    print(missing_df.to_string())
else:
    print("  无缺失值")

# 特殊值检查
print("\n--- 特殊值检查（notRepairedDamage 列）---")
print(f"  值分布: {df['notRepairedDamage'].value_counts().to_dict()}")
