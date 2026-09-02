#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手车价格预测 - EDA（探索性数据分析）
包含统计分析 + 可视化图表
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ============ 中文字体配置 ============
# 强制设置 matplotlib 字体缓存和中文字体
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['savefig.dpi'] = 150
matplotlib.rcParams['figure.dpi'] = 100
sns.set_style("whitegrid")

# ============ 路径配置 ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "used_car_train_20200313.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "eda_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ 读取数据 ============
print("=" * 70)
print("【EDA 开始】")
print("=" * 70)
df = pd.read_csv(DATA_FILE, sep=" ")
print(f"数据加载完成: {df.shape[0]} 行 × {df.shape[1]} 列")

# ============ 1. 数据概览 ============
print("\n" + "=" * 70)
print("【1. 数据概览】")
print("=" * 70)
print(f"\n数据形状: {df.shape}")
print(f"内存占用: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
print(f"\n数据类型:")
print(df.dtypes.value_counts())

# ============ 2. 缺失值分析 ============
print("\n" + "=" * 70)
print("【2. 缺失值分析】")
print("=" * 70)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"缺失数": missing, "缺失率%": missing_pct})
missing_df = missing_df[missing_df["缺失数"] > 0].sort_values("缺失率%", ascending=False)
print("\n存在缺失值的字段:")
print(missing_df.to_string())

# 缺失值可视化
fig, ax = plt.subplots(figsize=(10, 4))
all_missing = missing_df.copy()
if len(all_missing) > 0:
    bars = ax.bar(all_missing.index, all_missing["缺失率%"], color="#ff6b6b", alpha=0.8)
    ax.set_ylabel("缺失率 (%)")
    ax.set_title("各字段缺失率")
    ax.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars, all_missing["缺失率%"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_missing_values.png"), dpi=150)
plt.close()

# ============ 3. 目标变量 price 分析 ============
print("\n" + "=" * 70)
print("【3. 目标变量 price 分析】")
print("=" * 70)
price = df["price"]
print(f"\n基本统计:")
print(f"  数量:     {len(price):,}")
print(f"  均值:     {price.mean():,.2f}")
print(f"  中位数:   {price.median():,.2f}")
print(f"  标准差:   {price.std():,.2f}")
print(f"  最小值:   {price.min():,.2f}")
print(f"  最大值:   {price.max():,.2f}")
print(f"  偏度:     {price.skew():.4f}")
print(f"  峰度:     {price.kurt():.4f}")

# 分位数
percentiles = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
print(f"\n分位数:")
for p in percentiles:
    print(f"  {p*100:.0f}%: {price.quantile(p):,.2f}")

# price 分布图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3.1 直方图
axes[0, 0].hist(price, bins=50, color="#00b4d8", alpha=0.7, edgecolor="white")
axes[0, 0].axvline(price.mean(), color="#ff6b6b", linestyle="--", label=f"均值: {price.mean():.0f}")
axes[0, 0].axvline(price.median(), color="#6bcf7f", linestyle="--", label=f"中位数: {price.median():.0f}")
axes[0, 0].set_xlabel("价格")
axes[0, 0].set_ylabel("频数")
axes[0, 0].set_title("price 分布直方图")
axes[0, 0].legend()

# 3.2 对数变换后直方图（处理右偏）
log_price = np.log1p(price)
axes[0, 1].hist(log_price, bins=50, color="#6bcf7f", alpha=0.7, edgecolor="white")
axes[0, 1].set_xlabel("log(price + 1)")
axes[0, 1].set_ylabel("频数")
axes[0, 1].set_title("log(price+1) 分布")

# 3.3 箱线图
axes[1, 0].boxplot(price, vert=True, patch_artist=True,
                   boxprops=dict(facecolor="lightblue"))
axes[1, 0].set_ylabel("价格")
axes[1, 0].set_title("price 箱线图")

# 3.4 Q-Q图
stats.probplot(price, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("price Q-Q图")
axes[1, 1].get_lines()[0].set_markerfacecolor("#ff6b6b")
axes[1, 1].get_lines()[1].set_color("#00b4d8")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_price_distribution.png"), dpi=150)
plt.close()

# ============ 4. 数值特征分析 ============
print("\n" + "=" * 70)
print("【4. 数值特征分析】")
print("=" * 70)

# 数值特征统计
num_cols = ["power", "kilometer", "price"] + [f"v_{i}" for i in range(15)]
num_stats = df[num_cols].describe().T
num_stats["偏度"] = df[num_cols].skew()
num_stats["峰度"] = df[num_cols].kurt()
num_stats["缺失数"] = df[num_cols].isnull().sum()
num_stats["缺失率%"] = (num_stats["缺失数"] / len(df) * 100).round(2)
print("\n数值特征统计表:")
print(num_stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "偏度", "峰度", "缺失率%"]].to_string())

# 数值特征分布直方图
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    ax = axes[i]
    data = df[col].dropna()
    ax.hist(data, bins=30, color="#00b4d8", alpha=0.7, edgecolor="white")
    ax.set_title(col, fontsize=11)
    ax.tick_params(labelsize=8)
plt.suptitle("数值特征分布", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_numeric_distributions.png"), dpi=150)
plt.close()

# ============ 5. 类别特征分析 ============
print("\n" + "=" * 70)
print("【5. 类别特征分析】")
print("=" * 70)

cat_cols = ["brand", "bodyType", "fuelType", "gearbox", "notRepairedDamage", "regionCode", "model"]
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ax = axes[i]
    value_counts = df[col].value_counts()
    if len(value_counts) > 20:
        # 类别太多时，只显示前20个
        top_categories = value_counts.head(19)
        other = value_counts.iloc[19:].sum()
        top_categories["其他"] = other
        value_counts = top_categories
    ax.barh(range(len(value_counts)), value_counts.values, color="#6bcf7f", alpha=0.8)
    ax.set_yticks(range(len(value_counts)))
    ax.set_yticklabels(value_counts.index, fontsize=9)
    ax.set_xlabel("数量")
    ax.set_title(f"{col} 分布 (共{df[col].nunique()}类)")
    ax.tick_params(labelsize=8)

plt.suptitle("类别特征分布", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_categorical_distributions.png"), dpi=150)
plt.close()

# ============ 6. 相关性分析 ============
print("\n" + "=" * 70)
print("【6. 相关性分析】")
print("=" * 70)

# 预处理: 将 notRepairedDamage 转为数值（0: 无损伤, 1: 有损伤, -1: 未知"-"）
df_corr = df.copy()
df_corr["notRepairedDamage"] = df_corr["notRepairedDamage"].replace({"-": -1, "0.0": 0, "1.0": 1}).astype(float)

# 计算与 price 的相关系数（仅数值特征）
feature_cols = [c for c in df_corr.columns if c != "price" and c != "SaleID"]
# 确保所有特征都是数值型
numeric_cols = df_corr[feature_cols + ["price"]].select_dtypes(include=[np.number]).columns.tolist()
correlations = df_corr[numeric_cols].corr()["price"].drop("price").dropna().sort_values(key=abs, ascending=False)
print("\n与 price 的相关性（按绝对值排序）:")
for col, corr in correlations.items():
    bar = "█" * int(min(abs(corr) * 50, 50))
    sign = "+" if corr > 0 else ""
    print(f"  {col:<20s}: {sign}{corr:.4f} {bar}")

# 热力图
fig, ax = plt.subplots(figsize=(16, 14))
corr_matrix = df_corr[numeric_cols].corr().dropna(how="all")
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8})
ax.set_title("特征相关性热力图")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_correlation_heatmap.png"), dpi=150)
plt.close()

# Top 10 特征与 price 的散点图
top10_features = correlations.head(10).index.tolist()
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, col in enumerate(top10_features):
    ax = axes[i]
    ax.scatter(df_corr[col], df_corr["price"], alpha=0.3, s=5, color="#00b4d8")
    ax.set_xlabel(col)
    ax.set_ylabel("price")
    ax.set_title(f"{col} vs price\n(corr={correlations[col]:.3f})", fontsize=10)
plt.suptitle("Top 10 特征与 price 的散点关系", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_top10_scatter.png"), dpi=150)
plt.close()

# ============ 7. 类别特征与 price 的关系 ============
print("\n" + "=" * 70)
print("【7. 类别特征与 price 的关系】")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
plot_cols = ["brand", "bodyType", "fuelType", "gearbox", "notRepairedDamage", "regionCode"]

for i, col in enumerate(plot_cols):
    ax = axes[i]
    # 计算每个类别的 price 统计
    grouped = df.groupby(col)["price"].agg(["mean", "median", "count"]).sort_values("mean", ascending=False)
    if len(grouped) > 15:
        grouped = grouped.head(15)
    x = range(len(grouped))
    ax.bar(x, grouped["mean"], color="#00b4d8", alpha=0.7, label="均值")
    ax.bar(x, grouped["median"], color="#6bcf7f", alpha=0.5, label="中位数")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=45, fontsize=9)
    ax.set_ylabel("price")
    ax.set_title(f"{col} vs price")
    ax.legend(fontsize=8)

plt.suptitle("类别特征与 price 的关系", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_categorical_vs_price.png"), dpi=150)
plt.close()

# ============ 8. 异常值检测 ============
print("\n" + "=" * 70)
print("【8. 异常值检测（IQR 法）】")
print("=" * 70)

outlier_report = []
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_pct = (outliers / len(df) * 100).round(2)
    outlier_report.append({
        "特征": col,
        "Q1": round(Q1, 2),
        "Q3": round(Q3, 2),
        "IQR": round(IQR, 2),
        "下限": round(lower, 2),
        "上限": round(upper, 2),
        "异常值数": outliers,
        "异常值率%": outlier_pct
    })

outlier_df = pd.DataFrame(outlier_report)
print("\n异常值检测报告:")
print(outlier_df.to_string(index=False))

# 异常值可视化
fig, ax = plt.subplots(figsize=(12, 6))
outlier_df_sorted = outlier_df.sort_values("异常值率%", ascending=True)
bars = ax.barh(outlier_df_sorted["特征"], outlier_df_sorted["异常值率%"],
               color="#ff6b6b", alpha=0.7)
ax.set_xlabel("异常值率 (%)")
ax.set_title("各特征的异常值比例")
for bar, val in zip(bars, outlier_df_sorted["异常值率%"]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_outlier_detection.png"), dpi=150)
plt.close()

# ============ 9. 特征工程建议 ============
print("\n" + "=" * 70)
print("【9. 特征工程建议】")
print("=" * 70)
print("""
基于 EDA 分析，建议进行以下特征工程:

1. 缺失值处理:
   - model (0.00%), bodyType (3.00%), gearbox (3.99%), fuelType (5.79%)
   - 建议: 用众数填充或增加"缺失"类别

2. 异常值处理:
   - price 存在明显右偏 (偏度=3.35), 建议对数变换
   - power 可能存在异常高值

3. 特征变换:
   - price: 使用 log1p 变换 (减小偏态)
   - notRepairedDamage: 将 "-" 值处理为 0 或新增类别
   - regDate: 可转为"车龄"特征 (2024 - 注册年份)
   - creatDate: 可转为"上架时长"特征

4. 类别特征编码:
   - brand, model, bodyType, fuelType, gearbox, regionCode
   - 建议: 目标编码或频率编码

5. 高价值特征:
   - power (功率): 与价格正相关
   - brand (品牌): 不同品牌价格差异大
   - v_0, v_1, v_11 等匿名特征
""")

# ============ 完成 ============
print("\n" + "=" * 70)
print("【EDA 完成】")
print("=" * 70)
print(f"\n所有图表已保存至: {OUTPUT_DIR}/")
print("\n生成的文件:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  - {f}")
