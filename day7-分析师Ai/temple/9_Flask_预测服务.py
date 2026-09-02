#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手车价格预测 - Flask 预测服务
功能：
  1. 加载训练好的 CatBoost 增强版模型
  2. 提供 REST API：POST /api/predict
  3. 提供 Web 前端页面：GET /
  4. 复用 7_CatBoost增强特征工程版.py 的特征工程逻辑
启动：python 9_Flask_预测服务.py
访问：http://127.0.0.1:5000
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# ============ 路径配置 ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model_output", "catboost_enhanced_model.cbm")
INFO_PATH = os.path.join(SCRIPT_DIR, "model_output", "catboost_enhanced_info.json")
TEMPLATES_DIR = os.path.join(SCRIPT_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATES_DIR)

# ============ 加载模型和预处理信息 ============
print("=" * 70)
print("【二手车价格预测 - Flask 预测服务】")
print("=" * 70)

print("\n【1. 加载模型和预处理信息】")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}，请先运行 7_CatBoost增强特征工程版.py")

model = CatBoostRegressor()
model.load_model(MODEL_PATH)
print(f"  模型加载完成: {MODEL_PATH}")

with open(INFO_PATH, "r", encoding="utf-8") as f:
    preprocess_info = json.load(f)

medians = {k: float(v) for k, v in preprocess_info["medians"].items()}
feature_cols = preprocess_info["feature_cols"]
cat_features = preprocess_info["cat_features"]
reference_date = pd.to_datetime(preprocess_info["reference_date"])
test_mae = preprocess_info.get("test_mae", 0)

print(f"  特征数量: {len(feature_cols)}")
print(f"  类别特征: {len(cat_features)} 个")
print(f"  参考日期: {reference_date.strftime('%Y-%m-%d')}")
print(f"  验证集 MAE: {test_mae:.2f}")

# ============ 特征工程函数（与训练保持一致）============
def advanced_feature_engineering(input_data):
    """对用户输入的车辆信息进行特征工程"""
    df = pd.DataFrame([input_data])

    # 1. notRepairedDamage
    df["notRepairedDamage"] = df["notRepairedDamage"].replace({"-": -1, "0.0": 0, "1.0": 1}).astype(float) \
        if "notRepairedDamage" in df.columns else -1

    # 2. 日期特征工程
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

    # 日期衍生特征
    df["car_age"] = (df["create_date"] - df["reg_date"]).dt.days / 365.25
    df["car_age"] = df["car_age"].clip(0, 100)
    df["car_age_months"] = df["car_age"] * 12
    df["car_age_now"] = (reference_date - df["reg_date"]).dt.days / 365.25
    df["car_age_now"] = df["car_age_now"].clip(0, 100)
    df["listing_days"] = (reference_date - df["create_date"]).dt.days.clip(0)
    df["reg_quarter"] = df["reg_month"].apply(lambda x: (x - 1) // 3 + 1)
    df["create_quarter"] = df["create_month"].apply(lambda x: (x - 1) // 3 + 1)
    df["reg_decade"] = (df["reg_year"] // 10) * 10
    df["reg_is_early_year"] = (df["reg_month"] <= 3).astype(int)
    df["reg_is_late_year"] = (df["reg_month"] >= 10).astype(int)

    # 3. Power 特征
    df["power_clipped"] = df["power"].clip(0, 600)
    df["power_log"] = np.log1p(df["power_clipped"])
    power_bins = [0, 30, 60, 90, 120, 150, 200, 300, 9999]
    power_labels = ["0-30", "30-60", "60-90", "90-120", "120-150", "150-200", "200-300", "300+"]
    df["power_bin"] = pd.cut(df["power_clipped"], bins=power_bins, labels=power_labels, right=False)
    df["is_high_power"] = (df["power_clipped"] >= 150).astype(int)

    # 4. 里程特征
    df["usage_intensity"] = df["kilometer"] / (df["car_age"] + 0.5)
    km_bins = [0, 3, 6, 9, 12, 15]
    km_labels = ["0-3万", "3-6万", "6-9万", "9-12万", "12-15万"]
    df["kilometer_bin"] = pd.cut(df["kilometer"], bins=km_bins, labels=km_labels, right=False)

    # 5. 衍生数值特征
    df["power_per_year"] = df["power_clipped"] / (df["car_age"] + 0.5)
    df["power_km_ratio"] = df["power_clipped"] / (df["kilometer"] + 0.5)
    df["car_age_sq"] = df["car_age"] ** 2
    df["is_new_car"] = (df["car_age"] <= 1).astype(int)
    df["is_almost_new"] = ((df["car_age"] > 1) & (df["car_age"] <= 3)).astype(int)
    df["is_old_car"] = (df["car_age"] > 8).astype(int)

    # 6. 交互类别特征
    df["brand_model"] = df["brand"].astype(str) + "_" + df["model"].astype(str)
    df["brand_bodyType"] = df["brand"].astype(str) + "_" + df["bodyType"].astype(str)
    df["fuel_gear"] = df["fuelType"].astype(str) + "_" + df["gearbox"].astype(str)
    df["brand_fuel"] = df["brand"].astype(str) + "_" + df["fuelType"].astype(str)
    df["age_group"] = pd.cut(df["car_age"], bins=[0, 3, 6, 10, 100], labels=["0-3y", "3-6y", "6-10y", "10y+"])
    df["brand_age"] = df["brand"].astype(str) + "_" + df["age_group"].astype(str)
    df["body_fuel"] = df["bodyType"].astype(str) + "_" + df["fuelType"].astype(str)
    df["region_province"] = (df["regionCode"] // 100).astype(str)

    # 7. 删除中间列
    cols_to_drop = ["regDate", "creatDate", "regDate_str", "creatDate_str", "reg_date", "create_date"]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # 8. 类别特征转字符串
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # 9. 缺失值填充
    for col, med_val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(med_val)

    # 补齐缺失的特征列
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].copy()
    return X

# ============ 字段说明 ============
FIELD_INFO = {
    "brand": {"name": "品牌", "type": "number", "default": 0, "min": 0, "max": 39, "desc": "0-39，参考训练集 brand 字段"},
    "model": {"name": "车型编码", "type": "number", "default": 30, "min": 0, "max": 247},
    "bodyType": {"name": "车身类型", "type": "select", "default": 1,
                 "options": [["0", "豪华轿车"], ["1", "微型车"], ["2", "厢型车"],
                             ["3", "大巴车"], ["4", "跑车"], ["5", "皮卡"],
                             ["6", "微型客车"], ["7", "其他"]]},
    "fuelType": {"name": "燃油类型", "type": "select", "default": 0,
                 "options": [["0", "汽油"], ["1", "柴油"], ["2", "液化石油气"],
                             ["3", "天然气"], ["4", "混合动力"], ["5", "电动"], ["6", "其他"]]},
    "gearbox": {"name": "变速箱", "type": "select", "default": 0,
                "options": [["0", "手动"], ["1", "自动"]]},
    "power": {"name": "发动机功率(W)", "type": "number", "default": 110, "min": 0, "max": 600},
    "kilometer": {"name": "行驶里程(万km)", "type": "number", "default": 12, "min": 0, "max": 15, "step": 0.5},
    "regionCode": {"name": "地区编码", "type": "number", "default": 1000, "min": 0, "max": 9999},
    "regDate": {"name": "注册日期", "type": "date", "default": "2010-01-01"},
    "creatDate": {"name": "上架日期", "type": "date", "default": "2016-03-01"},
}

# ============ 路由：首页 ============
@app.route("/")
def index():
    return render_template("index.html", field_info=FIELD_INFO, test_mae=test_mae)

# ============ 路由：预测 API ============
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        print(f"\n[预测请求] {data}")

        # 日期转 YYYYMMDD 格式
        if "regDate" in data and "-" in str(data["regDate"]):
            data["regDate"] = int(str(data["regDate"]).replace("-", ""))
        if "creatDate" in data and "-" in str(data["creatDate"]):
            data["creatDate"] = int(str(data["creatDate"]).replace("-", ""))

        # notRepairedDamage 默认值
        if "notRepairedDamage" not in data:
            data["notRepairedDamage"] = "0.0"

        # 类型转换
        int_fields = ["brand", "model", "bodyType", "fuelType", "gearbox", "regionCode"]
        for f in int_fields:
            if f in data:
                data[f] = int(float(data[f]))
        for f in ["power", "kilometer"]:
            if f in data:
                data[f] = float(data[f])

        # 特征工程
        X = advanced_feature_engineering(data)
        print(f"  特征工程完成，形状: {X.shape}")

        # 模型预测
        prediction = float(model.predict(X)[0])
        prediction = max(0, prediction)
        print(f"  预测价格: {prediction:.2f}")

        # 置信区间
        low = max(0, prediction - test_mae)
        high = prediction + test_mae

        return jsonify({
            "success": True,
            "price": round(prediction, 2),
            "price_low": round(low, 2),
            "price_high": round(high, 2),
            "mae": round(test_mae, 2),
            "features_used": len(feature_cols),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 400

# ============ 路由：字段信息 ============
@app.route("/api/fields")
def fields():
    return jsonify({"fields": FIELD_INFO, "feature_count": len(feature_cols)})

# ============ 路由：健康检查 ============
@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "CatBoost Enhanced",
        "features": len(feature_cols),
        "test_mae": test_mae
    })

# ============ 主程序入口 ============
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("【2. 启动 Flask 服务】")
    print("=" * 70)
    PORT = 5001
    print(f"  访问地址: http://127.0.0.1:{PORT}")
    print(f"  API 文档: http://127.0.0.1:{PORT}/api/health")
    print(f"  预测接口: POST http://127.0.0.1:{PORT}/api/predict")
    print("=" * 70)
    app.run(host="0.0.0.0", port=PORT, debug=False)