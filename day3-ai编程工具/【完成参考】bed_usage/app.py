#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医院床位使用率可视化大屏 - Flask 后端
提供数据处理与 API 接口
"""
import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template

app = Flask(__name__)

# 数据文件路径
DATA_FILE = os.path.join(os.path.dirname(__file__), "hospital_bed_usage_data.xlsx")


def load_data():
    """加载数据"""
    df = pd.read_excel(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_latest(df):
    """获取最新时间点数据"""
    latest_ts = df["timestamp"].max()
    return df[df["timestamp"] == latest_ts].copy(), latest_ts


# ============ 数据处理函数 ============

def process_summary(df_latest):
    """KPI 指标卡数据"""
    total = int(df_latest["total_beds"].sum())
    occupied = int(df_latest["occupied_beds"].sum())
    available = int(df_latest["available_beds"].sum())
    full_count = int((df_latest["special_status"] == "满床").sum())
    rate = round(occupied / total * 100, 2) if total > 0 else 0

    # 分区域统计
    district_agg = df_latest.groupby("hospital_district").agg({
        "total_beds": "sum",
        "occupied_beds": "sum",
        "available_beds": "sum",
    }).reset_index()
    district_agg["occupancy_rate"] = (
        district_agg["occupied_beds"] / district_agg["total_beds"] * 100
    ).round(2)

    return {
        "total_beds": total,
        "occupied_beds": occupied,
        "available_beds": available,
        "occupancy_rate": rate,
        "full_wards": full_count,
        "latest_date": df_latest["timestamp"].max().strftime("%Y-%m-%d"),
        "district_data": district_agg.to_dict("records"),
    }


def process_hospital_ranking(df_latest):
    """各医院使用率排名"""
    agg = df_latest.groupby(
        ["hospital_id", "hospital_name", "hospital_district"]
    ).agg({
        "total_beds": "sum",
        "occupied_beds": "sum",
        "available_beds": "sum",
    }).reset_index()

    agg["occupancy_rate"] = (
        agg["occupied_beds"] / agg["total_beds"] * 100
    ).round(2)
    agg = agg.sort_values("occupancy_rate", ascending=False)

    return {
        "hospitals": agg["hospital_name"].tolist(),
        "rates": agg["occupancy_rate"].tolist(),
        "total_beds": agg["total_beds"].tolist(),
        "occupied_beds": agg["occupied_beds"].tolist(),
        "available_beds": agg["available_beds"].tolist(),
        "districts": agg["hospital_district"].tolist(),
    }


def process_department_distribution(df_latest):
    """各科室床位分布"""
    agg = df_latest.groupby("department_name").agg({
        "total_beds": "sum",
        "occupied_beds": "sum",
        "available_beds": "sum",
        "ward_id": "nunique",
    }).reset_index()
    agg.columns = ["department", "total_beds", "occupied_beds", "available_beds", "ward_count"]
    agg = agg.sort_values("total_beds", ascending=False)

    return {
        "departments": agg["department"].tolist(),
        "total": agg["total_beds"].tolist(),
        "occupied": agg["occupied_beds"].tolist(),
        "available": agg["available_beds"].tolist(),
    }


def process_available_beds(df_latest):
    """空闲病床统计"""
    # 按医院统计空闲床位
    hospital_avail = df_latest.groupby("hospital_name")["available_beds"].sum()
    hospital_avail = hospital_avail.sort_values(ascending=False)

    # 按区间分布
    bins = [0, 50, 100, 200, float("inf")]
    labels = ["0-50", "50-100", "100-200", "200+"]
    df_latest["avail_bin"] = pd.cut(
        df_latest["available_beds"], bins=bins, labels=labels, right=False
    )
    avail_dist = df_latest.groupby("avail_bin", observed=False)["ward_id"].count()

    return {
        "by_hospital": [
            {"name": name, "value": int(val)}
            for name, val in hospital_avail.items()
        ],
        "distribution": [
            {"name": str(k), "value": int(v)}
            for k, v in avail_dist.items()
        ],
        "total_available": int(df_latest["available_beds"].sum()),
    }


def process_abnormal(df_latest):
    """异常情况统计"""
    status_list = ["满床", "临时关闭", "维修中", "仅急诊入院"]
    hospital_list = df_latest["hospital_name"].unique().tolist()

    data = {}
    for status in status_list:
        data[status] = []
        for h in hospital_list:
            count = len(
                df_latest[
                    (df_latest["hospital_name"] == h)
                    & (df_latest["special_status"] == status)
                ]
            )
            data[status].append(count)

    return {
        "hospitals": hospital_list,
        "series": [
            {"name": status, "data": data[status]}
            for status in status_list
            if sum(data[status]) > 0
        ],
    }


# ============ 路由 ============

@app.route("/")
def index():
    """大屏主页"""
    return render_template("index.html")


@app.route("/api/summary")
def api_summary():
    """KPI 数据"""
    df = load_data()
    df_latest, _ = get_latest(df)
    return jsonify(process_summary(df_latest))


@app.route("/api/hospital-ranking")
def api_hospital_ranking():
    """医院使用率排名"""
    df = load_data()
    df_latest, _ = get_latest(df)
    return jsonify(process_hospital_ranking(df_latest))


@app.route("/api/department-distribution")
def api_department_distribution():
    """科室分布"""
    df = load_data()
    df_latest, _ = get_latest(df)
    return jsonify(process_department_distribution(df_latest))


@app.route("/api/available-beds")
def api_available_beds():
    """空闲床位统计"""
    df = load_data()
    df_latest, _ = get_latest(df)
    return jsonify(process_available_beds(df_latest))


@app.route("/api/abnormal")
def api_abnormal():
    """异常情况"""
    df = load_data()
    df_latest, _ = get_latest(df)
    return jsonify(process_abnormal(df_latest))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
