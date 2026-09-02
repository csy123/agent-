#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
香港疫情实时监控大屏 - Flask应用主文件
读取香港各区疫情数据，通过API向前端提供数据服务
"""

import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template

app = Flask(__name__)

# 数据文件路径
DATA_FILE = os.path.join(os.path.dirname(__file__), "香港各区疫情数据_20250322.xlsx")


def load_data():
    """加载疫情数据"""
    df = pd.read_excel(DATA_FILE)
    df["报告日期"] = pd.to_datetime(df["报告日期"])
    return df


def get_summary_stats(df):
    """获取核心指标汇总统计"""
    latest_date = df["报告日期"].max()
    latest_data = df[df["报告日期"] == latest_date]

    return {
        "latest_date": latest_date.strftime("%Y-%m-%d"),
        "total_confirmed": int(latest_data["累计确诊"].sum()),
        "total_recovered": int(latest_data["累计康复"].sum()),
        "total_deaths": int(latest_data["累计死亡"].sum()),
        "active_cases": int(latest_data["现存确诊"].sum()),
        "new_confirmed": int(latest_data["新增确诊"].sum()),
        "new_recovered": int(latest_data["新增康复"].sum()),
        "new_deaths": int(latest_data["新增死亡"].sum()),
    }


def get_trend_data(df, days=30):
    """获取趋势数据（最近N天）"""
    # 按日期汇总
    daily = df.groupby("报告日期").agg({
        "新增确诊": "sum",
        "累计确诊": "sum",
        "新增康复": "sum",
        "累计康复": "sum",
        "新增死亡": "sum",
        "累计死亡": "sum",
    }).reset_index()

    daily = daily.sort_values("报告日期").tail(days)

    dates = daily["报告日期"].dt.strftime("%m-%d").tolist()
    new_confirmed = daily["新增确诊"].tolist()
    total_confirmed = daily["累计确诊"].tolist()
    new_recovered = daily["新增康复"].tolist()
    new_deaths = daily["新增死亡"].tolist()

    return {
        "dates": dates,
        "new_confirmed": new_confirmed,
        "total_confirmed": total_confirmed,
        "new_recovered": new_recovered,
        "new_deaths": new_deaths,
    }


def get_district_data(df):
    """获取各区详细数据"""
    latest_date = df["报告日期"].max()
    latest_data = df[df["报告日期"] == latest_date]

    districts = []
    for _, row in latest_data.iterrows():
        districts.append({
            "name": row["地区名称"],
            "new_confirmed": int(row["新增确诊"]),
            "total_confirmed": int(row["累计确诊"]),
            "active_cases": int(row["现存确诊"]),
            "recovery_rate": round(
                row["累计康复"] / row["累计确诊"] * 100, 2
            ) if row["累计确诊"] > 0 else 0,
            "incidence_rate": round(row["发病率(每10万人)"], 2),
            "risk_level": row["风险等级"],
            "population": int(row["人口"]),
        })

    return sorted(districts, key=lambda x: x["total_confirmed"], reverse=True)


def get_risk_distribution(df):
    """获取风险等级分布"""
    latest_date = df["报告日期"].max()
    latest_data = df[df["报告日期"] == latest_date]

    risk_counts = latest_data["风险等级"].value_counts().to_dict()
    return [
        {"name": level, "value": count}
        for level, count in risk_counts.items()
    ]


def get_top_districts(df, top_n=5):
    """获取疫情最严重的TOP N地区"""
    latest_date = df["报告日期"].max()
    latest_data = df[df["报告日期"] == latest_date]

    top = latest_data.nlargest(top_n, "累计确诊")
    return [
        {
            "name": row["地区名称"],
            "value": int(row["累计确诊"]),
        }
        for _, row in top.iterrows()
    ]


# ============ 路由 ============

@app.route("/")
def index():
    """主页"""
    return render_template("index.html")


@app.route("/api/summary")
def api_summary():
    """核心指标API"""
    df = load_data()
    return jsonify(get_summary_stats(df))


@app.route("/api/trend")
def api_trend():
    """趋势数据API"""
    df = load_data()
    return jsonify(get_trend_data(df))


@app.route("/api/districts")
def api_districts():
    """各区详细数据API"""
    df = load_data()
    return jsonify(get_district_data(df))


@app.route("/api/risk")
def api_risk():
    """风险等级分布API"""
    df = load_data()
    return jsonify(get_risk_distribution(df))


@app.route("/api/top")
def api_top():
    """TOP5疫情地区API"""
    df = load_data()
    return jsonify(get_top_districts(df))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
