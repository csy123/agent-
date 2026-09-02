#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医院床位使用率可视化
使用 ECharts 生成交互式图表，输出为 HTML 文件
"""

import pandas as pd
import json
import os

# ============ 1. 读取数据并处理 ============
DATA_FILE = os.path.join(os.path.dirname(__file__), "hospital_bed_usage_data.xlsx")
df = pd.read_excel(DATA_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 取最新时间点
latest_ts = df["timestamp"].max()
df_latest = df[df["timestamp"] == latest_ts].copy()

# --- 1.1 各医院整体使用率 ---
hospital_stats = df_latest.groupby(
    ["hospital_id", "hospital_name", "hospital_district"]
).agg({
    "total_beds": "sum",
    "occupied_beds": "sum",
    "available_beds": "sum",
}).reset_index()

hospital_stats["occupancy_rate"] = (
    hospital_stats["occupied_beds"] / hospital_stats["total_beds"] * 100
).round(2)

hospital_stats = hospital_stats.sort_values("occupancy_rate", ascending=False)

# --- 1.2 各医院×科室使用率 ---
dept_stats = df_latest.groupby(
    ["hospital_name", "department_name"]
).agg({
    "total_beds": "sum",
    "occupied_beds": "sum",
    "available_beds": "sum",
}).reset_index()

dept_stats["occupancy_rate"] = (
    dept_stats["occupied_beds"] / dept_stats["total_beds"] * 100
).round(2)

# 构建热力图数据：行=科室，列=医院
hospitals = hospital_stats["hospital_name"].tolist()
departments = dept_stats["department_name"].unique().tolist()

heatmap_data = []
for _, row in dept_stats.iterrows():
    hi = hospitals.index(row["hospital_name"])
    di = departments.index(row["department_name"])
    heatmap_data.append([hi, di, row["occupancy_rate"]])

# --- 1.3 使用率区间分布 ---
bins = [0, 50, 70, 85, 100]
labels = ["<50%", "50-70%", "70-85%", "85-100%"]
df_latest["使用率区间"] = pd.cut(
    df_latest["occupancy_rate"], bins=bins, labels=labels, include_lowest=True
)
dist_data = df_latest.groupby("使用率区间", observed=False).agg({
    "ward_id": "count",
}).rename(columns={"ward_id": "count"}).reset_index()

# --- 1.4 异常情况 ---
full_bed = df_latest[df_latest["special_status"] == "满床"]
closed = df_latest[df_latest["special_status"] == "临时关闭"]
repair = df_latest[df_latest["special_status"] == "维修中"]

# 按医院统计异常
abnormal = []
for h in hospitals:
    h_data = df_latest[df_latest["hospital_name"] == h]
    full_count = len(h_data[h_data["special_status"] == "满床"])
    closed_count = len(h_data[h_data["special_status"] == "临时关闭"])
    repair_count = len(h_data[h_data["special_status"] == "维修中"])
    if full_count or closed_count or repair_count:
        abnormal.append({
            "hospital": h,
            "full": full_count,
            "closed": closed_count,
            "repair": repair_count,
        })

# --- 1.5 全院汇总 ---
total_beds = int(df_latest["total_beds"].sum())
occupied_beds = int(df_latest["occupied_beds"].sum())
available_beds = int(df_latest["available_beds"].sum())
overall_rate = round(occupied_beds / total_beds * 100, 2)
full_wards = len(full_bed)

# ============ 2. 生成 HTML ============
html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>香港医院床位使用率可视化</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.3.3/dist/echarts.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
    background: linear-gradient(135deg, #0c1929, #1a2a3a, #0c1929);
    color: #e0e6ed; min-height: 100vh;
}}
.header {{
    text-align: center; padding: 18px;
    background: linear-gradient(90deg, transparent, rgba(0,180,216,0.25), transparent);
    border-bottom: 2px solid rgba(0,180,216,0.4);
}}
.header h1 {{
    font-size: 26px; letter-spacing: 4px;
    background: linear-gradient(90deg, #00b4d8, #90e0ef);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.header .date {{ color: #90e0ef; margin-top: 6px; font-size: 13px; }}

.summary {{
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 16px; padding: 18px 20px;
}}
.summary .card {{
    text-align: center; padding: 18px 12px;
    border-radius: 10px;
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(0,180,216,0.3);
}}
.card .label {{ font-size: 13px; color: #90e0ef; margin-bottom: 8px; }}
.card .value {{ font-size: 30px; font-weight: bold; font-family: "DIN", Arial, sans-serif; }}
.card.sub .value {{ color: #6bcf7f; }}
.card.occ .value {{ color: #00b4d8; }}
.card.full .value {{ color: #ff6b6b; }}
.card.avail .value {{ color: #ffd93d; }}

.charts {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
    padding: 0 20px 20px;
}}
.chart-panel {{
    border-radius: 10px;
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(0,180,216,0.3);
    overflow: hidden;
}}
.chart-panel.full {{ grid-column: span 2; }}
.panel-header {{
    padding: 10px 18px;
    border-bottom: 1px solid rgba(0,180,216,0.2);
    background: rgba(0,180,216,0.1);
}}
.panel-header h3 {{ font-size: 15px; color: #90e0ef; font-weight: 500; }}
.chart-body {{ padding: 12px; height: 380px; }}
.chart-body.tall {{ height: 420px; }}

.footer {{
    text-align: center; padding: 12px; font-size: 12px; color: #8899aa;
    border-top: 1px solid rgba(0,180,216,0.2);
}}
</style>
</head>
<body>

<div class="header">
    <h1>🏥 香港医院床位使用率可视化</h1>
    <div class="date">数据时间点：{latest_ts.strftime('%Y-%m-%d')} | 共 {len(hospitals)} 家医院</div>
</div>

<div class="summary">
    <div class="card">
        <div class="label">总床位数</div>
        <div class="value">{total_beds:,}</div>
    </div>
    <div class="card sub">
        <div class="label">已占用</div>
        <div class="value">{occupied_beds:,}</div>
    </div>
    <div class="card avail">
        <div class="label">可用床位</div>
        <div class="value">{available_beds:,}</div>
    </div>
    <div class="card occ">
        <div class="label">整体使用率</div>
        <div class="value">{overall_rate}%</div>
    </div>
    <div class="card full">
        <div class="label">满床病房</div>
        <div class="value">{full_wards}</div>
    </div>
</div>

<div class="charts">
    <div class="chart-panel">
        <div class="panel-header"><h3>📊 各医院病床使用率排名</h3></div>
        <div class="chart-body" id="chart-rank"></div>
    </div>
    <div class="chart-panel">
        <div class="panel-header"><h3>🥧 床位使用率区间分布</h3></div>
        <div class="chart-body" id="chart-dist"></div>
    </div>
    <div class="chart-panel full">
        <div class="panel-header"><h3>🔥 各医院 × 科室 使用率热力图</h3></div>
        <div class="chart-body tall" id="chart-heatmap"></div>
    </div>
    <div class="chart-panel full">
        <div class="panel-header"><h3>⚠️ 异常情况统计（满床/关闭/维修）</h3></div>
        <div class="chart-body" id="chart-abnormal"></div>
    </div>
</div>

<div class="footer">
    数据源：hospital_bed_usage_data.xlsx | 最新日期：{latest_ts.strftime('%Y-%m-%d')}
</div>

<script>
// ============ 图表1：医院使用率排名 ============
var rankChart = echarts.init(document.getElementById('chart-rank'));
var rankOption = {{
    tooltip: {{
        trigger: 'axis',
        backgroundColor: 'rgba(12,25,41,0.9)',
        borderColor: '#00b4d8',
        textStyle: {{ color: '#e0e6ed' }},
        formatter: function(p) {{
            var d = p[0];
            return d.name + '<br/>' + 
                   '使用率: <b>' + d.value + '%</b><br/>' +
                   '区域: ' + (hospitalDistrict[d.dataIndex] || '');
        }}
    }},
    grid: {{ left: 100, right: 60, top: 10, bottom: 30 }},
    xAxis: {{
        type: 'value', max: 100,
        axisLine: {{ lineStyle: {{ color: '#00b4d8' }} }},
        axisLabel: {{ color: '#90e0ef', formatter: '{{value}}%' }},
        splitLine: {{ lineStyle: {{ color: 'rgba(0,180,216,0.1)' }} }}
    }},
    yAxis: {{
        type: 'category',
        data: {json.dumps(hospital_stats["hospital_name"].tolist())}.reverse(),
        axisLine: {{ lineStyle: {{ color: '#00b4d8' }} }},
        axisLabel: {{ color: '#90e0ef', fontSize: 11 }}
    }},
    series: [{{
        type: 'bar',
        data: {json.dumps(hospital_stats["occupancy_rate"].tolist())}.reverse(),
        barWidth: '55%',
        itemStyle: {{
            color: function(p) {{
                var v = p.value;
                if (v >= 85) return '#ff6b6b';
                if (v >= 70) return '#ffd93d';
                return '#6bcf7f';
            }},
            borderRadius: [0, 4, 4, 0]
        }},
        label: {{
            show: true, position: 'right', color: '#90e0ef',
            formatter: '{{c}}%'
        }},
        markLine: {{
            silent: true,
            lineStyle: {{ color: '#ff6b6b', type: 'dashed' }},
            data: [{{ xAxis: 85, label: {{ formatter: '警戒线 85%', color: '#ff6b6b' }} }}]
        }}
    }}]
}};

var hospitalDistrict = {json.dumps(dict(zip(hospital_stats["hospital_name"], hospital_stats["hospital_district"])))};

rankChart.setOption(rankOption);

// ============ 图表2：使用率区间分布 ============
var distChart = echarts.init(document.getElementById('chart-dist'));
var distData = {json.dumps([
    {"name": str(row["使用率区间"]), "value": int(row["count"])}
    for _, row in dist_data.iterrows()
])};

var distOption = {{
    tooltip: {{
        trigger: 'item',
        backgroundColor: 'rgba(12,25,41,0.9)',
        borderColor: '#00b4d8',
        textStyle: {{ color: '#e0e6ed' }},
        formatter: '{{b}}<br/>病房数: {{c}}<br/>占比: {{d}}%'
    }},
    legend: {{ bottom: 10, textStyle: {{ color: '#90e0ef' }} }},
    color: ['#6bcf7f', '#ffd93d', '#ff9f43', '#ff6b6b'],
    series: [{{
        type: 'pie',
        radius: ['40%', '70%'],
        center: ['50%', '45%'],
        avoidLabelOverlap: true,
        label: {{
            show: true,
            formatter: '{{b}}\\n{{c}}间 ({{d}}%)',
            color: '#90e0ef'
        }},
        data: distData
    }}]
}};
distChart.setOption(distOption);

// ============ 图表3：热力图 ============
var heatmapChart = echarts.init(document.getElementById('chart-heatmap'));

// 为热力图补全数据（没有数据的组合用 -1 表示）
var heatmapData = {json.dumps(heatmap_data)};
var hospitals = {json.dumps(hospitals)};
var departments = {json.dumps(departments)};

var heatOption = {{
    tooltip: {{
        backgroundColor: 'rgba(12,25,41,0.9)',
        borderColor: '#00b4d8',
        textStyle: {{ color: '#e0e6ed' }},
        formatter: function(p) {{
            if (p.value[2] === -1) return hospitals[p.value[0]] + ' - ' + departments[p.value[1]] + '<br/>暂无数据';
            return hospitals[p.value[0]] + ' - ' + departments[p.value[1]] + '<br/>使用率: <b>' + p.value[2] + '%</b>';
        }}
    }},
    grid: {{ left: 140, right: 60, top: 40, bottom: 80 }},
    xAxis: {{
        type: 'category',
        data: hospitals,
        axisLabel: {{ color: '#90e0ef', rotate: 30, fontSize: 11 }},
        axisLine: {{ lineStyle: {{ color: '#00b4d8' }} }},
        splitArea: {{ show: true }}
    }},
    yAxis: {{
        type: 'category',
        data: departments,
        axisLabel: {{ color: '#90e0ef', fontSize: 11 }},
        axisLine: {{ lineStyle: {{ color: '#00b4d8' }} }},
        splitArea: {{ show: true }}
    }},
    visualMap: {{
        min: 0, max: 100,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: 10,
        textStyle: {{ color: '#90e0ef' }},
        inRange: {{
            color: ['#1a3a4a', '#6bcf7f', '#ffd93d', '#ff6b6b']
        }},
        text: ['高', '低']
    }},
    series: [{{
        type: 'heatmap',
        data: heatmapData,
        label: {{
            show: true,
            color: '#fff',
            fontSize: 9,
            formatter: function(p) {{
                return p.value[2] === -1 ? '-' : p.value[2] + '%';
            }}
        }},
        itemStyle: {{ borderColor: 'rgba(12,25,41,0.8)' }}
    }}]
}};
heatmapChart.setOption(heatOption);

// ============ 图表4：异常情况 ============
var abnormalChart = echarts.init(document.getElementById('chart-abnormal'));
var abnormalData = {json.dumps(abnormal)};

var abOption = {{
    tooltip: {{
        trigger: 'axis',
        backgroundColor: 'rgba(12,25,41,0.9)',
        borderColor: '#00b4d8',
        textStyle: {{ color: '#e0e6ed' }},
        axisPointer: {{ type: 'shadow' }}
    }},
    legend: {{ textStyle: {{ color: '#90e0ef' }}, top: 5 }},
    grid: {{ left: 100, right: 30, top: 40, bottom: 30 }},
    xAxis: {{
        type: 'value',
        axisLine: {{ lineStyle: {{ color: '#00b4d8' }} }},
        axisLabel: {{ color: '#90e0ef' }},
        splitLine: {{ lineStyle: {{ color: 'rgba(0,180,216,0.1)' }} }}
    }},
    yAxis: {{
        type: 'category',
        data: abnormalData.map(function(d) {{ return d.hospital; }}),
        axisLine: {{ lineStyle: {{ color: '#00b4d8' }} }},
        axisLabel: {{ color: '#90e0ef', fontSize: 11 }}
    }},
    series: [
        {{
            name: '满床', type: 'bar', stack: 'total',
            data: abnormalData.map(function(d) {{ return d.full; }}),
            itemStyle: {{ color: '#ff6b6b' }},
            label: {{ show: true, position: 'right', color: '#fff', fontSize: 10 }}
        }},
        {{
            name: '临时关闭', type: 'bar', stack: 'total',
            data: abnormalData.map(function(d) {{ return d.closed; }}),
            itemStyle: {{ color: '#ffd93d' }},
            label: {{ show: true, position: 'right', color: '#fff', fontSize: 10 }}
        }},
        {{
            name: '维修中', type: 'bar', stack: 'total',
            data: abnormalData.map(function(d) {{ return d.repair; }}),
            itemStyle: {{ color: '#ff9f43' }},
            label: {{ show: true, position: 'right', color: '#fff', fontSize: 10 }}
        }}
    ]
}};
abnormalChart.setOption(abOption);

// ============ 自适应 ============
window.addEventListener('resize', function() {{
    rankChart.resize();
    distChart.resize();
    heatmapChart.resize();
    abnormalChart.resize();
}});
</script>
</body>
</html>
"""

# ============ 3. 保存 HTML ============
output_path = os.path.join(os.path.dirname(__file__), "bed_usage_dashboard.html")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ 可视化页面已生成: {output_path}")
print(f"\n包含图表:")
print(f"  1. 各医院病床使用率排名（条形图 + 警戒线）")
print(f"  2. 床位使用率区间分布（环形饼图）")
print(f"  3. 各医院×科室使用率热力图")
print(f"  4. 异常情况统计（堆叠条形图）")
print(f"\n核心指标:")
print(f"  总床位: {total_beds:,}")
print(f"  已占用: {occupied_beds:,}")
print(f"  可用: {available_beds:,}")
print(f"  整体使用率: {overall_rate}%")
print(f"  满床病房: {full_wards} 间")
