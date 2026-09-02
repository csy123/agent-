/**
 * 香港疫情实时监控大屏 - 图表渲染逻辑
 * 使用 ECharts 渲染各类图表
 */

// ============ 工具函数 ============

/** 数字格式化（添加千分位） */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/** 风险等级对应颜色 */
function getRiskColor(level) {
    const map = {
        "低风险": "#6bcf7f",
        "中风险": "#ffd93d",
        "高风险": "#ff6b6b",
    };
    return map[level] || "#90e0ef";
}

// ============ 时间更新 ============

function updateDateTime() {
    const now = new Date();
    const str = now.toLocaleString("zh-CN", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
    });
    document.getElementById("datetime").textContent = str;
}

// ============ 数据加载与渲染 ============

/** 加载核心指标 */
async function loadSummary() {
    try {
        const resp = await fetch("/api/summary");
        const data = await resp.json();

        document.getElementById("total-confirmed").textContent =
            formatNumber(data.total_confirmed);
        document.getElementById("active-cases").textContent =
            formatNumber(data.active_cases);
        document.getElementById("total-recovered").textContent =
            formatNumber(data.total_recovered);
        document.getElementById("total-deaths").textContent =
            formatNumber(data.total_deaths);

        document.getElementById("new-confirmed").textContent =
            formatNumber(data.new_confirmed);
        document.getElementById("new-recovered").textContent =
            formatNumber(data.new_recovered);
        document.getElementById("new-deaths").textContent =
            formatNumber(data.new_deaths);

        document.getElementById("update-time").textContent = data.latest_date;
    } catch (err) {
        console.error("加载核心指标失败:", err);
    }
}

/** 渲染趋势图 */
let trendChart = null;
async function loadTrend() {
    try {
        const resp = await fetch("/api/trend");
        const data = await resp.json();

        if (!trendChart) {
            trendChart = echarts.init(
                document.getElementById("trend-chart")
            );
        }

        const option = {
            tooltip: {
                trigger: "axis",
                backgroundColor: "rgba(12, 25, 41, 0.9)",
                borderColor: "#00b4d8",
                textStyle: { color: "#e0e6ed" },
            },
            legend: {
                textStyle: { color: "#90e0ef" },
                top: 5,
            },
            grid: {
                left: 50,
                right: 30,
                top: 40,
                bottom: 30,
            },
            xAxis: {
                type: "category",
                data: data.dates,
                axisLine: { lineStyle: { color: "#00b4d8" } },
                axisLabel: { color: "#90e0ef", fontSize: 11 },
            },
            yAxis: {
                type: "value",
                axisLine: { lineStyle: { color: "#00b4d8" } },
                axisLabel: { color: "#90e0ef" },
                splitLine: {
                    lineStyle: { color: "rgba(0, 180, 216, 0.1)" },
                },
            },
            series: [
                {
                    name: "新增确诊",
                    type: "line",
                    smooth: true,
                    data: data.new_confirmed,
                    itemStyle: { color: "#ff6b6b" },
                    areaStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: "rgba(255, 107, 107, 0.4)" },
                            { offset: 1, color: "rgba(255, 107, 107, 0)" },
                        ]),
                    },
                },
                {
                    name: "新增康复",
                    type: "line",
                    smooth: true,
                    data: data.new_recovered,
                    itemStyle: { color: "#6bcf7f" },
                    areaStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: "rgba(107, 207, 127, 0.4)" },
                            { offset: 1, color: "rgba(107, 207, 127, 0)" },
                        ]),
                    },
                },
                {
                    name: "累计确诊",
                    type: "line",
                    smooth: true,
                    data: data.total_confirmed,
                    itemStyle: { color: "#00b4d8" },
                    lineStyle: { width: 2, type: "dashed" },
                },
            ],
        };

        trendChart.setOption(option);
    } catch (err) {
        console.error("加载趋势数据失败:", err);
    }
}

/** 渲染风险等级饼图 */
let riskChart = null;
async function loadRisk() {
    try {
        const resp = await fetch("/api/risk");
        const data = await resp.json();

        if (!riskChart) {
            riskChart = echarts.init(
                document.getElementById("risk-chart")
            );
        }

        const colors = {
            "低风险": "#6bcf7f",
            "中风险": "#ffd93d",
            "高风险": "#ff6b6b",
        };

        const coloredData = data.map((item) => ({
            ...item,
            itemStyle: { color: colors[item.name] || "#90e0ef" },
        }));

        const option = {
            tooltip: {
                trigger: "item",
                backgroundColor: "rgba(12, 25, 41, 0.9)",
                borderColor: "#00b4d8",
                textStyle: { color: "#e0e6ed" },
                formatter: "{b}: {c}个区 ({d}%)",
            },
            legend: {
                orient: "vertical",
                right: 20,
                top: "middle",
                textStyle: { color: "#90e0ef" },
            },
            series: [
                {
                    name: "风险等级",
                    type: "pie",
                    radius: ["40%", "70%"],
                    center: ["40%", "50%"],
                    avoidLabelOverlap: true,
                    label: {
                        show: true,
                        formatter: "{b}\n{c}个区",
                        color: "#90e0ef",
                    },
                    data: coloredData,
                },
            ],
        };

        riskChart.setOption(option);
    } catch (err) {
        console.error("加载风险数据失败:", err);
    }
}

/** 渲染TOP5柱状图 */
let topChart = null;
async function loadTop() {
    try {
        const resp = await fetch("/api/top");
        const data = await resp.json();

        if (!topChart) {
            topChart = echarts.init(
                document.getElementById("top-chart")
            );
        }

        const names = data.map((d) => d.name);
        const values = data.map((d) => d.value);

        const option = {
            tooltip: {
                trigger: "axis",
                backgroundColor: "rgba(12, 25, 41, 0.9)",
                borderColor: "#00b4d8",
                textStyle: { color: "#e0e6ed" },
                formatter: "{b}<br/>累计确诊: {c}",
            },
            grid: {
                left: 80,
                right: 30,
                top: 20,
                bottom: 30,
            },
            xAxis: {
                type: "value",
                axisLine: { lineStyle: { color: "#00b4d8" } },
                axisLabel: { color: "#90e0ef" },
                splitLine: {
                    lineStyle: { color: "rgba(0, 180, 216, 0.1)" },
                },
            },
            yAxis: {
                type: "category",
                data: names.reverse(),
                axisLine: { lineStyle: { color: "#00b4d8" } },
                axisLabel: { color: "#90e0ef" },
            },
            series: [
                {
                    type: "bar",
                    data: values.reverse(),
                    barWidth: "50%",
                    itemStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                            { offset: 0, color: "rgba(255, 107, 107, 0.6)" },
                            { offset: 1, color: "rgba(255, 107, 107, 1)" },
                        ]),
                        borderRadius: [0, 4, 4, 0],
                    },
                    label: {
                        show: true,
                        position: "right",
                        color: "#90e0ef",
                        formatter: (p) => formatNumber(p.value),
                    },
                },
            ],
        };

        topChart.setOption(option);
    } catch (err) {
        console.error("加载TOP5数据失败:", err);
    }
}

/** 渲染各区详细表格 */
async function loadDistricts() {
    try {
        const resp = await fetch("/api/districts");
        const data = await resp.json();

        const tbody = document.getElementById("district-tbody");
        tbody.innerHTML = data
            .map(
                (d) => `
            <tr>
                <td>${d.name}</td>
                <td>${formatNumber(d.total_confirmed)}</td>
                <td>${formatNumber(d.active_cases)}</td>
                <td>${d.recovery_rate}%</td>
                <td>${d.incidence_rate}</td>
                <td style="color: ${getRiskColor(d.risk_level)}; font-weight: 500;">
                    ${d.risk_level}
                </td>
            </tr>
        `
            )
            .join("");
    } catch (err) {
        console.error("加载各区数据失败:", err);
    }
}

// ============ 初始化与定时刷新 ============

/** 加载全部数据 */
function loadAll() {
    loadSummary();
    loadTrend();
    loadRisk();
    loadTop();
    loadDistricts();
}

/** 窗口大小变化时重置图表 */
window.addEventListener("resize", () => {
    trendChart && trendChart.resize();
    riskChart && riskChart.resize();
    topChart && topChart.resize();
});

/** 页面加载完成后启动 */
document.addEventListener("DOMContentLoaded", () => {
    updateDateTime();
    setInterval(updateDateTime, 1000);

    loadAll();

    // 每30秒自动刷新数据
    setInterval(loadAll, 30000);
});
