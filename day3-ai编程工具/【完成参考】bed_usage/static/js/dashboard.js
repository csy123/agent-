/**
 * 医院床位使用率可视化大屏 - 图表渲染逻辑
 */

// ============ 工具函数 ============
function formatNumber(num) {
    if (num === undefined || num === null) return '--';
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

// ============ 时间更新 ============
function updateDateTime() {
    const now = new Date();
    const str = now.toLocaleString('zh-CN', {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
    document.getElementById('datetime').textContent = str;
}

// ============ KPI 数据 ============
async function loadKPI() {
    try {
        const resp = await fetch('/api/summary');
        const data = await resp.json();

        document.getElementById('kpi-total').textContent =
            formatNumber(data.total_beds);
        document.getElementById('kpi-rate').textContent =
            data.occupancy_rate + '%';
        document.getElementById('kpi-occupied').textContent =
            formatNumber(data.occupied_beds);
        document.getElementById('kpi-available').textContent =
            formatNumber(data.available_beds);
        document.getElementById('kpi-full').textContent =
            formatNumber(data.full_wards);
        document.getElementById('update-date').textContent = data.latest_date;
    } catch (err) {
        console.error('加载 KPI 失败:', err);
    }
}

// ============ 图表1：医院使用率排名 ============
let hospitalChart = null;
async function loadHospitalChart() {
    try {
        const resp = await fetch('/api/hospital-ranking');
        const data = await resp.json();

        if (!hospitalChart) {
            hospitalChart = echarts.init(document.getElementById('chart-hospital'));
        }

        const option = {
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(12, 25, 41, 0.9)',
                borderColor: '#00b4d8',
                textStyle: { color: '#e0e6ed' },
                formatter: function(params) {
                    const p = params[0];
                    const idx = p.dataIndex;
                    return `
                        <b>${data.hospitals[idx]}</b><br/>
                        区域: ${data.districts[idx]}<br/>
                        总床位: ${formatNumber(data.total_beds[idx])}<br/>
                        已占用: ${formatNumber(data.occupied_beds[idx])}<br/>
                        空闲: ${formatNumber(data.available_beds[idx])}<br/>
                        <b style="color:#ffd93d">使用率: ${p.value}%</b>
                    `;
                }
            },
            grid: { left: 120, right: 60, top: 10, bottom: 30 },
            xAxis: {
                type: 'value',
                max: 100,
                axisLine: { lineStyle: { color: '#00b4d8' } },
                axisLabel: { color: '#90e0ef', formatter: '{value}%' },
                splitLine: { lineStyle: { color: 'rgba(0,180,216,0.1)' } }
            },
            yAxis: {
                type: 'category',
                data: data.hospitals.slice().reverse(),
                axisLine: { lineStyle: { color: '#00b4d8' } },
                axisLabel: { color: '#90e0ef', fontSize: 11 }
            },
            series: [{
                type: 'bar',
                data: data.rates.slice().reverse(),
                barWidth: '55%',
                itemStyle: {
                    color: function(params) {
                        const v = params.value;
                        if (v >= 85) return '#ff6b6b';
                        if (v >= 70) return '#ffd93d';
                        return '#6bcf7f';
                    },
                    borderRadius: [0, 4, 4, 0]
                },
                label: {
                    show: true,
                    position: 'right',
                    color: '#90e0ef',
                    formatter: '{c}%'
                },
                markLine: {
                    silent: true,
                    lineStyle: { color: '#ff6b6b', type: 'dashed' },
                    data: [{ xAxis: 85, label: { formatter: '警戒线 85%', color: '#ff6b6b' } }]
                }
            }]
        };

        hospitalChart.setOption(option, true);
    } catch (err) {
        console.error('加载医院排名失败:', err);
    }
}

// ============ 图表2：科室病床分布 ============
let deptChart = null;
async function loadDeptChart() {
    try {
        const resp = await fetch('/api/department-distribution');
        const data = await resp.json();

        if (!deptChart) {
            deptChart = echarts.init(document.getElementById('chart-department'));
        }

        const option = {
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(12, 25, 41, 0.9)',
                borderColor: '#00b4d8',
                textStyle: { color: '#e0e6ed' },
                axisPointer: { type: 'shadow' }
            },
            legend: {
                textStyle: { color: '#90e0ef' },
                top: 0
            },
            grid: { left: 50, right: 30, top: 30, bottom: 50 },
            xAxis: {
                type: 'category',
                data: data.departments,
                axisLine: { lineStyle: { color: '#00b4d8' } },
                axisLabel: { color: '#90e0ef', rotate: 30, fontSize: 10 }
            },
            yAxis: {
                type: 'value',
                axisLine: { lineStyle: { color: '#00b4d8' } },
                axisLabel: { color: '#90e0ef' },
                splitLine: { lineStyle: { color: 'rgba(0,180,216,0.1)' } }
            },
            series: [
                {
                    name: '已占用',
                    type: 'bar',
                    stack: 'total',
                    data: data.occupied,
                    itemStyle: { color: '#ff6b6b' },
                    label: { show: false }
                },
                {
                    name: '空闲',
                    type: 'bar',
                    stack: 'total',
                    data: data.available,
                    itemStyle: { color: '#6bcf7f' },
                    label: { show: false }
                }
            ]
        };

        deptChart.setOption(option, true);
    } catch (err) {
        console.error('加载科室分布失败:', err);
    }
}

// ============ 图表3：空闲病床统计 ============
let availableChart = null;
async function loadAvailableChart() {
    try {
        const resp = await fetch('/api/available-beds');
        const data = await resp.json();

        if (!availableChart) {
            availableChart = echarts.init(document.getElementById('chart-available'));
        }

        const option = {
            tooltip: {
                trigger: 'item',
                backgroundColor: 'rgba(12, 25, 41, 0.9)',
                borderColor: '#00b4d8',
                textStyle: { color: '#e0e6ed' }
            },
            legend: {
                type: 'scroll',
                orient: 'vertical',
                right: 10,
                top: 'middle',
                textStyle: { color: '#90e0ef', fontSize: 11 }
            },
            color: ['#6bcf7f', '#ffd93d', '#ff9f43', '#ff6b6b', '#00b4d8', '#90e0ef', '#c0c0c0', '#f368e0', '#54a0ff', '#5f27cd', '#00d2d3', '#ee5a24'],
            series: [{
                type: 'pie',
                radius: ['40%', '70%'],
                center: ['35%', '50%'],
                data: data.by_hospital,
                label: {
                    show: true,
                    formatter: '{d}%',
                    color: '#90e0ef'
                },
                labelLine: {
                    lineStyle: { color: '#00b4d8' }
                },
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowColor: 'rgba(0, 180, 216, 0.5)'
                    }
                }
            }]
        };

        availableChart.setOption(option, true);
    } catch (err) {
        console.error('加载空闲床位失败:', err);
    }
}

// ============ 图表4：异常情况统计 ============
let abnormalChart = null;
async function loadAbnormalChart() {
    try {
        const resp = await fetch('/api/abnormal');
        const data = await resp.json();

        if (!abnormalChart) {
            abnormalChart = echarts.init(document.getElementById('chart-abnormal'));
        }

        const colorMap = {
            '满床': '#ff6b6b',
            '临时关闭': '#ffd93d',
            '维修中': '#ff9f43',
            '仅急诊入院': '#a55eea'
        };

        const seriesList = data.series.map(s => ({
            name: s.name,
            type: 'bar',
            stack: 'total',
            data: s.data,
            itemStyle: { color: colorMap[s.name] || '#90e0ef' },
            label: {
                show: true,
                position: 'top',
                color: '#fff',
                fontSize: 10,
                formatter: function(p) { return p.value > 0 ? p.value : ''; }
            }
        }));

        const option = {
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(12, 25, 41, 0.9)',
                borderColor: '#00b4d8',
                textStyle: { color: '#e0e6ed' },
                axisPointer: { type: 'shadow' }
            },
            legend: {
                textStyle: { color: '#90e0ef' },
                top: 0
            },
            grid: { left: 50, right: 30, top: 30, bottom: 30 },
            xAxis: {
                type: 'category',
                data: data.hospitals,
                axisLine: { lineStyle: { color: '#00b4d8' } },
                axisLabel: { color: '#90e0ef', rotate: 25, fontSize: 10 }
            },
            yAxis: {
                type: 'value',
                name: '病房数',
                nameTextStyle: { color: '#90e0ef' },
                axisLine: { lineStyle: { color: '#00b4d8' } },
                axisLabel: { color: '#90e0ef' },
                splitLine: { lineStyle: { color: 'rgba(0,180,216,0.1)' } }
            },
            series: seriesList
        };

        abnormalChart.setOption(option, true);
    } catch (err) {
        console.error('加载异常统计失败:', err);
    }
}

// ============ 初始化 ============

function loadAll() {
    loadKPI();
    loadHospitalChart();
    loadDeptChart();
    loadAvailableChart();
    loadAbnormalChart();
}

// 窗口大小变化时重绘图表
window.addEventListener('resize', function() {
    hospitalChart && hospitalChart.resize();
    deptChart && deptChart.resize();
    availableChart && availableChart.resize();
    abnormalChart && abnormalChart.resize();
});

// 页面加载完成后启动（兼容 script 标签在 body 底部的情况）
function initDashboard() {
    updateDateTime();
    setInterval(updateDateTime, 1000);
    loadAll();
    // 每 30 秒自动刷新
    setInterval(loadAll, 30000);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDashboard);
} else {
    initDashboard();
}
