import React from 'react';
import ReactECharts from 'echarts-for-react';

interface Props { data: { date: string; cumulative_return: number }[]; }

const CumulativeReturnChart: React.FC<Props> = ({ data }) => {
  const option = {
    title: { text: '累计收益曲线', left: 'center' },
    tooltip: { trigger: 'axis' as const },
    xAxis: { type: 'category' as const, data: data.map((d) => d.date), axisLabel: { rotate: 45, fontSize: 10 } },
    yAxis: { type: 'value' as const, axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` } },
    series: [{ name: '累计收益', type: 'line', data: data.map((d) => d.cumulative_return), smooth: true, areaStyle: { opacity: 0.1 }, lineStyle: { color: '#1677ff' } }],
    dataZoom: [{ type: 'inside' }, { type: 'slider' }],
  };
  return <ReactECharts option={option} style={{ height: 350 }} />;
};

export default CumulativeReturnChart;
