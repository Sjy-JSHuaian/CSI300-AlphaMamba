import React from 'react';
import ReactECharts from 'echarts-for-react';

interface Props {
  data: Record<string, number>;
}

const SectorPieChart: React.FC<Props> = ({ data }) => {
  const option = {
    title: { text: '行业分布', left: 'center', textStyle: { fontSize: 14 } },
    tooltip: { trigger: 'item' as const },
    series: [{
      type: 'pie',
      radius: ['40%', '70%'],
      data: Object.entries(data).map(([name, value]) => ({ name, value })),
      label: { formatter: '{b}\n{d}%' },
    }],
  };

  if (Object.keys(data).length === 0) {
    return <div style={{ textAlign: 'center', padding: 40, color: '#8c8c8c' }}>暂无行业数据</div>;
  }
  return <ReactECharts option={option} style={{ height: 280 }} />;
};

export default SectorPieChart;
