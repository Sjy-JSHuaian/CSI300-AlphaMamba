import React from 'react';
import { Card, Progress, Tag, Space } from 'antd';
import { RiseOutlined, FallOutlined, MinusOutlined } from '@ant-design/icons';

interface Props {
  bullStrength: number;
  regimeLabel: string;
  nStocks: number;
}

const MarketStatusCard: React.FC<Props> = ({ bullStrength, regimeLabel, nStocks }) => {
  const getStatus = () => {
    if (bullStrength > 0.8) return { icon: <RiseOutlined />, color: '#cf1322', text: '强牛市' };
    if (bullStrength > 0.5) return { icon: <RiseOutlined />, color: '#fa8c16', text: '牛市' };
    if (bullStrength > 0.2) return { icon: <MinusOutlined />, color: '#8c8c8c', text: '震荡市' };
    return { icon: <FallOutlined />, color: '#1890ff', text: '弱势/熊市' };
  };

  const status = getStatus();

  return (
    <Card size="small" title="🌡️ 市场状态" style={{ marginBottom: 16 }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ color: status.color, fontSize: 18 }}>{status.icon}</span>
          <span style={{ fontWeight: 600, fontSize: 16 }}>{status.text}</span>
          <Tag color="blue">{regimeLabel}</Tag>
        </div>
        <Progress
          percent={Math.round(bullStrength * 100)}
          strokeColor={{ '0%': '#1890ff', '50%': '#fa8c16', '100%': '#cf1322' }}
          format={(p) => `Bull Strength: ${(p! / 100).toFixed(3)}`}
        />
        <span style={{ color: '#8c8c8c', fontSize: 12 }}>可用股票: {nStocks} 只</span>
      </Space>
    </Card>
  );
};

export default MarketStatusCard;
