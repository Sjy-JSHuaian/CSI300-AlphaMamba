import React from 'react';
import { Row, Col, Card, Statistic } from 'antd';
import { RiseOutlined, TrophyOutlined, FallOutlined, PercentageOutlined } from '@ant-design/icons';

interface Props { metrics: Record<string, number>; }

const MetricsCardRow: React.FC<Props> = ({ metrics }) => (
  <Row gutter={16} style={{ marginBottom: 16 }}>
    <Col span={6}><Card><Statistic title="累计收益" value={(metrics.cumulative_return * 100).toFixed(2)} suffix="%" prefix={<RiseOutlined />} /></Card></Col>
    <Col span={6}><Card><Statistic title="夏普比率" value={metrics.sharpe} prefix={<TrophyOutlined />} /></Card></Col>
    <Col span={6}><Card><Statistic title="最大回撤" value={(metrics.max_drawdown * 100).toFixed(2)} suffix="%" prefix={<FallOutlined />} valueStyle={{ color: '#cf1322' }} /></Card></Col>
    <Col span={6}><Card><Statistic title="胜率" value={(metrics.win_rate * 100).toFixed(1)} suffix="%" prefix={<PercentageOutlined />} /></Card></Col>
  </Row>
);

export default MetricsCardRow;
