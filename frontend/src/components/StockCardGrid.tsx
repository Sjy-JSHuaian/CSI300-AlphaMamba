import React from 'react';
import { Card, Progress, Row, Col } from 'antd';
import type { StockResult } from '../types';

interface Props {
  stocks: StockResult[];
}

const medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'];

const StockCardGrid: React.FC<Props> = ({ stocks }) => (
  <Row gutter={[12, 12]}>
    {stocks.map((stock, i) => (
      <Col xs={24} sm={12} md={Math.floor(24 / Math.min(stocks.length || 1, 5))} key={stock.stock_id}>
        <Card
          size="small"
          hoverable
          title={
            <span>
              <span style={{ fontSize: 20, marginRight: 8 }}>{medals[i] || `#${i + 1}`}</span>
              <strong>{stock.stock_id}</strong>
            </span>
          }
        >
          <Progress
            percent={Math.round(stock.weight * 100)}
            size="small"
            strokeColor={i === 0 ? '#cf1322' : i === 1 ? '#fa8c16' : '#1677ff'}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8 }}>
            <span style={{ color: '#8c8c8c' }}>权重</span>
            <strong>{(stock.weight * 100).toFixed(1)}%</strong>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: '#8c8c8c' }}>得分</span>
            <span>{stock.score.toFixed(4)}</span>
          </div>
        </Card>
      </Col>
    ))}
  </Row>
);

export default StockCardGrid;
