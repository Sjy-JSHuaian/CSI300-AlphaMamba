import React, { useState, useEffect } from 'react';
import { Table, Spin, Alert, Tag } from 'antd';
import { api } from '../api/client';
import type { HistoryRecord, HistoryResponse } from '../types';

const HistoryTab: React.FC = () => {
  const [data, setData] = useState<HistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getHistory(50).then(setData).catch((e) => setError(e.message)).finally(() => setLoading(false));
  }, []);

  if (loading) return <Spin size="large" style={{ display: 'block', margin: '40px auto' }} />;
  if (error) return <Alert type="error" message={error} />;

  const columns = [
    { title: '日期', dataIndex: 'date', key: 'date', width: 120 },
    { title: '市场状态', key: 'market', width: 180,
      render: (_: any, r: HistoryRecord) => (
        <><Tag color={r.bull_strength > 0.5 ? 'red' : 'blue'}>BS: {r.bull_strength.toFixed(3)}</Tag><Tag>{r.regime_label}</Tag></>
      ),
    },
    { title: '持仓', key: 'holdings',
      render: (_: any, r: HistoryRecord) => (
        r.stocks?.map((s, i) => <Tag key={s} color="green">{s} ({(r.weights[i] * 100).toFixed(1)}%)</Tag>)
      ),
    },
  ];

  return (
    <div>
      <Table dataSource={data?.history || []} columns={columns} rowKey="date" size="small" pagination={{ pageSize: 20 }}
        expandable={{
          expandedRowRender: (r: HistoryRecord) => (
            <div>{r.stocks?.map((s, i) => (<div key={s} style={{ padding: '4px 0' }}>{['🥇','🥈','🥉','4','5'][i] || `#${i+1}`} {s} — {(r.weights[i] * 100).toFixed(1)}%</div>))}</div>
          ),
        }}
      />
    </div>
  );
};

export default HistoryTab;
