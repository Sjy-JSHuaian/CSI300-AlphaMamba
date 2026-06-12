import React, { useState, useEffect } from 'react';
import { DatePicker, Button, Table, Spin, Alert, Row, Col, Card } from 'antd';
import dayjs from 'dayjs';
import { api } from '../api/client';
import type { CompareResponse, DatesResponse } from '../types';

const ALL_MODES = ['phase6b', 'consensus', 'meta_ranker', 'phase7'];
const MODE_LABELS: Record<string, string> = {
  phase6b: 'Phase 6B', consensus: 'Consensus', meta_ranker: 'Meta Ranker', phase7: 'Phase 7',
};

const CompareTab: React.FC = () => {
  const [dates, setDates] = useState<DatesResponse | null>(null);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { api.getDates().then(setDates).catch(console.error); }, []);

  const handleCompare = async () => {
    if (!selectedDate) return;
    setLoading(true); setError(null);
    try { setResult(await api.compare(selectedDate, ALL_MODES)); }
    catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  const buildTableData = () => {
    if (!result) return [];
    const allStocks = new Map<string, Record<string, any>>();
    ALL_MODES.forEach((mode) => {
      const r = result.results[mode];
      if (r && 'portfolio' in r && r.portfolio) {
        r.portfolio.forEach((s: any) => {
          if (!allStocks.has(s.stock_id)) allStocks.set(s.stock_id, { stock_id: s.stock_id });
          const entry = allStocks.get(s.stock_id)!;
          entry[`${mode}_score`] = s.score;
          entry[`${mode}_weight`] = s.weight;
        });
      }
    });
    return Array.from(allStocks.values());
  };

  const columns = [
    { title: '股票代码', dataIndex: 'stock_id', key: 'stock_id', fixed: 'left' as const, width: 100 },
    ...ALL_MODES.map((mode) => ({
      title: MODE_LABELS[mode], key: mode, children: [
        { title: '得分', dataIndex: `${mode}_score`, key: `${mode}_score`, width: 100, render: (v: number) => v?.toFixed(4) || '-' },
        { title: '权重%', dataIndex: `${mode}_weight`, key: `${mode}_weight`, width: 90, render: (v: number) => v ? `${(v * 100).toFixed(1)}%` : '-' },
      ],
    })),
  ];

  return (
    <div>
      <div style={{ marginBottom: 16, display: 'flex', gap: 12, alignItems: 'center' }}>
        <DatePicker value={selectedDate ? dayjs(selectedDate) : null}
          onChange={(d) => setSelectedDate(d?.format('YYYY-MM-DD') || null)}
          disabledDate={(d) => !dates?.dates.includes(d.format('YYYY-MM-DD'))} placeholder="选择日期" />
        <Button type="primary" onClick={handleCompare} loading={loading}>🔬 并行对比</Button>
        {result && <span style={{ color: '#8c8c8c' }}>Bull Strength: <strong>{result.bull_strength.toFixed(3)}</strong> | {result.regime_label}</span>}
      </div>
      {error && <Alert type="error" message={error} closable style={{ marginBottom: 16 }} />}
      {loading && <Spin size="large" style={{ display: 'block', margin: '40px auto' }} />}
      {result && !loading && (
        <>
          <Card title="📊 多模式得分对比" style={{ marginBottom: 16 }}>
            <Table dataSource={buildTableData()} columns={columns} rowKey="stock_id" size="small" scroll={{ x: 900 }} pagination={{ pageSize: 20 }} />
          </Card>
          <Row gutter={16}>
            {ALL_MODES.map((mode) => {
              const r = result.results[mode];
              if (!r || 'error' in r) return null;
              return (
                <Col span={6} key={mode}>
                  <Card size="small" title={MODE_LABELS[mode]}>
                    {(r as any).portfolio?.slice(0, 5).map((s: any, i: number) => (
                      <div key={s.stock_id} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}>
                        <span>{['🥇','🥈','🥉','4','5'][i]} {s.stock_id}</span>
                        <span>{(s.weight * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </Card>
                </Col>
              );
            })}
          </Row>
        </>
      )}
    </div>
  );
};

export default CompareTab;
