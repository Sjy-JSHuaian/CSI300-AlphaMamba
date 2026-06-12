import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Button, DatePicker, Select, Spin, Alert } from 'antd';
import dayjs from 'dayjs';
import { api } from '../api/client';
import type { PredictResponse, DatesResponse } from '../types';
import ParamSliders, { ParamValues, DEFAULT_PARAMS } from './ParamSliders';
import StockCardGrid from './StockCardGrid';
import SectorPieChart from './SectorPieChart';

const MODES = [
  { value: 'phase6b', label: 'Phase 6B (Bull/NonBull)' },
  { value: 'consensus', label: 'Consensus' },
  { value: 'meta_ranker', label: 'Meta Ranker V2' },
  { value: 'phase7', label: 'Phase 7 (Safe Mode)' },
];

const TuningTab: React.FC = () => {
  const [dates, setDates] = useState<DatesResponse | null>(null);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [mode, setMode] = useState('phase6b');
  const [params, setParams] = useState<ParamValues>(DEFAULT_PARAMS);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { api.getDates().then(setDates).catch(console.error); }, []);

  const handleRun = async () => {
    if (!selectedDate) return;
    setLoading(true); setError(null);
    try {
      const r = await api.predict({ date: selectedDate, mode, ...params });
      setResult(r);
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  return (
    <Row gutter={24}>
      <Col span={10}>
        <Card title="🎛️ 参数配置" extra={<Button onClick={() => { setParams(DEFAULT_PARAMS); setResult(null); }} size="small">重置</Button>}>
          <div style={{ marginBottom: 16, display: 'flex', gap: 12 }}>
            <DatePicker
              value={selectedDate ? dayjs(selectedDate) : null}
              onChange={(d) => setSelectedDate(d?.format('YYYY-MM-DD') || null)}
              disabledDate={(d) => !dates?.dates.includes(d.format('YYYY-MM-DD'))}
              placeholder="选择日期" style={{ flex: 1 }}
            />
            <Select value={mode} onChange={setMode} options={MODES} style={{ width: 200 }} />
          </div>
          <ParamSliders values={params} onChange={setParams} />
          <Button type="primary" onClick={handleRun} loading={loading} block size="large" style={{ marginTop: 16 }}>
            🔄 应用参数并重跑
          </Button>
        </Card>
      </Col>
      <Col span={14}>
        {error && <Alert type="error" message={error} closable style={{ marginBottom: 16 }} />}
        {loading && <Spin size="large" style={{ display: 'block', margin: '80px auto' }} />}
        {result && !loading && (
          <>
            <Card size="small" title={`📈 结果 · Bull Strength: ${result.bull_strength.toFixed(3)}`} style={{ marginBottom: 16 }}>
              <StockCardGrid stocks={result.portfolio} />
            </Card>
            <SectorPieChart data={result.sector_distribution} />
          </>
        )}
      </Col>
    </Row>
  );
};

export default TuningTab;
