import React, { useState, useEffect } from 'react';
import { DatePicker, Select, Button, Spin, Alert } from 'antd';
import dayjs from 'dayjs';
import { api } from '../api/client';
import type { BacktestResponse, DatesResponse } from '../types';
import MetricsCardRow from './MetricsCardRow';
import CumulativeReturnChart from './CumulativeReturnChart';

const BacktestTab: React.FC = () => {
  const [dates, setDates] = useState<DatesResponse | null>(null);
  const [startDate, setStartDate] = useState<string | null>(null);
  const [endDate, setEndDate] = useState<string | null>(null);
  const [mode, setMode] = useState('phase6b');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BacktestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getDates().then((d) => { setDates(d); setStartDate(d.first); setEndDate(d.last); }).catch(console.error);
  }, []);

  const handleRun = async () => {
    if (!startDate || !endDate) return;
    setLoading(true); setError(null);
    try { setResult(await api.runBacktest({ start_date: startDate, end_date: endDate, mode })); }
    catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  return (
    <div>
      <div style={{ marginBottom: 16, display: 'flex', gap: 12, alignItems: 'center' }}>
        <DatePicker value={startDate ? dayjs(startDate) : null} onChange={(d) => setStartDate(d?.format('YYYY-MM-DD') || null)} placeholder="起始日期" />
        <span>至</span>
        <DatePicker value={endDate ? dayjs(endDate) : null} onChange={(d) => setEndDate(d?.format('YYYY-MM-DD') || null)} placeholder="结束日期" />
        <Select value={mode} onChange={setMode} options={[
          { value: 'phase6b', label: 'Phase 6B' }, { value: 'consensus', label: 'Consensus' },
          { value: 'meta_ranker', label: 'Meta Ranker' }, { value: 'phase7', label: 'Phase 7' },
        ]} style={{ width: 160 }} />
        <Button type="primary" onClick={handleRun} loading={loading}>📈 运行回测</Button>
      </div>
      {error && <Alert type="error" message={error} closable style={{ marginBottom: 16 }} />}
      {loading && <Spin size="large" style={{ display: 'block', margin: '40px auto' }} />}
      {result && !loading && (
        <>
          <MetricsCardRow metrics={result.metrics} />
          <CumulativeReturnChart data={result.daily_returns} />
        </>
      )}
    </div>
  );
};

export default BacktestTab;
