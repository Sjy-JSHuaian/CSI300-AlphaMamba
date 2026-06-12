import React, { useState, useEffect } from 'react';
import { DatePicker, Select, Button, Spin, Alert, Row, Col } from 'antd';
import dayjs from 'dayjs';
import { api } from '../api/client';
import type { PredictResponse, MarketStatus, DatesResponse } from '../types';
import MarketStatusCard from './MarketStatusCard';
import StockCardGrid from './StockCardGrid';
import SectorPieChart from './SectorPieChart';

const MODES = [
  { value: 'phase6b', label: 'Phase 6B (Bull/NonBull)' },
  { value: 'consensus', label: 'Consensus' },
  { value: 'meta_ranker', label: 'Meta Ranker V2' },
  { value: 'phase7', label: 'Phase 7 (Safe Mode)' },
];

const PredictTab: React.FC = () => {
  const [dates, setDates] = useState<DatesResponse | null>(null);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [mode, setMode] = useState('phase6b');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [marketStatus, setMarketStatus] = useState<MarketStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getDates().then(setDates).catch(console.error);
  }, []);

  const handlePredict = async () => {
    if (!selectedDate) return;
    setLoading(true);
    setError(null);
    try {
      const [status, predResult] = await Promise.all([
        api.getStatus(selectedDate),
        api.predict({ date: selectedDate, mode }),
      ]);
      setMarketStatus(status);
      setResult(predResult);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col>
          <DatePicker
            value={selectedDate ? dayjs(selectedDate) : null}
            onChange={(d) => setSelectedDate(d?.format('YYYY-MM-DD') || null)}
            disabledDate={(d) => !dates?.dates.includes(d.format('YYYY-MM-DD'))}
            placeholder="选择预测日期"
            style={{ width: 200 }}
          />
        </Col>
        <Col>
          <Select value={mode} onChange={setMode} options={MODES} style={{ width: 220 }} />
        </Col>
        <Col>
          <Button type="primary" onClick={handlePredict} loading={loading} size="large">
            🚀 运行预测
          </Button>
        </Col>
      </Row>

      {error && <Alert type="error" message={error} closable style={{ marginBottom: 16 }} />}

      {marketStatus && (
        <MarketStatusCard
          bullStrength={marketStatus.bull_strength}
          regimeLabel={marketStatus.regime_label}
          nStocks={marketStatus.n_stocks}
        />
      )}

      {loading && <Spin size="large" style={{ display: 'block', margin: '40px auto' }} />}

      {result && !loading && (
        <>
          <StockCardGrid stocks={result.portfolio} />
          <Row gutter={16} style={{ marginTop: 24 }}>
            <Col span={12}>
              <SectorPieChart data={result.sector_distribution} />
            </Col>
            <Col span={12}>
              <div style={{ background: '#fafafa', padding: 16, borderRadius: 8 }}>
                <div style={{ marginBottom: 12 }}><strong>📈 得分统计</strong></div>
                {Object.entries(result.score_distribution).map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0' }}>
                    <span style={{ color: '#8c8c8c' }}>{k}</span>
                    <span>{typeof v === 'number' ? v.toFixed(4) : String(v)}</span>
                  </div>
                ))}
                {result.consensus_overlap > 0 && (
                  <div style={{ marginTop: 12 }}>
                    <strong>🤝 共识重叠: </strong><span>{result.consensus_overlap} 只股票</span>
                  </div>
                )}
              </div>
            </Col>
          </Row>
        </>
      )}
    </div>
  );
};

export default PredictTab;
