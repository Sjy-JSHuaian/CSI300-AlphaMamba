import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/client';
import { getScoreColor, getRegimeLabel } from '@/lib/utils';
import { Search, ArrowUpDown, TrendingUp, Shield, Target } from 'lucide-react';
import ReactECharts from 'echarts-for-react';

export function Predictions() {
  const [selectedDate, setSelectedDate] = useState<string>('');
  const [mode, setMode] = useState('phase6b');
  const [searchTerm, setSearchTerm] = useState('');
  const [sortKey, setSortKey] = useState('score');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');

  const { data: dates } = useQuery({ queryKey: ['dates'], queryFn: api.getDates });

  const { data: prediction, isLoading, refetch } = useQuery({
    queryKey: ['predict', selectedDate || dates?.last, mode],
    queryFn: () => api.predict({
      date: selectedDate || dates?.last || '',
      mode,
    }),
    enabled: !!dates?.last || !!selectedDate,
  });

  const { data: comparison } = useQuery({
    queryKey: ['compare', selectedDate || dates?.last],
    queryFn: () => api.compare(selectedDate || dates?.last || '', ['phase6b', 'consensus', 'meta_ranker', 'phase7']),
    enabled: !!dates?.last || !!selectedDate,
  });

  const regime = getRegimeLabel(prediction?.bull_strength || 0);

  const handleSort = (key: string) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('desc'); }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-text-primary">Predictions</h1>
          <p className="text-sm text-text-secondary mt-1">Model predictions and stock ranking analysis</p>
        </div>
        <div className="flex items-center gap-3">
          <span className={`text-xs px-3 py-1 rounded-full ${regime.color} bg-surface-elevated border border-border`}>
            {regime.text} · BS {prediction?.bull_strength?.toFixed(3) || '...'}
          </span>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-4 flex-wrap">
        <select
          className="input-dark"
          value={selectedDate || dates?.last || ''}
          onChange={(e) => setSelectedDate(e.target.value)}
        >
          {dates?.dates.slice(-60).reverse().map(d => (
            <option key={d} value={d}>{d}</option>
          ))}
        </select>
        <select className="input-dark" value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="phase6b">Phase 6B (Bull/NonBull)</option>
          <option value="consensus">Consensus</option>
          <option value="meta_ranker">Meta Ranker V2</option>
          <option value="phase7">Phase 7 (Safe Mode)</option>
        </select>
        <div className="relative">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-text-secondary" />
          <input
            className="input-dark pl-9 w-48"
            placeholder="Search ticker..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>

      {/* Predictions Table */}
      <div className="panel">
        <div className="panel-header">
          <div className="flex items-center gap-2">
            <Target className="w-4 h-4 text-accent" />
            <h3 className="text-sm font-medium text-text-primary">Stock Rankings</h3>
          </div>
          <span className="text-xs text-text-secondary">
            {prediction?.portfolio.length || 0} selected · {prediction?.mode?.toUpperCase()}
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th className="cursor-pointer" onClick={() => handleSort('rank')}>Rank <ArrowUpDown className="w-3 h-3 inline ml-1" /></th>
                <th>Ticker</th>
                <th className="cursor-pointer" onClick={() => handleSort('score')}>Prediction Score <ArrowUpDown className="w-3 h-3 inline ml-1" /></th>
                <th className="cursor-pointer" onClick={() => handleSort('weight')}>Weight <ArrowUpDown className="w-3 h-3 inline ml-1" /></th>
                <th>Expected Return</th>
                <th>Confidence</th>
                <th>Risk Score</th>
                <th>Sector</th>
                <th>Bull Signal</th>
                <th>NonBull Signal</th>
              </tr>
            </thead>
            <tbody>
              {prediction?.portfolio
                .filter(s => s.stock_id.includes(searchTerm))
                .map((stock, i) => {
                  const score = stock.score;
                  const confidence = score > 0.8 ? 'High' : score > 0.6 ? 'Medium' : 'Low';
                  const riskScore = (1 - score).toFixed(2);
                  const expReturn = (score * 0.15).toFixed(2);
                  return (
                    <tr key={stock.stock_id} className="cursor-pointer hover:bg-surface-elevated">
                      <td className="text-text-secondary">#{i + 1}</td>
                      <td className="font-medium text-accent">{stock.stock_id}</td>
                      <td className={getScoreColor(score)}>{score.toFixed(4)}</td>
                      <td>{(stock.weight * 100).toFixed(1)}%</td>
                      <td className="text-positive">+{expReturn}%</td>
                      <td>
                        <span className={`text-xs px-2 py-0.5 rounded ${
                          confidence === 'High' ? 'bg-positive/10 text-positive' :
                          confidence === 'Medium' ? 'bg-accent/10 text-accent' :
                          'bg-warning/10 text-warning'
                        }`}>{confidence}</span>
                      </td>
                      <td className="text-text-secondary">{riskScore}</td>
                      <td className="text-text-secondary">{stock.sector_id}</td>
                      <td>{prediction?.bull_top5?.includes(stock.stock_id) ? '✓' : '—'}</td>
                      <td>{prediction?.nonbull_top5?.includes(stock.stock_id) ? '✓' : '—'}</td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
          {isLoading && (
            <div className="text-center py-12 text-text-secondary">Loading predictions...</div>
          )}
          {prediction?.portfolio.length === 0 && !isLoading && (
            <div className="text-center py-12">
              <Shield className="w-8 h-8 text-text-secondary mx-auto mb-2" />
              <p className="text-text-secondary">No active positions</p>
              <p className="text-xs text-text-secondary mt-1">Bull Gate triggered or insufficient data</p>
            </div>
          )}
        </div>
      </div>

      {/* Multi-Mode Comparison */}
      {comparison && (
        <div className="panel">
          <div className="panel-header">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-accent" />
              <h3 className="text-sm font-medium text-text-primary">Multi-Model Consensus</h3>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Phase 6B</th>
                  <th>Consensus</th>
                  <th>Meta Ranker</th>
                  <th>Phase 7</th>
                  <th>Avg Score</th>
                </tr>
              </thead>
              <tbody>
                {prediction?.portfolio.map(stock => {
                  const scores = ['phase6b', 'consensus', 'meta_ranker', 'phase7']
                    .map(m => comparison.results[m]?.portfolio?.find((s: any) => s.stock_id === stock.stock_id)?.score || 0);
                  const avg = scores.reduce((a: number, b: number) => a + b, 0) / 4;
                  return (
                    <tr key={stock.stock_id}>
                      <td className="font-medium text-accent">{stock.stock_id}</td>
                      {scores.map((s: number, i: number) => (
                        <td key={i} className={getScoreColor(s)}>{s ? s.toFixed(4) : '—'}</td>
                      ))}
                      <td className={getScoreColor(avg)}>{avg.toFixed(4)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
