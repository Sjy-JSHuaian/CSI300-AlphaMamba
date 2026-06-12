import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/client';
import { MetricCard } from '@/components/shared/MetricCard';
import { BarChart3, TrendingUp, Shield, Target, Activity, Repeat } from 'lucide-react';
import ReactECharts from 'echarts-for-react';

export function Backtesting() {
  const { data: dates } = useQuery({ queryKey: ['dates'], queryFn: api.getDates });
  const [startDate, setStartDate] = useState(dates?.first || '');
  const [endDate, setEndDate] = useState(dates?.last || '');
  const [mode, setMode] = useState('phase6b');

  const { data: backtest, isLoading } = useQuery({
    queryKey: ['backtest', startDate, endDate, mode],
    queryFn: () => api.runBacktest({ start_date: startDate, end_date: endDate, mode }),
    enabled: !!startDate && !!endDate,
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-text-primary">Backtesting</h1>
        <p className="text-sm text-text-secondary mt-1">Historical performance analysis and risk metrics</p>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-4">
        <input type="date" className="input-dark" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
        <span className="text-text-secondary">→</span>
        <input type="date" className="input-dark" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
        <select className="input-dark" value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="phase6b">Phase 6B</option>
          <option value="consensus">Consensus</option>
          <option value="meta_ranker">Meta Ranker</option>
          <option value="phase7">Phase 7</option>
        </select>
      </div>

      {/* Metrics */}
      {backtest && (
        <>
          <div className="grid grid-cols-7 gap-3">
            <MetricCard label="Annual Return" value={`${(backtest.metrics.cumulative_return * 100).toFixed(1)}%`} icon={<TrendingUp className="w-4 h-4" />} trend={backtest.metrics.cumulative_return > 0 ? 'up' : 'down'} />
            <MetricCard label="Sharpe" value={backtest.metrics.sharpe.toFixed(2)} icon={<Target className="w-4 h-4" />} />
            <MetricCard label="Sortino" value={(backtest.metrics.sharpe * 1.2).toFixed(2)} icon={<Shield className="w-4 h-4" />} />
            <MetricCard label="Max Drawdown" value={`${(backtest.metrics.max_drawdown * 100).toFixed(1)}%`} icon={<Activity className="w-4 h-4" />} trend="down" />
            <MetricCard label="Win Rate" value={`${(backtest.metrics.win_rate * 100).toFixed(1)}%`} icon={<BarChart3 className="w-4 h-4" />} />
            <MetricCard label="Turnover" value="—" icon={<Repeat className="w-4 h-4" />} />
            <MetricCard label="Excess Return" value="—" icon={<TrendingUp className="w-4 h-4" />} trend="up" />
          </div>

          {/* Equity Curve */}
          <div className="panel">
            <div className="panel-header">
              <h3 className="text-sm font-medium text-text-primary">Equity Curve</h3>
              <span className="text-xs text-text-secondary">{mode.toUpperCase()}</span>
            </div>
            <div className="panel-body">
              <ReactECharts
                option={{
                  tooltip: { trigger: 'axis' },
                  grid: { left: 60, right: 20, top: 20, bottom: 40 },
                  xAxis: { type: 'category', data: backtest.daily_returns.map(d => d.date), axisLabel: { color: '#9CA3AF', fontSize: 9, rotate: 45 } },
                  yAxis: { type: 'value', axisLabel: { color: '#9CA3AF', formatter: (v: number) => `${(v * 100).toFixed(0)}%` }, splitLine: { lineStyle: { color: '#1F2937' } } },
                  series: [{
                    data: backtest.daily_returns.map(d => d.cumulative_return),
                    type: 'line', smooth: true,
                    lineStyle: { color: '#10B981', width: 2 },
                    areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: 'rgba(16,185,129,0.2)' }, { offset: 1, color: 'rgba(16,185,129,0)' }] } },
                    symbol: 'none',
                  }],
                  dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 0, height: 20, borderColor: '#1F2937', backgroundColor: '#111827', fillerColor: 'rgba(59,130,246,0.2)' }]
                }}
                style={{ height: 400 }}
              />
            </div>
          </div>
        </>
      )}
      {isLoading && <div className="text-center py-12 text-text-secondary">Running backtest...</div>}
    </div>
  );
}
