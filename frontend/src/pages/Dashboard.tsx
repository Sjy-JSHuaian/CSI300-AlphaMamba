import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/client';
import { MetricCard } from '@/components/shared/MetricCard';
import { getRegimeLabel, getScoreColor } from '@/lib/utils';
import { TrendingUp, TrendingDown, Activity, BarChart3, Zap, Target, Shield, Layers } from 'lucide-react';
import ReactECharts from 'echarts-for-react';
import { useState } from 'react';

export function Dashboard() {
  const [selectedDate, setSelectedDate] = useState<string | null>(null);

  const { data: dates, isLoading: datesLoading } = useQuery({
    queryKey: ['dates'],
    queryFn: api.getDates,
  });

  const { data: status, isLoading: statusLoading } = useQuery({
    queryKey: ['status', selectedDate || dates?.last],
    queryFn: () => api.getStatus(selectedDate || dates?.last || ''),
    enabled: !!dates?.last || !!selectedDate,
  });

  const { data: prediction, isLoading: predLoading } = useQuery({
    queryKey: ['predict', selectedDate || dates?.last],
    queryFn: () => api.predict({ date: selectedDate || dates?.last || '', mode: 'phase6b' }),
    enabled: !!dates?.last || !!selectedDate,
  });

  const regime = getRegimeLabel(status?.bull_strength || 0);

  // Market regime gauge chart
  const gaugeOption = {
    series: [{
      type: 'gauge',
      startAngle: 210, endAngle: -30,
      center: ['50%', '60%'],
      radius: '90%',
      min: 0, max: 1,
      splitNumber: 10,
      axisLine: {
        show: true,
        lineStyle: {
          width: 18,
          color: [[0.3, '#EF4444'], [0.5, '#F59E0B'], [0.8, '#3B82F6'], [1, '#10B981']]
        }
      },
      pointer: { length: '60%', width: 6, itemStyle: { color: '#F9FAFB' } },
      detail: {
        valueAnimation: true,
        formatter: (v: number) => `BS: ${v.toFixed(3)}`,
        color: '#F9FAFB', fontSize: 18, fontFamily: 'Geist',
        offsetCenter: [0, '75%']
      },
      data: [{ value: status?.bull_strength || 0.5 }]
    }]
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-text-primary">Dashboard</h1>
          <p className="text-sm text-text-secondary mt-1">Market overview and research summary</p>
        </div>
        <div className="flex items-center gap-3">
          {datesLoading ? (
            <span className="text-xs text-text-secondary">Loading dates...</span>
          ) : dates?.dates ? (
            <>
              <select
                className="input-dark text-sm max-w-[180px]"
                value={selectedDate || dates.last || ''}
                onChange={(e) => setSelectedDate(e.target.value)}
              >
                {Object.entries(
                  dates.dates.reduce((groups: Record<string, string[]>, d: string) => {
                    const ym = d.slice(0, 7);
                    if (!groups[ym]) groups[ym] = [];
                    groups[ym].push(d);
                    return groups;
                  }, {})
                )
                  .reverse()
                  .map(([ym, ds]) => (
                    <optgroup key={ym} label={ym}>
                      {ds.reverse().map(d => (
                        <option key={d} value={d}>{d}</option>
                      ))}
                    </optgroup>
                  ))}
              </select>
              <span className="text-xs text-text-secondary">
                {dates.first} ~ {dates.last} · {dates.count} days
              </span>
            </>
          ) : (
            <span className="text-xs text-negative">Backend not connected</span>
          )}
        </div>
      </div>

      {/* Market Overview Cards */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          label="Market Regime"
          value={regime.text}
          trend={status?.bull_strength && status.bull_strength > 0.5 ? 'up' : 'down'}
          subtitle={`Bull Strength: ${status?.bull_strength?.toFixed(3) || '...'}`}
          icon={<Activity className="w-4 h-4" />}
        />
        <MetricCard
          label="Active Stocks"
          value={status?.n_stocks || '...'}
          subtitle="CSI 300 Constituents"
          icon={<Layers className="w-4 h-4" />}
        />
        <MetricCard
          label="Prediction Coverage"
          value={prediction ? `${prediction.portfolio.length}/5` : '...'}
          subtitle="Top-5 Selected"
          icon={<Target className="w-4 h-4" />}
        />
        <MetricCard
          label="Consensus Overlap"
          value={prediction?.consensus_overlap || '...'}
          subtitle="Bull ∩ NonBull"
          icon={<Zap className="w-4 h-4" />}
        />
      </div>

      {/* Chart + Top Picks Row */}
      <div className="grid grid-cols-3 gap-6">
        {/* Gauge Chart */}
        <div className="panel">
          <div className="panel-header">
            <h3 className="text-sm font-medium text-text-primary">Market Regime Monitor</h3>
          </div>
          <div className="panel-body" style={{ height: 280 }}>
            <ReactECharts option={gaugeOption} style={{ height: '100%' }} />
          </div>
        </div>

        {/* Top Picks */}
        <div className="col-span-2 panel">
          <div className="panel-header">
            <h3 className="text-sm font-medium text-text-primary">Today's Top Picks</h3>
            <span className="text-xs text-text-secondary">
              {prediction?.mode?.toUpperCase() || '...'} Model
            </span>
          </div>
          <div className="panel-body">
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Ticker</th>
                  <th>Score</th>
                  <th>Weight</th>
                  <th>Bull Score</th>
                  <th>NonBull Score</th>
                  <th>Sector ID</th>
                </tr>
              </thead>
              <tbody>
                {prediction?.portfolio.map((stock, i) => (
                  <tr key={stock.stock_id}>
                    <td className="text-text-secondary">#{i + 1}</td>
                    <td className="font-medium">{stock.stock_id}</td>
                    <td className={getScoreColor(stock.score)}>{stock.score.toFixed(4)}</td>
                    <td>{(stock.weight * 100).toFixed(1)}%</td>
                    <td className="text-text-secondary">{prediction.bull_top5.includes(stock.stock_id) ? '✓' : '—'}</td>
                    <td className="text-text-secondary">{prediction.nonbull_top5.includes(stock.stock_id) ? '✓' : '—'}</td>
                    <td className="text-text-secondary">{stock.sector_id}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {prediction && prediction.portfolio.length === 0 && (
              <p className="text-sm text-text-secondary text-center py-8">No active positions (Bull Gate triggered)</p>
            )}
          </div>
        </div>
      </div>

      {/* Score Distribution + Sector */}
      <div className="grid grid-cols-2 gap-6">
        <div className="panel">
          <div className="panel-header">
            <h3 className="text-sm font-medium text-text-primary">Score Distribution</h3>
          </div>
          <div className="panel-body">
            {prediction?.score_distribution && Object.keys(prediction.score_distribution).length > 0 ? (
              <div className="grid grid-cols-2 gap-4">
                {Object.entries(prediction.score_distribution).map(([key, val]) => (
                  <div key={key} className="flex justify-between py-2 border-b border-border/50">
                    <span className="text-sm text-text-secondary capitalize">{key}</span>
                    <span className="text-sm font-number text-text-primary">{typeof val === 'number' ? val.toFixed(4) : val}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-text-secondary text-center py-8">No data available</p>
            )}
          </div>
        </div>
        <div className="panel">
          <div className="panel-header">
            <h3 className="text-sm font-medium text-text-primary">Sector Allocation</h3>
          </div>
          <div className="panel-body">
            {prediction?.sector_distribution && Object.keys(prediction.sector_distribution).length > 0 ? (
              <div className="space-y-3">
                {Object.entries(prediction.sector_distribution).map(([sector, count]) => (
                  <div key={sector} className="flex items-center gap-3">
                    <span className="text-xs text-text-secondary w-16">Sector {sector}</span>
                    <div className="flex-1 h-4 bg-background rounded-full overflow-hidden">
                      <div
                        className="h-full bg-accent rounded-full transition-all"
                        style={{ width: `${(Number(count) / prediction.portfolio.length) * 100}%` }}
                      />
                    </div>
                    <span className="text-xs font-number text-text-primary">{count} stocks</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-text-secondary text-center py-8">No sector data available</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
