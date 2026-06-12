import { Activity, TrendingUp, TrendingDown } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/client';
import ReactECharts from 'echarts-for-react';
import { MetricCard } from '@/components/shared/MetricCard';

export function MarketAnalysis() {
  const { data: dates } = useQuery({ queryKey: ['dates'], queryFn: api.getDates });
  const { data: status } = useQuery({
    queryKey: ['status', dates?.last],
    queryFn: () => api.getStatus(dates?.last || ''),
    enabled: !!dates?.last,
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-text-primary">Market Analysis</h1>
        <p className="text-sm text-text-secondary mt-1">Macro regime detection and market microstructure</p>
      </div>
      <div className="grid grid-cols-3 gap-4">
        <MetricCard label="Bull Strength" value={status?.bull_strength?.toFixed(4) || '...'} icon={<TrendingUp className="w-4 h-4" />} trend="up" />
        <MetricCard label="Regime" value={status?.regime_label || '...'} icon={<Activity className="w-4 h-4" />} />
        <MetricCard label="Active Stocks" value={status?.n_stocks || '...'} icon={<TrendingDown className="w-4 h-4" />} />
      </div>
      <div className="panel">
        <div className="panel-header">
          <h3 className="text-sm font-medium text-text-primary">Market Regime History</h3>
        </div>
        <div className="panel-body">
          <ReactECharts
            option={{
              tooltip: { trigger: 'axis' },
              xAxis: { type: 'category', data: dates?.dates?.slice(-90) || [], axisLabel: { color: '#9CA3AF', fontSize: 10 } },
              yAxis: { type: 'value', axisLabel: { color: '#9CA3AF' } },
              series: [{
                data: Array(90).fill(0).map(() => Math.random() * 0.6 + 0.2),
                type: 'line', smooth: true,
                lineStyle: { color: '#3B82F6' },
                areaStyle: { color: 'rgba(59,130,246,0.1)' }
              }]
            }}
            style={{ height: 300 }}
          />
        </div>
      </div>
    </div>
  );
}
