import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/client';
import { MetricCard } from '@/components/shared/MetricCard';
import ReactECharts from 'echarts-for-react';
import { TrendingUp, Target, BarChart3, Layers, Activity } from 'lucide-react';

export function Performance() {
  const { t } = useTranslation();

  const { data, isLoading, error } = useQuery({
    queryKey: ['performance'],
    queryFn: api.getPerformance,
  });

  const metrics = useMemo(() => {
    if (!data) return null;
    return {
      accuracy: data.accuracy ?? data.accuracy_metrics?.accuracy ?? 0,
      hit_rate: data.hit_rate ?? data.accuracy_metrics?.hit_rate ?? 0,
      mean_score: data.mean_score ?? data.accuracy_metrics?.mean_score ?? 0,
      total_snapshots: data.total_snapshots ?? data.total ?? 0,
      accuracy_trend: data.accuracy_trend ?? data.trends?.accuracy ?? [],
      rolling_sharpe: data.rolling_sharpe ?? data.trends?.sharpe ?? [],
      ranking_stability: data.ranking_stability ?? data.trends?.stability ?? [],
      bull_strength_over_time: data.bull_strength_over_time ?? data.trends?.bull_strength ?? [],
    };
  }, [data]);

  const accuracyOption = useMemo(() => {
    if (!metrics?.accuracy_trend?.length) return undefined;
    const dates = metrics.accuracy_trend.map((d: any) => d.date || d.x || '');
    const values = metrics.accuracy_trend.map((d: any) => d.value ?? d.y ?? d.accuracy ?? 0);
    return {
      backgroundColor: 'transparent',
      grid: { left: 55, right: 20, top: 20, bottom: 40 },
      xAxis: {
        type: 'category',
        data: dates,
        axisLabel: { color: '#9CA3AF', fontSize: 10, rotate: 45 },
        axisLine: { lineStyle: { color: '#374151' } },
      },
      yAxis: {
        type: 'value',
        name: t('performance.accuracy'),
        nameTextStyle: { color: '#9CA3AF', fontSize: 10 },
        axisLabel: { color: '#9CA3AF', fontSize: 10 },
        splitLine: { lineStyle: { color: '#1F2937' } },
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#111827',
        borderColor: '#374151',
        textStyle: { color: '#F9FAFB', fontSize: 12 },
      },
      series: [
        {
          type: 'line',
          data: values,
          smooth: true,
          symbol: 'circle',
          symbolSize: 4,
          lineStyle: { color: '#10B981', width: 2 },
          itemStyle: { color: '#10B981' },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(16, 185, 129, 0.25)' },
                { offset: 1, color: 'rgba(16, 185, 129, 0.02)' },
              ],
            },
          },
        },
      ],
    };
  }, [metrics, t]);

  const stabilityOption = useMemo(() => {
    if (!metrics?.ranking_stability?.length) return undefined;
    const dates = metrics.ranking_stability.map((d: any) => d.date || d.x || '');
    const values = metrics.ranking_stability.map((d: any) => d.value ?? d.y ?? d.stability ?? 0);
    return {
      backgroundColor: 'transparent',
      grid: { left: 55, right: 20, top: 20, bottom: 40 },
      xAxis: {
        type: 'category',
        data: dates,
        axisLabel: { color: '#9CA3AF', fontSize: 10, rotate: 45 },
        axisLine: { lineStyle: { color: '#374151' } },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 1,
        name: t('performance.ranking_stability'),
        nameTextStyle: { color: '#9CA3AF', fontSize: 10 },
        axisLabel: { color: '#9CA3AF', fontSize: 10 },
        splitLine: { lineStyle: { color: '#1F2937' } },
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#111827',
        borderColor: '#374151',
        textStyle: { color: '#F9FAFB', fontSize: 12 },
      },
      series: [
        {
          type: 'line',
          data: values,
          smooth: true,
          symbol: 'circle',
          symbolSize: 4,
          lineStyle: { color: '#F59E0B', width: 2 },
          itemStyle: { color: '#F59E0B' },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(245, 158, 11, 0.25)' },
                { offset: 1, color: 'rgba(245, 158, 11, 0.02)' },
              ],
            },
          },
        },
      ],
    };
  }, [metrics, t]);

  const bullStrengthOption = useMemo(() => {
    if (!metrics?.bull_strength_over_time?.length) return undefined;
    const dates = metrics.bull_strength_over_time.map((d: any) => d.date || d.x || '');
    const values = metrics.bull_strength_over_time.map((d: any) => d.value ?? d.y ?? d.bull_strength ?? 0);
    return {
      backgroundColor: 'transparent',
      grid: { left: 55, right: 20, top: 20, bottom: 40 },
      xAxis: {
        type: 'category',
        data: dates,
        axisLabel: { color: '#9CA3AF', fontSize: 10, rotate: 45 },
        axisLine: { lineStyle: { color: '#374151' } },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 1,
        name: t('performance.bull_strength_over_time'),
        nameTextStyle: { color: '#9CA3AF', fontSize: 10 },
        axisLabel: { color: '#9CA3AF', fontSize: 10 },
        splitLine: { lineStyle: { color: '#1F2937' } },
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#111827',
        borderColor: '#374151',
        textStyle: { color: '#F9FAFB', fontSize: 12 },
      },
      series: [
        {
          type: 'line',
          data: values,
          smooth: true,
          symbol: 'circle',
          symbolSize: 4,
          lineStyle: { color: '#3B82F6', width: 2 },
          itemStyle: { color: '#3B82F6' },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(59, 130, 246, 0.3)' },
                { offset: 1, color: 'rgba(59, 130, 246, 0.02)' },
              ],
            },
          },
        },
      ],
    };
  }, [metrics, t]);

  const rollingSharpeOption = useMemo(() => {
    if (!metrics?.rolling_sharpe?.length) return undefined;
    const dates = metrics.rolling_sharpe.map((d: any) => d.date || d.x || '');
    const values = metrics.rolling_sharpe.map((d: any) => d.value ?? d.y ?? d.sharpe ?? 0);
    return {
      backgroundColor: 'transparent',
      grid: { left: 55, right: 20, top: 20, bottom: 40 },
      xAxis: {
        type: 'category',
        data: dates,
        axisLabel: { color: '#9CA3AF', fontSize: 10, rotate: 45 },
        axisLine: { lineStyle: { color: '#374151' } },
      },
      yAxis: {
        type: 'value',
        name: t('performance.rolling_sharpe'),
        nameTextStyle: { color: '#9CA3AF', fontSize: 10 },
        axisLabel: { color: '#9CA3AF', fontSize: 10 },
        splitLine: { lineStyle: { color: '#1F2937' } },
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#111827',
        borderColor: '#374151',
        textStyle: { color: '#F9FAFB', fontSize: 12 },
      },
      series: [
        {
          type: 'line',
          data: values,
          smooth: true,
          symbol: 'circle',
          symbolSize: 4,
          lineStyle: { color: '#8B5CF6', width: 2 },
          itemStyle: { color: '#8B5CF6' },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(139, 92, 246, 0.25)' },
                { offset: 1, color: 'rgba(139, 92, 246, 0.02)' },
              ],
            },
          },
        },
      ],
    };
  }, [metrics, t]);

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-center py-20">
          <div className="flex items-center gap-3 text-text-secondary">
            <TrendingUp className="w-5 h-5 animate-pulse" />
            <span>{t('common.loading')}</span>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-center py-20">
          <p className="text-negative text-sm">{(error as Error).message}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-text-primary">{t('performance.title')}</h1>
          <p className="text-sm text-text-secondary mt-1">{t('performance.subtitle')}</p>
        </div>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          label={t('performance.accuracy')}
          value={metrics ? `${(metrics.accuracy * 100).toFixed(1)}%` : '—'}
          subtitle={t('performance.accuracy_trend')}
          icon={<Target className="w-4 h-4" />}
        />
        <MetricCard
          label={t('performance.hit_rate')}
          value={metrics ? `${(metrics.hit_rate * 100).toFixed(1)}%` : '—'}
          subtitle={t('performance.ranking_stability')}
          icon={<BarChart3 className="w-4 h-4" />}
        />
        <MetricCard
          label={t('performance.mean_score')}
          value={metrics ? metrics.mean_score.toFixed(4) : '—'}
          subtitle={t('performance.accuracy')}
          icon={<Activity className="w-4 h-4" />}
        />
        <MetricCard
          label={t('performance.total_snapshots')}
          value={metrics ? metrics.total_snapshots : '—'}
          subtitle={t('performance.rolling_sharpe')}
          icon={<Layers className="w-4 h-4" />}
        />
      </div>

      {/* Accuracy Trend Chart */}
      <div className="grid grid-cols-2 gap-6">
        <div className="panel">
          <div className="panel-header">
            <h3 className="text-sm font-medium text-text-primary">{t('performance.accuracy_trend')}</h3>
          </div>
          <div className="panel-body" style={{ height: 300 }}>
            {accuracyOption ? (
              <ReactECharts option={accuracyOption} style={{ height: '100%' }} />
            ) : (
              <p className="text-sm text-text-secondary text-center py-12">{t('common.no_data')}</p>
            )}
          </div>
        </div>

        {/* Rolling Sharpe Chart */}
        <div className="panel">
          <div className="panel-header">
            <h3 className="text-sm font-medium text-text-primary">{t('performance.rolling_sharpe')}</h3>
          </div>
          <div className="panel-body" style={{ height: 300 }}>
            {rollingSharpeOption ? (
              <ReactECharts option={rollingSharpeOption} style={{ height: '100%' }} />
            ) : (
              <p className="text-sm text-text-secondary text-center py-12">{t('common.no_data')}</p>
            )}
          </div>
        </div>
      </div>

      {/* Ranking Stability + Bull Strength */}
      <div className="grid grid-cols-2 gap-6">
        <div className="panel">
          <div className="panel-header">
            <h3 className="text-sm font-medium text-text-primary">{t('performance.ranking_stability')}</h3>
          </div>
          <div className="panel-body" style={{ height: 300 }}>
            {stabilityOption ? (
              <ReactECharts option={stabilityOption} style={{ height: '100%' }} />
            ) : (
              <p className="text-sm text-text-secondary text-center py-12">{t('common.no_data')}</p>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h3 className="text-sm font-medium text-text-primary">{t('performance.bull_strength_over_time')}</h3>
          </div>
          <div className="panel-body" style={{ height: 300 }}>
            {bullStrengthOption ? (
              <ReactECharts option={bullStrengthOption} style={{ height: '100%' }} />
            ) : (
              <p className="text-sm text-text-secondary text-center py-12">{t('common.no_data')}</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
