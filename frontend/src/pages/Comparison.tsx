import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/client';
import { GitCompare, TrendingUp, Minus, ArrowUp, ArrowDown, Activity } from 'lucide-react';
import ReactECharts from 'echarts-for-react';

interface RankingChange {
  stock_id: string;
  rank_a?: number;
  rank_b?: number;
  rank_delta?: number;
  score_a?: number;
  score_b?: number;
  score_delta?: number;
}

interface ComparisonData {
  date_a: string;
  date_b: string;
  bull_strength_a: number;
  bull_strength_b: number;
  regime_a: string;
  regime_b: string;
  portfolio_a: string[];
  portfolio_b: string[];
  stayed: string[];
  entered: string[];
  left: string[];
  overlap_pct: number;
  ranking_changes: RankingChange[];
  snapshot_a?: any;
  snapshot_b?: any;
}

export function Comparison() {
  const { t } = useTranslation();
  const [dateA, setDateA] = useState('');
  const [dateB, setDateB] = useState('');
  const [compareEnabled, setCompareEnabled] = useState(false);

  const { data: dates } = useQuery({
    queryKey: ['dates'],
    queryFn: api.getDates,
  });

  const { data: comparison, isLoading, error } = useQuery({
    queryKey: ['compareSnapshots', dateA, dateB],
    queryFn: () => api.compareSnapshots(dateA, dateB),
    enabled: compareEnabled && !!dateA && !!dateB,
  });

  const handleCompare = () => {
    if (dateA && dateB) {
      setCompareEnabled(true);
    }
  };

  const bullStrengthDelta =
    comparison ? (comparison.bull_strength_b || 0) - (comparison.bull_strength_a || 0) : 0;

  const getRegimeColor = (label: string) => {
    const lower = (label || '').toLowerCase();
    if (lower.includes('bull')) return 'text-positive';
    if (lower.includes('bear')) return 'text-negative';
    return 'text-warning';
  };

  const getDeltaColor = (val: number) => {
    if (val > 0) return 'text-positive';
    if (val < 0) return 'text-negative';
    return 'text-text-secondary';
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.6) return 'text-positive';
    if (score >= 0.4) return 'text-warning';
    return 'text-negative';
  };

  // Score comparison scatter chart
  const scatterOption = comparison?.ranking_changes?.length ? {
    backgroundColor: 'transparent',
    grid: { left: 60, right: 30, top: 30, bottom: 50 },
    xAxis: {
      name: `${t('comparison.date_a')} Score`,
      nameTextStyle: { color: '#9CA3AF', fontSize: 11 },
      type: 'value',
      axisLabel: { color: '#9CA3AF', fontSize: 10 },
      splitLine: { lineStyle: { color: '#1F2937' } },
    },
    yAxis: {
      name: `${t('comparison.date_b')} Score`,
      nameTextStyle: { color: '#9CA3AF', fontSize: 11 },
      type: 'value',
      axisLabel: { color: '#9CA3AF', fontSize: 10 },
      splitLine: { lineStyle: { color: '#1F2937' } },
    },
    tooltip: {
      trigger: 'item',
      backgroundColor: '#111827',
      borderColor: '#374151',
      textStyle: { color: '#F9FAFB', fontSize: 12 },
      formatter: (params: any) => {
        const d = params.data;
        return `<strong>${d[2]}</strong><br/>Score A: ${d[0].toFixed(4)}<br/>Score B: ${d[1].toFixed(4)}`;
      },
    },
    series: [
      {
        type: 'scatter',
        data: comparison.ranking_changes.map((r: RankingChange) => [r.score_a || 0, r.score_b || 0, r.stock_id]),
        symbolSize: 8,
        itemStyle: {
          color: '#3B82F6',
          borderColor: '#60A5FA',
          borderWidth: 1,
        },
      },
    ],
  } : undefined;

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-center py-20">
          <div className="flex items-center gap-3 text-text-secondary">
            <GitCompare className="w-5 h-5 animate-pulse" />
            <span>{t('common.loading')}</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-text-primary">{t('comparison.title')}</h1>
          <p className="text-sm text-text-secondary mt-1">{t('comparison.subtitle')}</p>
        </div>
      </div>

      {/* Date Selectors */}
      <div className="panel">
        <div className="panel-header">
          <h3 className="text-sm font-medium text-text-primary">{t('comparison.compare')}</h3>
        </div>
        <div className="panel-body">
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <label className="text-xs text-text-secondary block mb-1.5">{t('comparison.date_a')}</label>
              <select
                className="input-dark w-full"
                value={dateA}
                onChange={(e) => setDateA(e.target.value)}
              >
                <option value="">{t('comparison.select_date')}</option>
                {dates?.dates.map((d) => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
            </div>
            <div className="pt-5">
              <GitCompare className="w-5 h-5 text-text-secondary" />
            </div>
            <div className="flex-1">
              <label className="text-xs text-text-secondary block mb-1.5">{t('comparison.date_b')}</label>
              <select
                className="input-dark w-full"
                value={dateB}
                onChange={(e) => setDateB(e.target.value)}
              >
                <option value="">{t('comparison.select_date')}</option>
                {dates?.dates.map((d) => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
            </div>
            <div className="pt-5">
              <button
                onClick={handleCompare}
                disabled={!dateA || !dateB}
                className="btn-primary px-6 py-2 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {t('comparison.compare')}
              </button>
            </div>
          </div>
        </div>
      </div>

      {comparison && (
        <>
          {/* Regime Change + Stats */}
          <div className="grid grid-cols-4 gap-4">
            <div className="metric-card">
              <div className="flex items-center justify-between mb-2">
                <span className="metric-label">{t('comparison.bull_strength')}</span>
                <span className="text-text-secondary"><Activity className="w-4 h-4" /></span>
              </div>
              <div className={`metric-value ${getDeltaColor(bullStrengthDelta)}`}>
                {bullStrengthDelta > 0 ? '+' : ''}{bullStrengthDelta.toFixed(4)}
              </div>
              <p className="text-xs text-text-secondary mt-1">
                A: {comparison.bull_strength_a?.toFixed(3)} → B: {comparison.bull_strength_b?.toFixed(3)}
              </p>
            </div>
            <div className="metric-card">
              <div className="flex items-center justify-between mb-2">
                <span className="metric-label">{t('comparison.regime_change')}</span>
                <span className="text-text-secondary"><TrendingUp className="w-4 h-4" /></span>
              </div>
              <div className={getRegimeColor(comparison.regime_a || comparison.regime_b || '')}>
                <span className="text-lg font-semibold">
                  {comparison.regime_a || comparison.snapshot_a?.regime_label || '—'}
                  {' → '}
                  {comparison.regime_b || comparison.snapshot_b?.regime_label || '—'}
                </span>
              </div>
              <p className="text-xs text-text-secondary mt-1">
                {comparison.regime_a !== comparison.regime_b ? 'Regime changed' : 'Regime unchanged'}
              </p>
            </div>
            <div className="metric-card">
              <div className="flex items-center justify-between mb-2">
                <span className="metric-label">{t('comparison.overlap_pct')}</span>
                <span className="text-text-secondary"><Minus className="w-4 h-4" /></span>
              </div>
              <div className="metric-value">
                {comparison.overlap_pct != null ? `${(comparison.overlap_pct * 100).toFixed(0)}%` : '—'}
              </div>
              <p className="text-xs text-text-secondary mt-1">
                {comparison.stayed?.length || 0} stocks in common
              </p>
            </div>
            <div className="metric-card">
              <div className="flex items-center justify-between mb-2">
                <span className="metric-label">{t('comparison.portfolio_changes')}</span>
                <span className="text-text-secondary"><GitCompare className="w-4 h-4" /></span>
              </div>
              <div className="metric-value">
                {comparison.stayed?.length || 0} / {(comparison.portfolio_a?.length || 0) + (comparison.portfolio_b?.length || 0)}
              </div>
              <p className="text-xs text-text-secondary mt-1">
                +{comparison.entered?.length || 0} / -{comparison.left?.length || 0}
              </p>
            </div>
          </div>

          {/* Portfolio Changes Row */}
          <div className="grid grid-cols-3 gap-6">
            {/* Stayed */}
            <div className="panel">
              <div className="panel-header">
                <div className="flex items-center gap-2">
                  <Minus className="w-4 h-4 text-warning" />
                  <h3 className="text-sm font-medium text-text-primary">{t('comparison.stayed')}</h3>
                  <span className="text-xs text-text-secondary">({comparison.stayed?.length || 0})</span>
                </div>
              </div>
              <div className="panel-body">
                {comparison.stayed?.length > 0 ? (
                  <div className="space-y-1.5">
                    {comparison.stayed.map((stock: string) => {
                      const change = comparison.ranking_changes?.find((r: RankingChange) => r.stock_id === stock);
                      return (
                        <div key={stock} className="flex items-center justify-between py-1.5 px-2 rounded bg-background">
                          <span className="text-sm font-medium text-text-primary">{stock}</span>
                          <div className="flex items-center gap-3">
                            <span className={`text-xs ${getDeltaColor(change?.score_delta || 0)}`}>
                              {change?.score_delta != null
                                ? `${change.score_delta > 0 ? '+' : ''}${change.score_delta.toFixed(3)}`
                                : '—'}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <p className="text-sm text-text-secondary text-center py-6">{t('common.no_data')}</p>
                )}
              </div>
            </div>

            {/* Entered */}
            <div className="panel">
              <div className="panel-header">
                <div className="flex items-center gap-2">
                  <ArrowUp className="w-4 h-4 text-positive" />
                  <h3 className="text-sm font-medium text-text-primary">{t('comparison.entered')}</h3>
                  <span className="text-xs text-text-secondary">({comparison.entered?.length || 0})</span>
                </div>
              </div>
              <div className="panel-body">
                {comparison.entered?.length > 0 ? (
                  <div className="space-y-1.5">
                    {comparison.entered.map((stock: string) => {
                      const change = comparison.ranking_changes?.find((r: RankingChange) => r.stock_id === stock);
                      return (
                        <div key={stock} className="flex items-center justify-between py-1.5 px-2 rounded bg-background">
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] text-positive">NEW</span>
                            <span className="text-sm font-medium text-text-primary">{stock}</span>
                          </div>
                          <span className={`text-xs ${getScoreColor(change?.score_b || 0)}`}>
                            {change?.score_b != null ? change.score_b.toFixed(3) : '—'}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <p className="text-sm text-text-secondary text-center py-6">{t('common.no_data')}</p>
                )}
              </div>
            </div>

            {/* Left */}
            <div className="panel">
              <div className="panel-header">
                <div className="flex items-center gap-2">
                  <ArrowDown className="w-4 h-4 text-negative" />
                  <h3 className="text-sm font-medium text-text-primary">{t('comparison.left')}</h3>
                  <span className="text-xs text-text-secondary">({comparison.left?.length || 0})</span>
                </div>
              </div>
              <div className="panel-body">
                {comparison.left?.length > 0 ? (
                  <div className="space-y-1.5">
                    {comparison.left.map((stock: string) => {
                      const change = comparison.ranking_changes?.find((r: RankingChange) => r.stock_id === stock);
                      return (
                        <div key={stock} className="flex items-center justify-between py-1.5 px-2 rounded bg-background">
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] text-negative">OUT</span>
                            <span className="text-sm font-medium text-text-primary">{stock}</span>
                          </div>
                          <span className={`text-xs ${getScoreColor(change?.score_a || 0)}`}>
                            {change?.score_a != null ? change.score_a.toFixed(3) : '—'}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <p className="text-sm text-text-secondary text-center py-6">{t('common.no_data')}</p>
                )}
              </div>
            </div>
          </div>

          {/* Ranking Changes + Score Scatter */}
          <div className="grid grid-cols-2 gap-6">
            {/* Ranking Changes Table */}
            <div className="panel">
              <div className="panel-header">
                <h3 className="text-sm font-medium text-text-primary">{t('comparison.ranking_changes')}</h3>
              </div>
              <div className="panel-body">
                {comparison.ranking_changes?.length > 0 ? (
                  <table className="data-table w-full">
                    <thead>
                      <tr>
                        <th>Stock ID</th>
                        <th>{t('comparison.rank_delta')}</th>
                        <th>{t('comparison.score_delta')}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {comparison.ranking_changes.map((r: RankingChange, i: number) => (
                        <tr key={r.stock_id || i}>
                          <td className="font-medium text-text-primary">{r.stock_id}</td>
                          <td className={getDeltaColor(r.rank_delta || 0)}>
                            {r.rank_delta != null
                              ? `${r.rank_delta > 0 ? '+' : ''}${r.rank_delta}`
                              : '—'}
                          </td>
                          <td className={getDeltaColor(r.score_delta || 0)}>
                            {r.score_delta != null
                              ? `${r.score_delta > 0 ? '+' : ''}${r.score_delta.toFixed(4)}`
                              : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <p className="text-sm text-text-secondary text-center py-6">{t('common.no_data')}</p>
                )}
              </div>
            </div>

            {/* Score Scatter Chart */}
            <div className="panel">
              <div className="panel-header">
                <h3 className="text-sm font-medium text-text-primary">{t('comparison.score_delta')}</h3>
              </div>
              <div className="panel-body" style={{ height: 320 }}>
                {scatterOption ? (
                  <ReactECharts option={scatterOption} style={{ height: '100%' }} />
                ) : (
                  <p className="text-sm text-text-secondary text-center py-12">{t('common.no_data')}</p>
                )}
              </div>
            </div>
          </div>

          {/* Side-by-side Top Picks */}
          <div className="grid grid-cols-2 gap-6">
            {/* Date A Top Picks */}
            <div className="panel">
              <div className="panel-header">
                <h3 className="text-sm font-medium text-text-primary">
                  {t('comparison.date_a')}: {comparison.date_a || comparison.snapshot_a?.date || dateA}
                </h3>
                <span className={`text-xs ${getRegimeColor(comparison.regime_a || '')}`}>
                  BS: {comparison.bull_strength_a?.toFixed(3) || '—'}
                </span>
              </div>
              <div className="panel-body">
                {comparison.portfolio_a?.length > 0 ? (
                  <div className="space-y-1.5">
                    {comparison.portfolio_a.map((stock: string, idx: number) => (
                      <div key={stock} className="flex items-center justify-between py-1.5 px-2 rounded bg-background">
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-text-secondary">#{idx + 1}</span>
                          <span className="text-sm font-medium text-text-primary">{stock}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-text-secondary text-center py-6">
                    {comparison.snapshot_a?.stocks?.length > 0
                      ? comparison.snapshot_a.stocks.map((s: string, i: number) => (
                          <span key={s} className="inline-block mr-2">{s}</span>
                        ))
                      : t('common.no_data')}
                  </p>
                )}
              </div>
            </div>

            {/* Date B Top Picks */}
            <div className="panel">
              <div className="panel-header">
                <h3 className="text-sm font-medium text-text-primary">
                  {t('comparison.date_b')}: {comparison.date_b || comparison.snapshot_b?.date || dateB}
                </h3>
                <span className={`text-xs ${getRegimeColor(comparison.regime_b || '')}`}>
                  BS: {comparison.bull_strength_b?.toFixed(3) || '—'}
                </span>
              </div>
              <div className="panel-body">
                {comparison.portfolio_b?.length > 0 ? (
                  <div className="space-y-1.5">
                    {comparison.portfolio_b.map((stock: string, idx: number) => (
                      <div key={stock} className="flex items-center justify-between py-1.5 px-2 rounded bg-background">
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-text-secondary">#{idx + 1}</span>
                          <span className="text-sm font-medium text-text-primary">{stock}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-text-secondary text-center py-6">
                    {comparison.snapshot_b?.stocks?.length > 0
                      ? comparison.snapshot_b.stocks.map((s: string, i: number) => (
                          <span key={s} className="inline-block mr-2">{s}</span>
                        ))
                      : t('common.no_data')}
                  </p>
                )}
              </div>
            </div>
          </div>
        </>
      )}

      {error && (
        <div className="panel">
          <div className="panel-body">
            <p className="text-negative text-sm text-center py-8">{(error as Error).message}</p>
          </div>
        </div>
      )}

      {!comparison && !isLoading && !error && compareEnabled && (
        <div className="panel">
          <div className="panel-body">
            <p className="text-sm text-text-secondary text-center py-12">{t('comparison.select_date')}</p>
          </div>
        </div>
      )}
    </div>
  );
}
