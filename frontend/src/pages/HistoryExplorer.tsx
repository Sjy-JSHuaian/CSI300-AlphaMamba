import { useState, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/client';
import ReactECharts from 'echarts-for-react';
import { Clock, Calendar, Table, TrendingUp, Search, Filter, ChevronLeft, ChevronRight } from 'lucide-react';

type ViewMode = 'timeline' | 'calendar' | 'table';
type SortField = 'date' | 'bull_strength' | 'regime_label';
type SortDir = 'asc' | 'desc';

interface SnapshotEntry {
  date: string;
  bull_strength: number;
  regime_label: string;
  stocks: string[];
  weights: number[];
  top_stock?: string;
  portfolio_size?: number;
}

export function HistoryExplorer() {
  const { t } = useTranslation();
  const [viewMode, setViewMode] = useState<ViewMode>('timeline');
  const [searchQuery, setSearchQuery] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [sortField, setSortField] = useState<SortField>('date');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [calendarYear, setCalendarYear] = useState(new Date().getFullYear());
  const [calendarMonth, setCalendarMonth] = useState(new Date().getMonth() + 1);

  const { data, isLoading, error } = useQuery({
    queryKey: ['snapshots', { limit: 500, startDate: startDate || undefined, endDate: endDate || undefined }],
    queryFn: () => api.getSnapshots(500, startDate || undefined, endDate || undefined),
  });

  const snapshots: SnapshotEntry[] = useMemo(() => {
    if (!data?.snapshots) return [];
    return data.snapshots.map((s: any) => ({
      ...s,
      top_stock: s.stocks?.[0] || s.top_stock || '—',
      portfolio_size: s.stocks?.length || s.portfolio_size || 0,
    }));
  }, [data]);

  const filteredSnapshots = useMemo(() => {
    if (!searchQuery.trim()) return snapshots;
    const q = searchQuery.toLowerCase();
    return snapshots.filter((s) => {
      return (
        s.date.includes(q) ||
        s.regime_label?.toLowerCase().includes(q) ||
        s.stocks?.some((stock: string) => stock.toLowerCase().includes(q))
      );
    });
  }, [snapshots, searchQuery]);

  const sortedSnapshots = useMemo(() => {
    const sorted = [...filteredSnapshots];
    sorted.sort((a, b) => {
      let cmp = 0;
      if (sortField === 'date') {
        cmp = a.date.localeCompare(b.date);
      } else if (sortField === 'bull_strength') {
        cmp = (a.bull_strength || 0) - (b.bull_strength || 0);
      } else if (sortField === 'regime_label') {
        cmp = (a.regime_label || '').localeCompare(b.regime_label || '');
      }
      return sortDir === 'asc' ? cmp : -cmp;
    });
    return sorted;
  }, [filteredSnapshots, sortField, sortDir]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDir('desc');
    }
  };

  const sortIndicator = (field: SortField) => {
    if (sortField !== field) return '';
    return sortDir === 'asc' ? ' ▲' : ' ▼';
  };

  // Bull strength trend chart
  const trendOption = useMemo(() => {
    const dates = filteredSnapshots.map((s) => s.date);
    const values = filteredSnapshots.map((s) => s.bull_strength);
    return {
      backgroundColor: 'transparent',
      grid: { left: 50, right: 20, top: 20, bottom: 40 },
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
          symbol: 'none',
          lineStyle: { color: '#3B82F6', width: 2 },
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
  }, [filteredSnapshots]);

  // Calendar grid
  const calendarDays = useMemo(() => {
    const snapshotDates = new Set(snapshots.map((s) => s.date));
    const daysInMonth = new Date(calendarYear, calendarMonth, 0).getDate();
    const firstDayOfWeek = new Date(calendarYear, calendarMonth - 1, 1).getDay();
    const days: (number | null)[] = [];
    for (let i = 0; i < firstDayOfWeek; i++) {
      days.push(null);
    }
    for (let d = 1; d <= daysInMonth; d++) {
      days.push(d);
    }
    return { days, snapshotDates };
  }, [snapshots, calendarYear, calendarMonth]);

  const prevMonth = () => {
    if (calendarMonth === 1) {
      setCalendarMonth(12);
      setCalendarYear(calendarYear - 1);
    } else {
      setCalendarMonth(calendarMonth - 1);
    }
  };

  const nextMonth = () => {
    if (calendarMonth === 12) {
      setCalendarMonth(1);
      setCalendarYear(calendarYear + 1);
    } else {
      setCalendarMonth(calendarMonth + 1);
    }
  };

  const monthNames = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
  ];

  const dayHeaders = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  const getBullStrengthColor = (val: number) => {
    if (val >= 0.7) return 'text-positive';
    if (val >= 0.5) return 'text-warning';
    return 'text-negative';
  };

  const getBullStrengthBg = (val: number) => {
    if (val >= 0.7) return 'bg-positive';
    if (val >= 0.5) return 'bg-warning';
    return 'bg-negative';
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-center py-20">
          <div className="flex items-center gap-3 text-text-secondary">
            <TrendingUp className="w-5 h-5 animate-pulse" />
            <span>{t('history.loading')}</span>
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
          <h1 className="text-2xl font-semibold text-text-primary">{t('history.title')}</h1>
          <p className="text-sm text-text-secondary mt-1">{t('history.subtitle')}</p>
        </div>
        <div className="flex items-center gap-2 bg-surface rounded-lg p-1 border border-border">
          <button
            onClick={() => setViewMode('timeline')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              viewMode === 'timeline' ? 'bg-accent text-white' : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            <Clock className="w-3.5 h-3.5" />
            {t('history.timeline_view')}
          </button>
          <button
            onClick={() => setViewMode('calendar')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              viewMode === 'calendar' ? 'bg-accent text-white' : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            <Calendar className="w-3.5 h-3.5" />
            {t('history.calendar_view')}
          </button>
          <button
            onClick={() => setViewMode('table')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              viewMode === 'table' ? 'bg-accent text-white' : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            <Table className="w-3.5 h-3.5" />
            {t('history.table_view')}
          </button>
        </div>
      </div>

      {/* Stats Bar */}
      <div className="grid grid-cols-4 gap-4">
        <div className="panel">
          <div className="panel-body py-4 flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center">
              <Clock className="w-5 h-5 text-accent" />
            </div>
            <div>
              <p className="text-xs text-text-secondary">{t('history.snapshot_count')}</p>
              <p className="text-xl font-semibold text-text-primary">{snapshots.length}</p>
            </div>
          </div>
        </div>
        <div className="panel">
          <div className="panel-body py-4 flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-positive/10 flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-positive" />
            </div>
            <div>
              <p className="text-xs text-text-secondary">{t('performance.mean_score')}</p>
              <p className="text-xl font-semibold text-text-primary">
                {snapshots.length > 0
                  ? (snapshots.reduce((sum, s) => sum + (s.bull_strength || 0), 0) / snapshots.length).toFixed(3)
                  : '—'}
              </p>
            </div>
          </div>
        </div>
        <div className="panel">
          <div className="panel-body py-4 flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-warning/10 flex items-center justify-center">
              <Filter className="w-5 h-5 text-warning" />
            </div>
            <div>
              <p className="text-xs text-text-secondary">{t('history.filter_date_range')}</p>
              <p className="text-xl font-semibold text-text-primary">
                {snapshots.length > 0
                  ? `${snapshots[snapshots.length - 1].date} ~ ${snapshots[0].date}`
                  : '—'}
              </p>
            </div>
          </div>
        </div>
        <div className="panel">
          <div className="panel-body py-4 flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-info/10 flex items-center justify-center">
              <Search className="w-5 h-5 text-info" />
            </div>
            <div>
              <p className="text-xs text-text-secondary">{t('history.search_records')}</p>
              <p className="text-xl font-semibold text-text-primary">
                {searchQuery ? filteredSnapshots.length : snapshots.length}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3">
        <div className="relative flex-1 max-w-xs">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-secondary" />
          <input
            type="text"
            placeholder={t('history.search_records')}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="input-dark pl-9 w-full"
          />
        </div>
        <input
          type="date"
          value={startDate}
          onChange={(e) => setStartDate(e.target.value)}
          className="input-dark text-sm"
        />
        <span className="text-text-secondary text-sm">—</span>
        <input
          type="date"
          value={endDate}
          onChange={(e) => setEndDate(e.target.value)}
          className="input-dark text-sm"
        />
      </div>

      {/* Bull Strength Trend Chart */}
      <div className="panel">
        <div className="panel-header">
          <h3 className="text-sm font-medium text-text-primary">{t('history.bull_strength_trend')}</h3>
        </div>
        <div className="panel-body" style={{ height: 280 }}>
          {filteredSnapshots.length > 0 ? (
            <ReactECharts option={trendOption} style={{ height: '100%' }} />
          ) : (
            <p className="text-sm text-text-secondary text-center py-12">{t('history.no_snapshots')}</p>
          )}
        </div>
      </div>

      {/* View Mode Content */}
      {viewMode === 'timeline' && (
        <div className="panel">
          <div className="panel-header">
            <h3 className="text-sm font-medium text-text-primary">{t('history.timeline_view')}</h3>
          </div>
          <div className="panel-body">
            {filteredSnapshots.length > 0 ? (
              <div className="relative pl-6">
                <div className="absolute left-2 top-0 bottom-0 w-px bg-border" />
                <div className="space-y-0">
                  {filteredSnapshots.map((snap, idx) => (
                    <div key={snap.date} className="relative pb-6 last:pb-0">
                      <div
                        className={`absolute left-[-22px] top-1 w-3 h-3 rounded-full border-2 border-surface ${getBullStrengthBg(snap.bull_strength)}`}
                      />
                      <div className="flex items-start gap-4">
                        <div className="min-w-[100px]">
                          <p className="text-sm font-medium text-text-primary">{snap.date}</p>
                          <p className="text-xs text-text-secondary mt-0.5">{snap.regime_label}</p>
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-3">
                            <span className={`text-xs font-medium ${getBullStrengthColor(snap.bull_strength)}`}>
                              BS: {snap.bull_strength?.toFixed(3)}
                            </span>
                            <span className="text-xs text-text-secondary">
                              {snap.portfolio_size} {t('history.snapshot_count')}
                            </span>
                          </div>
                          <div className="flex flex-wrap gap-1 mt-1.5">
                            {snap.stocks?.slice(0, 5).map((stock: string) => (
                              <span
                                key={stock}
                                className="px-1.5 py-0.5 rounded text-[10px] bg-background text-text-secondary border border-border"
                              >
                                {stock}
                              </span>
                            ))}
                            {snap.stocks?.length > 5 && (
                              <span className="px-1.5 py-0.5 rounded text-[10px] bg-background text-text-secondary border border-border">
                                +{snap.stocks.length - 5}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <p className="text-sm text-text-secondary text-center py-12">{t('history.no_snapshots')}</p>
            )}
          </div>
        </div>
      )}

      {viewMode === 'calendar' && (
        <div className="panel">
          <div className="panel-header flex items-center justify-between">
            <h3 className="text-sm font-medium text-text-primary">{t('history.calendar_view')}</h3>
            <div className="flex items-center gap-2">
              <button
                onClick={prevMonth}
                className="p-1 rounded hover:bg-surface transition-colors text-text-secondary hover:text-text-primary"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="text-sm font-medium text-text-primary min-w-[120px] text-center">
                {monthNames[calendarMonth - 1]} {calendarYear}
              </span>
              <button
                onClick={nextMonth}
                className="p-1 rounded hover:bg-surface transition-colors text-text-secondary hover:text-text-primary"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
          <div className="panel-body">
            <div className="grid grid-cols-7 gap-0.5">
              {dayHeaders.map((day) => (
                <div key={day} className="text-center py-2">
                  <span className="text-[10px] font-medium text-text-secondary uppercase">{day}</span>
                </div>
              ))}
              {calendarDays.days.map((day, idx) => {
                if (day === null) {
                  return <div key={`empty-${idx}`} className="aspect-square" />;
                }
                const dateStr = `${calendarYear}-${String(calendarMonth).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
                const hasSnapshot = calendarDays.snapshotDates.has(dateStr);
                const snapshot = snapshots.find((s) => s.date === dateStr);
                return (
                  <div
                    key={dateStr}
                    className={`aspect-square rounded-md flex flex-col items-center justify-center text-xs transition-colors ${
                      hasSnapshot
                        ? 'bg-accent/20 border border-accent/30 cursor-pointer hover:bg-accent/30'
                        : 'text-text-secondary'
                    }`}
                    title={hasSnapshot ? `${dateStr} - BS: ${snapshot?.bull_strength?.toFixed(3)}` : dateStr}
                  >
                    <span className={hasSnapshot ? 'text-text-primary font-medium' : ''}>{day}</span>
                    {hasSnapshot && (
                      <span className={`text-[9px] ${getBullStrengthColor(snapshot?.bull_strength || 0)}`}>
                        {snapshot?.bull_strength?.toFixed(2)}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {viewMode === 'table' && (
        <div className="panel">
          <div className="panel-header">
            <h3 className="text-sm font-medium text-text-primary">{t('history.table_view')}</h3>
          </div>
          <div className="panel-body">
            {sortedSnapshots.length > 0 ? (
              <table className="data-table w-full">
                <thead>
                  <tr>
                    <th className="cursor-pointer hover:text-text-primary" onClick={() => handleSort('date')}>
                      Date{sortIndicator('date')}
                    </th>
                    <th className="cursor-pointer hover:text-text-primary" onClick={() => handleSort('bull_strength')}>
                      Bull Strength{sortIndicator('bull_strength')}
                    </th>
                    <th className="cursor-pointer hover:text-text-primary" onClick={() => handleSort('regime_label')}>
                      Regime{sortIndicator('regime_label')}
                    </th>
                    <th>Top Stock</th>
                    <th>Portfolio Size</th>
                    <th>Stocks</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedSnapshots.map((snap) => (
                    <tr key={snap.date}>
                      <td className="font-medium text-text-primary">{snap.date}</td>
                      <td className={getBullStrengthColor(snap.bull_strength)}>
                        {snap.bull_strength?.toFixed(4)}
                      </td>
                      <td className="text-text-secondary">{snap.regime_label}</td>
                      <td className="text-text-primary">{snap.top_stock}</td>
                      <td className="text-text-secondary">{snap.portfolio_size}</td>
                      <td>
                        <div className="flex flex-wrap gap-1">
                          {snap.stocks?.slice(0, 3).map((stock: string) => (
                            <span
                              key={stock}
                              className="px-1.5 py-0.5 rounded text-[10px] bg-background text-text-secondary border border-border"
                            >
                              {stock}
                            </span>
                          ))}
                          {snap.stocks?.length > 3 && (
                            <span className="px-1.5 py-0.5 rounded text-[10px] bg-background text-text-secondary border border-border">
                              +{snap.stocks.length - 3}
                            </span>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="text-sm text-text-secondary text-center py-12">{t('history.no_snapshots')}</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
