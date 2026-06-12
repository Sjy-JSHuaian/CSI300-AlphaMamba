const BASE = '/api';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  getDates: () => request<import('../types').DatesResponse>('/market/dates'),
  getStatus: (date: string) =>
    request<import('../types').MarketStatus>(`/market/status?date=${date}`),
  predict: (data: import('../types').PredictRequest) =>
    request<import('../types').PredictResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  compare: (date: string, modes: string[]) =>
    request<import('../types').CompareResponse>('/predict/compare', {
      method: 'POST',
      body: JSON.stringify({ date, modes }),
    }),
  runBacktest: (data: { start_date: string; end_date: string; mode: string }) =>
    request<import('../types').BacktestResponse>('/backtest', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  getHistory: (limit?: number) =>
    request<import('../types').HistoryResponse>(`/history?limit=${limit || 30}`),
  getSnapshots: (limit?: number, startDate?: string, endDate?: string) => {
    const params = new URLSearchParams();
    if (limit) params.set('limit', String(limit));
    if (startDate) params.set('start_date', startDate);
    if (endDate) params.set('end_date', endDate);
    return request<{ snapshots: any[]; count: number; total: number }>(`/snapshots?${params}`);
  },
  getSnapshot: (date: string) => request<any>(`/snapshots/${date}`),
  compareSnapshots: (date1: string, date2: string) =>
    request<any>(`/snapshots/compare?date1=${date1}&date2=${date2}`),
  getPerformance: () => request<any>('/performance'),
  triggerUpdate: () => request<any>('/update', { method: 'POST' }),
  backfillSnapshots: (startDate?: string, endDate?: string) => {
    const body: any = {};
    if (startDate) body.start_date = startDate;
    if (endDate) body.end_date = endDate;
    return request<any>('/backfill', { method: 'POST', body: JSON.stringify(body) });
  },
};
