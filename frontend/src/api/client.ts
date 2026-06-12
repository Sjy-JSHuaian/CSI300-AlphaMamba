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
};
