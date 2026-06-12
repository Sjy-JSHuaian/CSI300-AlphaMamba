export interface StockResult {
  stock_id: string;
  weight: number;
  score: number;
  sector_id: number;
}

export interface PredictResponse {
  date: string;
  mode: string;
  bull_strength: number;
  regime_label: string;
  portfolio: StockResult[];
  bull_top5: string[];
  nonbull_top5: string[];
  consensus_overlap: number;
  score_distribution: Record<string, number>;
  sector_distribution: Record<string, number>;
}

export interface PredictRequest {
  date: string;
  mode: string;
  temperature?: number;
  max_per_industry?: number;
  corr_threshold?: number;
  min_score_ratio?: number;
  top_n?: number;
  select_k?: number;
  bull_gate?: number;
  turnover_penalty?: number;
  persistence_bonus?: number;
}

export interface MarketStatus {
  date: string;
  bull_strength: number;
  regime_label: string;
  n_stocks: number;
}

export interface DatesResponse {
  dates: string[];
  count: number;
  first: string;
  last: string;
}

export interface CompareResponse {
  date: string;
  bull_strength: number;
  regime_label: string;
  results: Record<string, PredictResponse>;
}

export interface BacktestResponse {
  mode: string;
  metrics: Record<string, number>;
  daily_returns: { date: string; cumulative_return: number }[];
  monthly_returns: { year: number; month: number; return: number }[];
}

export interface HistoryRecord {
  date: string;
  bull_strength: number;
  regime_label: string;
  stocks: string[];
  weights: number[];
}

export interface HistoryResponse {
  history: HistoryRecord[];
  count: number;
}
