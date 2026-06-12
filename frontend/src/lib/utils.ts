import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatPercent(value: number, decimals = 2): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatNumber(value: number, decimals = 2): string {
  return value.toFixed(decimals);
}

export function formatLargeNumber(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return value.toFixed(0);
}

export function getScoreColor(score: number): string {
  if (score >= 0.8) return "text-positive";
  if (score >= 0.6) return "text-accent";
  if (score >= 0.4) return "text-warning";
  return "text-text-secondary";
}

export function getRegimeLabel(bs: number): { text: string; color: string } {
  if (bs > 0.8) return { text: "Strong Bull", color: "text-positive" };
  if (bs > 0.5) return { text: "Bull", color: "text-accent" };
  if (bs > 0.2) return { text: "Neutral", color: "text-warning" };
  return { text: "Bearish", color: "text-negative" };
}
