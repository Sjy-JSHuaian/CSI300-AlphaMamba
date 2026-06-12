import { cn } from '@/lib/utils';

interface MetricCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
  icon?: React.ReactNode;
}

export function MetricCard({ label, value, subtitle, trend, icon }: MetricCardProps) {
  return (
    <div className="metric-card">
      <div className="flex items-center justify-between mb-2">
        <span className="metric-label">{label}</span>
        {icon && <span className="text-text-secondary">{icon}</span>}
      </div>
      <div className={cn(
        "metric-value",
        trend === 'up' && "text-positive",
        trend === 'down' && "text-negative",
      )}>
        {value}
      </div>
      {subtitle && (
        <p className="text-xs text-text-secondary mt-1">{subtitle}</p>
      )}
    </div>
  );
}
