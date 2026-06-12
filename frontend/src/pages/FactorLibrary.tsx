import { Layers, Zap } from 'lucide-react';

export function FactorLibrary() {
  const factors = [
    { name: 'ret1_slope60', category: 'Shape', ic: 0.042, rankIc: 0.038, status: 'active' },
    { name: 'vol_shock5', category: 'Shock', ic: -0.031, rankIc: -0.028, status: 'active' },
    { name: 'rank_alpha20', category: 'Rank', ic: 0.028, rankIc: 0.025, status: 'active' },
    { name: 'beta_20', category: 'Risk', ic: 0.025, rankIc: 0.022, status: 'active' },
    { name: 'rsi_ac1', category: 'Momentum', ic: 0.019, rankIc: 0.017, status: 'active' },
    { name: 'ind_strength20', category: 'Industry', ic: 0.015, rankIc: 0.013, status: 'active' },
    { name: 'corr_leader20', category: 'Interaction', ic: 0.012, rankIc: 0.011, status: 'active' },
    { name: 'macd_ac1', category: 'Momentum', ic: 0.010, rankIc: 0.009, status: 'active' },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-text-primary">Factor Library</h1>
        <p className="text-sm text-text-secondary mt-1">Alpha factor inventory with IC and performance metrics</p>
      </div>
      <div className="panel">
        <div className="panel-header">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-accent" />
            <h3 className="text-sm font-medium text-text-primary">Active Factors</h3>
          </div>
          <span className="text-xs text-text-secondary">{factors.length} factors</span>
        </div>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Factor Name</th>
                <th>Category</th>
                <th>IC</th>
                <th>Rank IC</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {factors.map(f => (
                <tr key={f.name}>
                  <td className="font-mono text-sm text-accent">{f.name}</td>
                  <td className="text-text-secondary">{f.category}</td>
                  <td className={f.ic > 0 ? 'text-positive' : 'text-negative'}>{f.ic > 0 ? '+' : ''}{f.ic.toFixed(4)}</td>
                  <td className={f.rankIc > 0 ? 'text-positive' : 'text-negative'}>{f.rankIc > 0 ? '+' : ''}{f.rankIc.toFixed(4)}</td>
                  <td><span className="badge-positive">{f.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
