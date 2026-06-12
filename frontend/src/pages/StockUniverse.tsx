import { Database, Search } from 'lucide-react';

export function StockUniverse() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-text-primary">Stock Universe</h1>
        <p className="text-sm text-text-secondary mt-1">CSI 300 constituent analysis and screening</p>
      </div>
      <div className="panel">
        <div className="panel-header">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-accent" />
            <h3 className="text-sm font-medium text-text-primary">Constituents</h3>
          </div>
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-text-secondary" />
            <input className="input-dark pl-9 w-64" placeholder="Search by ticker or name..." />
          </div>
        </div>
        <div className="panel-body text-center py-16">
          <Database className="w-12 h-12 text-text-secondary/30 mx-auto mb-4" />
          <p className="text-text-secondary">Stock universe data loading...</p>
          <p className="text-xs text-text-secondary mt-1">CSI 300 constituents with factor exposures and rankings</p>
        </div>
      </div>
    </div>
  );
}
