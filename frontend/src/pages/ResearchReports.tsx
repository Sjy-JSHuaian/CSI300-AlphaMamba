import { FileText, Download, Eye, Calendar } from 'lucide-react';

export function ResearchReports() {
  const reports = [
    { id: 1, title: 'Daily Alpha Report', date: '2026-06-12', type: 'Daily', status: 'ready' },
    { id: 2, title: 'Weekly Factor Review', date: '2026-06-11', type: 'Weekly', status: 'ready' },
    { id: 3, title: 'Market Regime Analysis', date: '2026-06-10', type: 'Analysis', status: 'ready' },
    { id: 4, title: 'Portfolio Risk Assessment', date: '2026-06-09', type: 'Risk', status: 'generating' },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-text-primary">Research Reports</h1>
          <p className="text-sm text-text-secondary mt-1">AI-generated institutional research</p>
        </div>
        <button className="btn-primary flex items-center gap-2">
          <FileText className="w-4 h-4" /> Generate New Report
        </button>
      </div>

      <div className="panel">
        <div className="panel-header">
          <h3 className="text-sm font-medium text-text-primary">Recent Reports</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr><th>Title</th><th>Date</th><th>Type</th><th>Status</th><th>Actions</th></tr>
            </thead>
            <tbody>
              {reports.map(r => (
                <tr key={r.id}>
                  <td className="font-medium">{r.title}</td>
                  <td className="text-text-secondary"><Calendar className="w-3 h-3 inline mr-1" />{r.date}</td>
                  <td><span className="badge-accent">{r.type}</span></td>
                  <td>{r.status === 'ready' ? <span className="badge-positive">Ready</span> : <span className="badge-warning">Generating...</span>}</td>
                  <td>
                    <div className="flex gap-2">
                      <button className="text-text-secondary hover:text-text-primary"><Eye className="w-4 h-4" /></button>
                      <button className="text-text-secondary hover:text-text-primary"><Download className="w-4 h-4" /></button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
