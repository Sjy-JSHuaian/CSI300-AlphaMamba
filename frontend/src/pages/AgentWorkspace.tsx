import { Bot, Zap, Brain, Sparkles } from 'lucide-react';

export function AgentWorkspace() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-text-primary">Agent Workspace</h1>
        <p className="text-sm text-text-secondary mt-1">Full-screen AI research agent interface</p>
      </div>
      <div className="panel">
        <div className="panel-body text-center py-16">
          <div className="w-16 h-16 rounded-2xl bg-accent/10 flex items-center justify-center mx-auto mb-4">
            <Bot className="w-8 h-8 text-accent" />
          </div>
          <h3 className="text-lg font-semibold text-text-primary mb-2">AlphaMamba Research Agent</h3>
          <p className="text-text-secondary max-w-md mx-auto mb-6">
            Use the AI Agent panel on the right for quick analysis, or open this workspace for deep research sessions.
          </p>
          <div className="flex items-center justify-center gap-4">
            <div className="flex items-center gap-2 text-xs text-text-secondary">
              <Zap className="w-3 h-3 text-positive" /> Prediction Engine
            </div>
            <div className="flex items-center gap-2 text-xs text-text-secondary">
              <Brain className="w-3 h-3 text-accent" /> Factor Analysis
            </div>
            <div className="flex items-center gap-2 text-xs text-text-secondary">
              <Sparkles className="w-3 h-3 text-warning" /> Report Generation
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
