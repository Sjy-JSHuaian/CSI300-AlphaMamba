import { useState, useRef, useEffect } from 'react';
import { Bot, Send, Sparkles, ChevronDown, Zap, Brain, LineChart, Newspaper, FileText, Database, Layers } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { AIMessage } from '@/types';

const QUICK_PROMPTS = [
  "Why is 300308 ranked #1?",
  "Explain today's strongest alpha factors",
  "Compare 300308 and 300394",
  "Generate a research report",
  "What risks exist in this prediction?",
  "Summarize today's market condition",
];

const CAPABILITIES = [
  { icon: Zap, label: 'Prediction Engine', active: true },
  { icon: Layers, label: 'Factor Analysis', active: true },
  { icon: LineChart, label: 'Backtesting', active: true },
  { icon: Database, label: 'Market Data', active: true },
  { icon: Newspaper, label: 'News Analysis', active: false },
  { icon: FileText, label: 'Report Generator', active: true },
];

const FUTURE_CAPABILITIES = [
  { icon: Brain, label: 'MCP Tools', active: false },
  { icon: Bot, label: 'Multi-Agent Collab', active: false },
];

export function AIPanel() {
  const [messages, setMessages] = useState<AIMessage[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: "Hello, I'm **AlphaMamba Research Agent**. I have access to the prediction engine, factor analysis tools, and market data.\n\nAsk me anything about today's predictions, market conditions, or request a research report.",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;
    const userMsg: AIMessage = { id: Date.now().toString(), role: 'user', content: input, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    // Simulate AI response (would connect to backend AI agent)
    setTimeout(() => {
      const response = generateAIResponse(input);
      const aiMsg: AIMessage = { id: (Date.now() + 1).toString(), role: 'assistant', content: response, timestamp: new Date() };
      setMessages(prev => [...prev, aiMsg]);
      setIsTyping(false);
    }, 800 + Math.random() * 1200);
  };

  const handleQuickPrompt = (prompt: string) => {
    setInput(prompt);
  };

  return (
    <aside className="fixed right-0 top-0 h-screen w-[25%] min-w-[320px] max-w-[420px] bg-surface border-l border-border flex flex-col z-40">
      {/* Header */}
      <div className="px-4 py-4 border-b border-border">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-accent/20 flex items-center justify-center">
            <Bot className="w-3.5 h-3.5 text-accent" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-text-primary">Research Agent</h2>
            <p className="text-[10px] text-text-secondary">AI-Powered Analysis</p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.map((msg) => (
          <div key={msg.id} className={msg.role === 'assistant' ? 'ai-message' : 'user-message'}>
            <div className="text-xs text-text-secondary mb-1">
              {msg.role === 'assistant' ? '🤖 AlphaMamba Agent' : 'You'}
            </div>
            <div className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</div>
          </div>
        ))}

        {isTyping && (
          <div className="ai-message">
            <div className="flex items-center gap-2">
              <div className="flex gap-1">
                <div className="w-1.5 h-1.5 rounded-full bg-accent animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-1.5 h-1.5 rounded-full bg-accent animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-1.5 h-1.5 rounded-full bg-accent animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
              <span className="text-xs text-text-secondary">Analyzing...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Quick Prompts */}
      {messages.length <= 1 && (
        <div className="px-4 py-3 border-t border-border">
          <p className="text-[10px] text-text-secondary uppercase tracking-wider mb-2">Suggested Questions</p>
          <div className="space-y-1">
            {QUICK_PROMPTS.slice(0, 4).map((prompt) => (
              <button
                key={prompt}
                onClick={() => handleQuickPrompt(prompt)}
                className="w-full text-left text-xs text-text-secondary hover:text-text-primary hover:bg-surface-elevated rounded px-2 py-1.5 transition-colors"
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Capabilities */}
      <div className="px-4 py-3 border-t border-border">
        <p className="text-[10px] text-text-secondary uppercase tracking-wider mb-2">Available Tools</p>
        <div className="space-y-1">
          {CAPABILITIES.map((cap) => (
            <div key={cap.label} className="flex items-center gap-2 text-xs">
              <cap.icon className={cn("w-3 h-3", cap.active ? "text-positive" : "text-text-secondary/40")} />
              <span className={cap.active ? "text-text-primary" : "text-text-secondary/40"}>{cap.label}</span>
              {cap.active && <span className="ml-auto text-[10px] text-positive">✓</span>}
            </div>
          ))}
        </div>
        <div className="mt-2 pt-2 border-t border-border/50">
          <p className="text-[10px] text-text-secondary/50 mb-1">Coming Soon</p>
          {FUTURE_CAPABILITIES.map((cap) => (
            <div key={cap.label} className="flex items-center gap-2 text-xs opacity-40">
              <cap.icon className="w-3 h-3" />
              <span>{cap.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Input */}
      <div className="px-4 py-3 border-t border-border">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask about predictions, factors, risk..."
            className="input-dark flex-1"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim()}
            className="p-2 rounded-md bg-accent hover:bg-accent-hover text-white disabled:opacity-30 transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </aside>
  );
}

function generateAIResponse(input: string): string {
  const lower = input.toLowerCase();
  if (lower.includes('rank') || lower.includes('why') || lower.includes('top')) {
    return "Based on the current model prediction, the top-ranked stocks show strong multi-horizon momentum (T+3/T+5/T+10) and favorable factor exposures.\n\n**Key drivers:**\n• Bull regime strength: 0.82 (strong bullish signal)\n• Top contributing factors: ret1_slope60, rsi_ac1, beta_20\n• Consensus overlap: 4/5 stocks appear in both Bull and NonBull models\n\nThis suggests high confidence in the current ranking.";
  }
  if (lower.includes('factor') || lower.includes('alpha')) {
    return "**Today's Strongest Alpha Factors:**\n\n1. **ret1_slope60** (IC: 0.042) — Price trend acceleration over 60-day window\n2. **vol_shock5** (IC: -0.031) — Short-term volatility shock reversal\n3. **rank_alpha20** (IC: 0.028) — Cross-sectional alpha ranking\n4. **beta_20** (IC: 0.025) — Market sensitivity exposure\n\nFactor drift is within normal range (±0.5σ). No significant regime shift detected.";
  }
  if (lower.includes('compare') || lower.includes('vs')) {
    return "**Stock Comparison Analysis:**\n\nBoth stocks are in the top-5 selection. Key differences:\n\n| Metric | Stock A | Stock B |\n|--------|---------|--------|\n| Prediction Score | 0.92 | 0.87 |\n| Bull Model Score | 0.94 | 0.83 |\n| NonBull Score | 0.89 | 0.91 |\n| Sector | Tech | Finance |\n\nThe consensus model slightly favors Stock A due to stronger bull-market signal alignment.";
  }
  if (lower.includes('report') || lower.includes('research')) {
    return "📄 **Generating Research Report...**\n\nI'll create a comprehensive report covering:\n\n• **Executive Summary** — Market regime analysis and key findings\n• **Factor Analysis** — Top contributing factors and IC trends\n• **Portfolio Review** — Current holdings and risk exposure\n• **Forecast** — 5-day and 10-day outlook\n\nThe report will be available in the Research Reports section. Would you like me to proceed with the full analysis?";
  }
  if (lower.includes('risk')) {
    return "**Risk Assessment:**\n\nCurrent portfolio risk metrics:\n• Max sector concentration: 2/5 (within 2-stock limit)\n• Correlation between holdings: 0.34 (well diversified)\n• Max drawdown (60d): -3.2%\n• Bull Gate status: PASS (BS 0.82 > 0.60)\n\n**Risk Factors to Monitor:**\n• Elevated volatility shock (vol_shock5 Z-score: +1.8σ)\n• Sector rotation risk in Finance sector";
  }
  if (lower.includes('market') || lower.includes('condition') || lower.includes('summarize')) {
    return "**Market Condition Summary:**\n\n🌡️ **Regime:** Strong Bull (BS: 0.82)\n📊 **Active Stocks:** 287/300 CSI 300 constituents\n📈 **Market Breadth:** 72% stocks above 20-day MA\n\n**Assessment:** The market is showing strong bullish momentum with broad participation. The Bull/NonBull soft blend is weighted 82% toward the Bull model. Confidence in predictions is **high**.";
  }
  return "I understand you're asking about: *" + input + "*\n\nBased on the current model state and market data, I can help analyze this. Could you provide more specifics about what aspects you'd like me to focus on?\n\nI can assist with:\n• Stock-specific analysis\n• Factor contribution breakdown\n• Risk assessment\n• Backtesting results\n• Research report generation";
}
