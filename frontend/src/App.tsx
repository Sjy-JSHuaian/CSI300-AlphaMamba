import { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Sidebar } from '@/components/layout/Sidebar';
import { AIPanel } from '@/components/layout/AIPanel';
import { Dashboard } from '@/pages/Dashboard';
import { Predictions } from '@/pages/Predictions';
import { StockUniverse } from '@/pages/StockUniverse';
import { FactorLibrary } from '@/pages/FactorLibrary';
import { MarketAnalysis } from '@/pages/MarketAnalysis';
import { Backtesting } from '@/pages/Backtesting';
import { ResearchReports } from '@/pages/ResearchReports';
import { AgentWorkspace } from '@/pages/AgentWorkspace';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60000,
      retry: 1,
    },
  },
});

function App() {
  const [activePage, setActivePage] = useState('dashboard');

  const renderPage = () => {
    switch (activePage) {
      case 'dashboard': return <Dashboard />;
      case 'predictions': return <Predictions />;
      case 'universe': return <StockUniverse />;
      case 'factors': return <FactorLibrary />;
      case 'market': return <MarketAnalysis />;
      case 'backtesting': return <Backtesting />;
      case 'reports': return <ResearchReports />;
      case 'agent': return <AgentWorkspace />;
      default: return <Dashboard />;
    }
  };

  return (
    <QueryClientProvider client={queryClient}>
      <div className="flex h-screen bg-background">
        <Sidebar activePage={activePage} onNavigate={setActivePage} />
        <main className="ml-[18%] mr-[25%] flex-1 min-h-screen overflow-y-auto">
          <div className="p-6">
            {renderPage()}
          </div>
        </main>
        <AIPanel />
      </div>
    </QueryClientProvider>
  );
}

export default App;
