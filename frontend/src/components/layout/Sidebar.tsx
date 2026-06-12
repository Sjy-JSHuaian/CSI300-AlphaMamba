import { useTranslation } from 'react-i18next';
import { cn } from '@/lib/utils';
import {
  LayoutDashboard, TrendingUp, Database, Layers, Activity,
  BarChart3, FileText, Bot, Settings, Zap, ChevronRight,
  Clock, TrendingDown, GitCompare
} from 'lucide-react';

interface NavItem {
  id: string;
  labelKey: string;
  icon: React.ElementType;
}

const navItems: NavItem[] = [
  { id: 'dashboard', labelKey: 'nav.dashboard', icon: LayoutDashboard },
  { id: 'predictions', labelKey: 'nav.predictions', icon: TrendingUp },
  { id: 'universe', labelKey: 'nav.stock_universe', icon: Database },
  { id: 'factors', labelKey: 'nav.factor_library', icon: Layers },
  { id: 'market', labelKey: 'nav.market_analysis', icon: Activity },
  { id: 'backtesting', labelKey: 'nav.backtesting', icon: BarChart3 },
  { id: 'history', labelKey: 'nav.history', icon: Clock },
  { id: 'comparison', labelKey: 'nav.comparison', icon: GitCompare },
  { id: 'performance', labelKey: 'nav.performance', icon: TrendingDown },
  { id: 'reports', labelKey: 'nav.research_reports', icon: FileText },
  { id: 'agent', labelKey: 'nav.agent_workspace', icon: Bot },
  { id: 'settings', labelKey: 'nav.settings', icon: Settings },
];

interface SidebarProps {
  activePage: string;
  onNavigate: (page: string) => void;
}

export function Sidebar({ activePage, onNavigate }: SidebarProps) {
  const { t } = useTranslation();

  return (
    <aside className="fixed left-0 top-0 h-screen w-[18%] min-w-[220px] max-w-[280px] bg-surface border-r border-border flex flex-col z-40">
      <div className="px-4 py-5 border-b border-border">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center">
            <Zap className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-text-primary tracking-tight">{t('app.title')}</h1>
            <p className="text-[10px] text-text-secondary uppercase tracking-widest">{t('app.subtitle')}</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activePage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={cn("sidebar-link w-full text-left", isActive && "active")}
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              <span className="text-sm">{t(item.labelKey)}</span>
              {isActive && <ChevronRight className="w-3 h-3 ml-auto" />}
            </button>
          );
        })}
      </nav>

      <div className="px-4 py-3 border-t border-border">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-positive animate-pulse" />
          <span className="text-xs text-text-secondary">{t('app.system_online')}</span>
        </div>
        <p className="text-[10px] text-text-secondary mt-1">{t('app.version')}</p>
      </div>
    </aside>
  );
}
