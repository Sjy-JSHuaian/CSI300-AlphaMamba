import { useState } from 'react';
import { cn } from '@/lib/utils';
import {
  LayoutDashboard, TrendingUp, Database, Layers, Activity,
  BarChart3, FileText, Bot, Settings, Zap, ChevronRight
} from 'lucide-react';

interface NavItem {
  id: string;
  label: string;
  icon: React.ElementType;
}

const navItems: NavItem[] = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'predictions', label: 'Predictions', icon: TrendingUp },
  { id: 'universe', label: 'Stock Universe', icon: Database },
  { id: 'factors', label: 'Factor Library', icon: Layers },
  { id: 'market', label: 'Market Analysis', icon: Activity },
  { id: 'backtesting', label: 'Backtesting', icon: BarChart3 },
  { id: 'reports', label: 'Research Reports', icon: FileText },
  { id: 'agent', label: 'Agent Workspace', icon: Bot },
  { id: 'settings', label: 'Settings', icon: Settings },
];

interface SidebarProps {
  activePage: string;
  onNavigate: (page: string) => void;
}

export function Sidebar({ activePage, onNavigate }: SidebarProps) {
  return (
    <aside className="fixed left-0 top-0 h-screen w-[18%] min-w-[220px] max-w-[280px] bg-surface border-r border-border flex flex-col z-40">
      {/* Logo */}
      <div className="px-4 py-5 border-b border-border">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center">
            <Zap className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-text-primary tracking-tight">AlphaMamba</h1>
            <p className="text-[10px] text-text-secondary uppercase tracking-widest">Research Platform</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activePage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={cn(
                "sidebar-link w-full text-left",
                isActive && "active"
              )}
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              <span className="text-sm">{item.label}</span>
              {isActive && <ChevronRight className="w-3 h-3 ml-auto" />}
            </button>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-border">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-positive animate-pulse" />
          <span className="text-xs text-text-secondary">System Online</span>
        </div>
        <p className="text-[10px] text-text-secondary mt-1">AlphaMamba v2.0</p>
      </div>
    </aside>
  );
}
