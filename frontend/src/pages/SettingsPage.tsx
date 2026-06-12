import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';
import { Globe, Database, RefreshCw, Download, Layers, Check } from 'lucide-react';

export function SettingsPage() {
  const { t, i18n } = useTranslation();
  const queryClient = useQueryClient();
  const currentLang = i18n.language?.startsWith('zh') ? 'zh-CN' : 'en';

  const [backfillStart, setBackfillStart] = useState('');
  const [backfillEnd, setBackfillEnd] = useState('');
  const [backfillMessage, setBackfillMessage] = useState<string | null>(null);
  const [updateMessage, setUpdateMessage] = useState<string | null>(null);

  const { data: snapshotsData } = useQuery({
    queryKey: ['snapshots-count'],
    queryFn: () => api.getSnapshots(1),
  });

  const backfillMutation = useMutation({
    mutationFn: () => api.backfillSnapshots(backfillStart || undefined, backfillEnd || undefined),
    onSuccess: (data: any) => {
      setBackfillMessage(data?.message || `Backfill completed: ${data?.count || 0} snapshots created`);
      queryClient.invalidateQueries({ queryKey: ['snapshots'] });
      queryClient.invalidateQueries({ queryKey: ['snapshots-count'] });
    },
    onError: (err: Error) => {
      setBackfillMessage(err.message);
    },
  });

  const updateMutation = useMutation({
    mutationFn: () => api.triggerUpdate(),
    onSuccess: (data: any) => {
      setUpdateMessage(data?.message || 'Update completed successfully');
      queryClient.invalidateQueries({ queryKey: ['snapshots'] });
      queryClient.invalidateQueries({ queryKey: ['snapshots-count'] });
      queryClient.invalidateQueries({ queryKey: ['predict'] });
      queryClient.invalidateQueries({ queryKey: ['status'] });
    },
    onError: (err: Error) => {
      setUpdateMessage(err.message);
    },
  });

  const handleBackfill = () => {
    setBackfillMessage(null);
    backfillMutation.mutate();
  };

  const handleUpdate = () => {
    setUpdateMessage(null);
    updateMutation.mutate();
  };

  const changeLanguage = (lang: string) => {
    i18n.changeLanguage(lang);
  };

  const snapshotCount = snapshotsData?.total ?? snapshotsData?.count ?? 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-text-primary">{t('settings.title')}</h1>
          <p className="text-sm text-text-secondary mt-1">{t('settings.subtitle')}</p>
        </div>
      </div>

      {/* Language Settings */}
      <div className="panel">
        <div className="panel-header">
          <div className="flex items-center gap-2">
            <Globe className="w-4 h-4 text-accent" />
            <h3 className="text-sm font-medium text-text-primary">{t('settings.language')}</h3>
          </div>
        </div>
        <div className="panel-body">
          <p className="text-sm text-text-secondary mb-4">{t('settings.language_description')}</p>
          <div className="space-y-3">
            <label className="flex items-center gap-3 p-3 rounded-lg bg-background border border-border cursor-pointer hover:border-accent/40 transition-colors">
              <input
                type="radio"
                name="language"
                value="en"
                checked={currentLang === 'en'}
                onChange={() => changeLanguage('en')}
                className="w-4 h-4 text-accent bg-surface border-border focus:ring-accent"
              />
              <div className="flex-1">
                <span className="text-sm font-medium text-text-primary">{t('settings.english')}</span>
                <span className="text-xs text-text-secondary ml-2">English</span>
              </div>
              {currentLang === 'en' && <Check className="w-4 h-4 text-accent" />}
            </label>

            <label className="flex items-center gap-3 p-3 rounded-lg bg-background border border-border cursor-pointer hover:border-accent/40 transition-colors">
              <input
                type="radio"
                name="language"
                value="zh-CN"
                checked={currentLang === 'zh-CN'}
                onChange={() => changeLanguage('zh-CN')}
                className="w-4 h-4 text-accent bg-surface border-border focus:ring-accent"
              />
              <div className="flex-1">
                <span className="text-sm font-medium text-text-primary">{t('settings.chinese')}</span>
                <span className="text-xs text-text-secondary ml-2">简体中文</span>
              </div>
              {currentLang === 'zh-CN' && <Check className="w-4 h-4 text-accent" />}
            </label>
          </div>
        </div>
      </div>

      {/* Data Management */}
      <div className="panel">
        <div className="panel-header">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-accent" />
            <h3 className="text-sm font-medium text-text-primary">{t('settings.data')}</h3>
          </div>
        </div>
        <div className="panel-body">
          {/* Snapshots Count */}
          <div className="flex items-center gap-4 p-4 rounded-lg bg-background border border-border mb-6">
            <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center">
              <Layers className="w-5 h-5 text-accent" />
            </div>
            <div>
              <p className="text-sm text-text-secondary">{t('settings.snapshots_count')}</p>
              <p className="text-2xl font-semibold text-text-primary">{snapshotCount}</p>
            </div>
          </div>

          {/* Backfill Section */}
          <div className="p-4 rounded-lg bg-background border border-border mb-4">
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1">
                <h4 className="text-sm font-medium text-text-primary mb-1">{t('settings.backfill')}</h4>
                <p className="text-xs text-text-secondary mb-3">{t('settings.backfill_description')}</p>
                <div className="flex items-center gap-3 mb-3">
                  <input
                    type="date"
                    value={backfillStart}
                    onChange={(e) => setBackfillStart(e.target.value)}
                    placeholder="Start date"
                    className="input-dark text-sm"
                  />
                  <span className="text-text-secondary text-xs">to</span>
                  <input
                    type="date"
                    value={backfillEnd}
                    onChange={(e) => setBackfillEnd(e.target.value)}
                    placeholder="End date"
                    className="input-dark text-sm"
                  />
                </div>
                {backfillMessage && (
                  <p className={`text-xs mb-2 ${backfillMutation.isError ? 'text-negative' : 'text-positive'}`}>
                    {backfillMessage}
                  </p>
                )}
              </div>
              <button
                onClick={handleBackfill}
                disabled={backfillMutation.isPending}
                className="btn-primary flex items-center gap-2 px-4 py-2 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {backfillMutation.isPending ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    {t('settings.backfill_running')}
                  </>
                ) : (
                  <>
                    <Download className="w-4 h-4" />
                    {t('settings.backfill_button')}
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Update Section */}
          <div className="p-4 rounded-lg bg-background border border-border">
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1">
                <h4 className="text-sm font-medium text-text-primary mb-1">{t('settings.update')}</h4>
                <p className="text-xs text-text-secondary mb-3">{t('settings.update_description')}</p>
                {updateMessage && (
                  <p className={`text-xs mb-2 ${updateMutation.isError ? 'text-negative' : 'text-positive'}`}>
                    {updateMessage}
                  </p>
                )}
              </div>
              <button
                onClick={handleUpdate}
                disabled={updateMutation.isPending}
                className="btn-primary flex items-center gap-2 px-4 py-2 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {updateMutation.isPending ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    {t('settings.update_running')}
                  </>
                ) : (
                  <>
                    <RefreshCw className="w-4 h-4" />
                    {t('settings.update_button')}
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
