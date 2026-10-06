import React from 'react';
import { X, Settings, Sliders, Key, Database, RefreshCw } from 'lucide-react';
import type { RAGConfig } from '../types/rag';

interface SettingsModalProps {
  config: RAGConfig;
  onSaveConfig: (newConfig: RAGConfig) => void;
  onReindex: () => void;
  onClose: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  config,
  onSaveConfig,
  onReindex,
  onClose
}) => {
  const [localConfig, setLocalConfig] = React.useState<RAGConfig>({ ...config });

  const handleSave = () => {
    onSaveConfig(localConfig);
    onReindex();
    onClose();
  };

  return (
    <div className="modal-overlay animate-slide-up" onClick={onClose}>
      <div 
        className="glass-panel max-w-lg w-full p-6 relative border-indigo-500/40 shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="flex items-start justify-between border-b border-[var(--border-subtle)] pb-4 mb-5">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-indigo-500/15 border border-indigo-500/30 text-indigo-400">
              <Settings className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">RAG Engine Settings</h3>
              <p className="text-xs text-[var(--text-muted)]">Configure retrieval parameters & LLM integration</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-[var(--text-muted)] hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Sliders Form */}
        <div className="space-y-5">
          
          {/* Top-K Chunks */}
          <div>
            <div className="flex justify-between items-center mb-1.5">
              <label className="text-xs font-semibold text-white flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-indigo-400" />
                <span>Top-K Retrieved Chunks</span>
              </label>
              <span className="font-mono text-xs font-bold text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/30">
                {localConfig.topK} Chunks
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="5"
              step="1"
              value={localConfig.topK}
              onChange={e => setLocalConfig({ ...localConfig, topK: parseInt(e.target.value) })}
              className="w-full accent-indigo-500 bg-white/10 rounded-lg cursor-pointer"
            />
            <p className="text-[11px] text-[var(--text-muted)] mt-1">
              Number of top context chunks sent to the LLM per query.
            </p>
          </div>

          {/* Chunk Size */}
          <div>
            <div className="flex justify-between items-center mb-1.5">
              <label className="text-xs font-semibold text-white flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-cyan-400" />
                <span>Chunk Size (Words)</span>
              </label>
              <span className="font-mono text-xs font-bold text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/30">
                {localConfig.chunkSize} Words
              </span>
            </div>
            <input
              type="range"
              min="80"
              max="350"
              step="10"
              value={localConfig.chunkSize}
              onChange={e => setLocalConfig({ ...localConfig, chunkSize: parseInt(e.target.value) })}
              className="w-full accent-cyan-500 bg-white/10 rounded-lg cursor-pointer"
            />
          </div>

          {/* Chunk Overlap */}
          <div>
            <div className="flex justify-between items-center mb-1.5">
              <label className="text-xs font-semibold text-white">
                Chunk Overlap (Words)
              </label>
              <span className="font-mono text-xs font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/30">
                {localConfig.chunkOverlap} Words
              </span>
            </div>
            <input
              type="range"
              min="10"
              max="60"
              step="5"
              value={localConfig.chunkOverlap}
              onChange={e => setLocalConfig({ ...localConfig, chunkOverlap: parseInt(e.target.value) })}
              className="w-full accent-emerald-500 bg-white/10 rounded-lg cursor-pointer"
            />
          </div>

          {/* LLM Provider Selection */}
          <div className="pt-2 border-t border-[var(--border-subtle)]">
            <label className="text-xs font-semibold text-white block mb-2">
              LLM Generator Engine
            </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setLocalConfig({ ...localConfig, llmProvider: 'local' })}
                className={`p-3 rounded-xl border text-left text-xs font-semibold transition-all ${
                  localConfig.llmProvider === 'local'
                    ? 'bg-indigo-600/20 border-indigo-500 text-indigo-200'
                    : 'bg-white/5 border-[var(--border-subtle)] text-[var(--text-muted)]'
                }`}
              >
                🤖 Built-in Local RAG Generator
              </button>

              <button
                type="button"
                onClick={() => setLocalConfig({ ...localConfig, llmProvider: 'openai' })}
                className={`p-3 rounded-xl border text-left text-xs font-semibold transition-all ${
                  localConfig.llmProvider === 'openai'
                    ? 'bg-indigo-600/20 border-indigo-500 text-indigo-200'
                    : 'bg-white/5 border-[var(--border-subtle)] text-[var(--text-muted)]'
                }`}
              >
                🔑 OpenAI API (GPT-3.5/4)
              </button>
            </div>

            {localConfig.llmProvider === 'openai' && (
              <div className="mt-3">
                <label className="text-xs text-[var(--text-muted)] block mb-1 flex items-center gap-1">
                  <Key className="w-3.5 h-3.5 text-amber-400" /> OpenAI API Key:
                </label>
                <input
                  type="password"
                  placeholder="sk-..."
                  value={localConfig.apiKey || ''}
                  onChange={e => setLocalConfig({ ...localConfig, apiKey: e.target.value })}
                  className="w-full bg-[#090d16] border border-[var(--border-subtle)] rounded-lg p-2.5 text-xs text-white focus:outline-none focus:border-indigo-500 font-mono"
                />
              </div>
            )}
          </div>

        </div>

        {/* Modal Actions */}
        <div className="mt-6 pt-4 border-t border-[var(--border-subtle)] flex items-center justify-between">
          <button
            onClick={() => {
              onReindex();
            }}
            className="btn-secondary text-xs text-cyan-300 border-cyan-500/30 hover:bg-cyan-500/10"
          >
            <RefreshCw className="w-3.5 h-3.5" /> Re-index Vectors
          </button>

          <div className="flex gap-2">
            <button onClick={onClose} className="btn-secondary text-xs">
              Cancel
            </button>
            <button onClick={handleSave} className="btn-primary text-xs">
              Save & Apply Parameters
            </button>
          </div>
        </div>

      </div>
    </div>
  );
};
