import React from 'react';
import { 
  Bot, 
  BookOpen, 
  Layers, 
  BarChart3, 
  Settings, 
  Cpu, 
  GraduationCap
} from 'lucide-react';

interface HeaderProps {
  activeTab: 'chat' | 'knowledge' | 'inspector' | 'benchmark';
  setActiveTab: (tab: 'chat' | 'knowledge' | 'inspector' | 'benchmark') => void;
  onOpenSettings: () => void;
  onOpenArchitecture: () => void;
  docCount: number;
  chunkCount: number;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  onOpenSettings,
  onOpenArchitecture,
  docCount,
  chunkCount
}) => {
  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-[var(--border-subtle)] px-6 py-3.5 mb-6">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        
        {/* Brand Section */}
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-cyan-500 shadow-[0_0_20px_rgba(99,102,241,0.5)]">
            <GraduationCap className="w-6 h-6 text-white" />
            <span className="absolute -top-1 -right-1 w-3 h-3 bg-emerald-400 rounded-full border-2 border-[#090d16] animate-pulse"></span>
          </div>

          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-indigo-200 to-cyan-300 bg-clip-text text-transparent">
                Nexus RAG AI
              </h1>
              <span className="badge badge-indigo text-[10px] py-0.5 px-2">v2.0 RAG Engine</span>
            </div>
            <p className="text-xs text-[var(--text-muted)] flex items-center gap-1.5 mt-0.5">
              <span>Nexus Institute of Technology</span>
              <span>•</span>
              <span className="text-indigo-400 font-medium">{docCount} Knowledge Docs</span>
              <span>•</span>
              <span className="text-cyan-400 font-medium">{chunkCount} Vector Chunks</span>
            </p>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav className="flex items-center p-1 bg-[#0f172a]/80 rounded-xl border border-[var(--border-subtle)] overflow-x-auto max-w-full">
          <button
            onClick={() => setActiveTab('chat')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-xs md:text-sm transition-all whitespace-nowrap ${
              activeTab === 'chat'
                ? 'bg-gradient-to-r from-indigo-600 to-indigo-700 text-white shadow-md'
                : 'text-[var(--text-muted)] hover:text-white hover:bg-white/5'
            }`}
          >
            <Bot className="w-4 h-4" />
            <span>RAG Assistant</span>
          </button>

          <button
            onClick={() => setActiveTab('knowledge')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-xs md:text-sm transition-all whitespace-nowrap ${
              activeTab === 'knowledge'
                ? 'bg-gradient-to-r from-indigo-600 to-indigo-700 text-white shadow-md'
                : 'text-[var(--text-muted)] hover:text-white hover:bg-white/5'
            }`}
          >
            <BookOpen className="w-4 h-4" />
            <span>College Corpus</span>
          </button>

          <button
            onClick={() => setActiveTab('inspector')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-xs md:text-sm transition-all whitespace-nowrap ${
              activeTab === 'inspector'
                ? 'bg-gradient-to-r from-indigo-600 to-indigo-700 text-white shadow-md'
                : 'text-[var(--text-muted)] hover:text-white hover:bg-white/5'
            }`}
          >
            <Layers className="w-4 h-4" />
            <span>Vector & Chunk Inspector</span>
          </button>

          <button
            onClick={() => setActiveTab('benchmark')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-xs md:text-sm transition-all whitespace-nowrap ${
              activeTab === 'benchmark'
                ? 'bg-gradient-to-r from-indigo-600 to-indigo-700 text-white shadow-md'
                : 'text-[var(--text-muted)] hover:text-white hover:bg-white/5'
            }`}
          >
            <BarChart3 className="w-4 h-4" />
            <span>Test Suite</span>
          </button>
        </nav>

        {/* Action Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={onOpenArchitecture}
            className="btn-secondary text-xs px-3 py-2 border-indigo-500/30 text-indigo-300 hover:text-white"
            title="View RAG System Architecture Diagram"
          >
            <Cpu className="w-3.5 h-3.5 text-indigo-400" />
            <span className="hidden sm:inline">RAG Architecture</span>
          </button>

          <button
            onClick={onOpenSettings}
            className="p-2 rounded-lg bg-white/5 border border-[var(--border-subtle)] text-[var(--text-muted)] hover:text-white hover:bg-white/10 transition-colors"
            title="RAG Parameters & LLM Settings"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>

      </div>
    </header>
  );
};
