import React from 'react';
import { X, FileText, Hash, Percent, Layers, CheckCircle2 } from 'lucide-react';
import type { SearchResult } from '../types/rag';

interface SourceChunkModalProps {
  result: SearchResult | null;
  onClose: () => void;
}

export const SourceChunkModal: React.FC<SourceChunkModalProps> = ({ result, onClose }) => {
  if (!result) return null;

  const { chunk, score, cosineScore, bm25Score, matchedTerms } = result;

  // Highlight matched terms in chunk text
  const highlightText = (text: string, terms: string[]) => {
    if (!terms || terms.length === 0) return text;
    
    // Create regex pattern safely
    const escapedTerms = terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    const pattern = new RegExp(`\\b(${escapedTerms.join('|')})\\b`, 'gi');

    const parts = text.split(pattern);
    return parts.map((part, i) => {
      const isMatch = terms.some(t => t.toLowerCase() === part.toLowerCase());
      return isMatch ? (
        <mark key={i} className="bg-indigo-500/30 text-indigo-200 border border-indigo-500/50 px-1 py-0.5 rounded font-semibold">
          {part}
        </mark>
      ) : (
        part
      );
    });
  };

  return (
    <div className="modal-overlay animate-slide-up" onClick={onClose}>
      <div 
        className="glass-panel max-w-2xl w-full p-6 relative border-indigo-500/40 shadow-2xl overflow-hidden"
        onClick={e => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="flex items-start justify-between border-b border-[var(--border-subtle)] pb-4 mb-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-indigo-500/15 border border-indigo-500/30 text-indigo-400">
              <FileText className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-lg font-bold text-white">{chunk.docTitle}</h3>
                <span className="badge badge-indigo text-[10px]">{chunk.category}</span>
              </div>
              <p className="text-xs text-[var(--text-muted)] mt-0.5 flex items-center gap-2">
                <span className="text-indigo-300 font-medium">{chunk.section}</span>
                <span>•</span>
                <span className="font-mono text-cyan-400">ID: {chunk.chunkId}</span>
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-[var(--text-muted)] hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Vector Similarity Metrics Bar */}
        <div className="grid grid-cols-3 gap-3 mb-5">
          <div className="p-3 rounded-xl bg-indigo-950/40 border border-indigo-500/20 text-center">
            <div className="text-xs text-[var(--text-muted)] flex items-center justify-center gap-1">
              <Percent className="w-3.5 h-3.5 text-indigo-400" />
              <span>Hybrid Score</span>
            </div>
            <p className="text-xl font-bold text-indigo-400 font-mono mt-0.5">
              {(score * 100).toFixed(1)}%
            </p>
          </div>

          <div className="p-3 rounded-xl bg-cyan-950/40 border border-cyan-500/20 text-center">
            <div className="text-xs text-[var(--text-muted)] flex items-center justify-center gap-1">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              <span>Cosine Vector Sim</span>
            </div>
            <p className="text-xl font-bold text-cyan-400 font-mono mt-0.5">
              {(cosineScore * 100).toFixed(1)}%
            </p>
          </div>

          <div className="p-3 rounded-xl bg-emerald-950/40 border border-emerald-500/20 text-center">
            <div className="text-xs text-[var(--text-muted)] flex items-center justify-center gap-1">
              <Hash className="w-3.5 h-3.5 text-emerald-400" />
              <span>BM25 Term Fit</span>
            </div>
            <p className="text-xl font-bold text-emerald-400 font-mono mt-0.5">
              {(bm25Score * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Matched Keywords */}
        {matchedTerms.length > 0 && (
          <div className="mb-4 flex items-center gap-2 flex-wrap">
            <span className="text-xs font-semibold text-[var(--text-muted)] flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> Matched Terms:
            </span>
            {matchedTerms.map((t, idx) => (
              <span key={idx} className="badge badge-emerald text-[11px] font-mono">
                {t}
              </span>
            ))}
          </div>
        )}

        {/* Chunk Content snippet */}
        <div className="bg-[#090d16] p-4 rounded-xl border border-[var(--border-subtle)] max-h-60 overflow-y-auto">
          <p className="text-xs font-semibold text-indigo-400 mb-2 uppercase tracking-wider font-mono">
            Raw Chunk Text Segment ({chunk.tokenCount} words):
          </p>
          <div className="text-sm leading-relaxed text-slate-200 whitespace-pre-wrap">
            {highlightText(chunk.text, matchedTerms)}
          </div>
        </div>

        {/* Footer */}
        <div className="mt-5 flex justify-end">
          <button onClick={onClose} className="btn-secondary text-xs px-4 py-2">
            Close Viewer
          </button>
        </div>

      </div>
    </div>
  );
};
