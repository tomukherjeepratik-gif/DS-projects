import React from 'react';
import { X, Cpu, Layers, Database, ArrowRight, FileText, Bot, MessageSquare } from 'lucide-react';

interface ArchitectureModalProps {
  onClose: () => void;
}

export const ArchitectureModal: React.FC<ArchitectureModalProps> = ({ onClose }) => {
  return (
    <div className="modal-overlay animate-slide-up" onClick={onClose}>
      <div 
        className="glass-panel max-w-4xl w-full p-6 relative border-indigo-500/40 shadow-2xl max-h-[90vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="flex items-start justify-between border-b border-[var(--border-subtle)] pb-4 mb-5">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-indigo-500/15 border border-indigo-500/30 text-indigo-400">
              <Cpu className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-white">Nexus RAG System Architecture</h3>
              <p className="text-xs text-[var(--text-muted)]">
                End-to-End Pipeline Breakdown: Retrieval-Augmented Generation & Vector Embeddings
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

        {/* Visual Pipeline Diagram Flow */}
        <div className="bg-[#090d16] p-5 rounded-2xl border border-indigo-500/30 mb-6">
          <h4 className="text-xs font-bold text-indigo-400 uppercase tracking-wider mb-4 font-mono flex items-center gap-2">
            <Layers className="w-4 h-4" /> Live Execution Pipeline Flow:
          </h4>

          <div className="grid grid-cols-1 md:grid-cols-5 gap-3 items-center text-center">
            
            {/* Step 1 */}
            <div className="p-3 rounded-xl bg-indigo-950/40 border border-indigo-500/30 flex flex-col items-center">
              <FileText className="w-6 h-6 text-indigo-400 mb-1" />
              <span className="text-xs font-bold text-white">1. Corpus Docs</span>
              <span className="text-[10px] text-[var(--text-muted)] mt-1">Courses, Fees, Rules, Faculty, Clubs</span>
            </div>

            <div className="hidden md:flex justify-center text-indigo-500">
              <ArrowRight className="w-5 h-5" />
            </div>

            {/* Step 2 */}
            <div className="p-3 rounded-xl bg-cyan-950/40 border border-cyan-500/30 flex flex-col items-center">
              <Layers className="w-6 h-6 text-cyan-400 mb-1" />
              <span className="text-xs font-bold text-white">2. Chunking</span>
              <span className="text-[10px] text-[var(--text-muted)] mt-1">150 Words Window / 30 Words Overlap</span>
            </div>

            <div className="hidden md:flex justify-center text-cyan-500">
              <ArrowRight className="w-5 h-5" />
            </div>

            {/* Step 3 */}
            <div className="p-3 rounded-xl bg-purple-950/40 border border-purple-500/30 flex flex-col items-center">
              <Database className="w-6 h-6 text-purple-400 mb-1" />
              <span className="text-xs font-bold text-white">3. Vector Search</span>
              <span className="text-[10px] text-[var(--text-muted)] mt-1">TF-IDF Vector Space + BM25 Hybrid</span>
            </div>

          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-center text-center mt-3">
            
            <div className="hidden md:flex justify-center text-purple-500 col-span-3">
              <ArrowRight className="w-5 h-5 rotate-90" />
            </div>

            {/* Step 4 */}
            <div className="p-3 rounded-xl bg-amber-950/40 border border-amber-500/30 flex flex-col items-center">
              <MessageSquare className="w-6 h-6 text-amber-400 mb-1" />
              <span className="text-xs font-bold text-white">4. Top-K Context</span>
              <span className="text-[10px] text-[var(--text-muted)] mt-1">Retrieves Top 3 Relevant Chunks</span>
            </div>

            <div className="hidden md:flex justify-center text-amber-500">
              <ArrowRight className="w-5 h-5" />
            </div>

            {/* Step 5 */}
            <div className="p-3 rounded-xl bg-emerald-950/40 border border-emerald-500/30 flex flex-col items-center">
              <Bot className="w-6 h-6 text-emerald-400 mb-1" />
              <span className="text-xs font-bold text-white">5. Grounded Answer</span>
              <span className="text-[10px] text-[var(--text-muted)] mt-1">Synthesizes LLM response + Source Citations</span>
            </div>

          </div>
        </div>

        {/* Core Concepts Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 rounded-xl bg-white/5 border border-[var(--border-subtle)]">
            <h5 className="text-sm font-bold text-indigo-300 flex items-center gap-2 mb-2">
              📌 Why Chunking & Overlap Matter?
            </h5>
            <p className="text-xs text-[var(--text-muted)] leading-relaxed">
              LLMs have context window constraints and retrieval noise issues. Chunking breaks large Markdown documents into logical segments. Overlapping words (30 words) prevents cutting sentences in half and ensures context continuity across boundaries.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-white/5 border border-[var(--border-subtle)]">
            <h5 className="text-sm font-bold text-cyan-300 flex items-center gap-2 mb-2">
              ⚡ Vector Embeddings & Hybrid Retrieval
            </h5>
            <p className="text-xs text-[var(--text-muted)] leading-relaxed">
              We compute TF-IDF vector embeddings with sublinear term-frequency scaling and L2 normalization, paired with BM25 keyword matching. This ensures both semantic similarity and exact entity precision (e.g. course codes like CS204 or fee amounts).
            </p>
          </div>

          <div className="p-4 rounded-xl bg-white/5 border border-[var(--border-subtle)]">
            <h5 className="text-sm font-bold text-emerald-300 flex items-center gap-2 mb-2">
              🛡️ Zero Hallucination Guarantee
            </h5>
            <p className="text-xs text-[var(--text-muted)] leading-relaxed">
              The RAG generator is strictly grounded in retrieved chunks. If the system does not find high-confidence chunks for a query, it refrains from guessing and explicitly notifies the user.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-white/5 border border-[var(--border-subtle)]">
            <h5 className="text-sm font-bold text-amber-300 flex items-center gap-2 mb-2">
              💬 Multi-turn Conversation Memory (Bonus)
            </h5>
            <p className="text-xs text-[var(--text-muted)] leading-relaxed">
              Follow-up queries (e.g. "Who is the professor for it?") append recent conversation context to expand vector search queries, enabling natural multi-turn dialogue while preserving precise retrieval.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-6 flex justify-end">
          <button onClick={onClose} className="btn-primary text-xs px-5 py-2">
            Got it
          </button>
        </div>

      </div>
    </div>
  );
};
