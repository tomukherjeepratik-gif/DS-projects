import React, { useState } from 'react';
import { 
  Layers, 
  Search, 
  Percent, 
  ArrowUpRight, 
  Zap 
} from 'lucide-react';
import type { RAGChunk, SearchResult, RAGConfig } from '../types/rag';
import { RAGEngine } from '../services/ragEngine';

interface ChunkInspectorTabProps {
  chunks: RAGChunk[];
  ragEngine: RAGEngine;
  config: RAGConfig;
  onSelectChunk: (result: SearchResult) => void;
}

export const ChunkInspectorTab: React.FC<ChunkInspectorTabProps> = ({
  chunks,
  ragEngine,
  config,
  onSelectChunk
}) => {
  const [testQuery, setTestQuery] = useState('tuition fees and scholarships for B.Tech');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearched, setIsSearched] = useState(false);
  const [selectedCategoryFilter, setSelectedCategoryFilter] = useState('All');

  const handleTestSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!testQuery.trim()) return;
    const results = ragEngine.search(testQuery.trim(), 10);
    setSearchResults(results);
    setIsSearched(true);
  };

  const categories = ['All', 'Courses', 'Fees', 'Clubs', 'Timetables', 'Faculty', 'Rules'];

  const filteredChunks = chunks.filter(c => 
    selectedCategoryFilter === 'All' || c.category === selectedCategoryFilter
  );

  return (
    <div className="glass-panel p-4 md:p-6 min-h-[600px] space-y-6">
      
      {/* Top Banner */}
      <div className="border-b border-[var(--border-subtle)] pb-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Layers className="w-5 h-5 text-cyan-400" />
              <span>Vector & Chunk Inspector</span>
            </h2>
            <p className="text-xs text-[var(--text-muted)]">
              Test dense vector embedding search, BM25 term matching, and chunk score ranking
            </p>
          </div>

          <div className="flex items-center gap-3 text-xs font-mono">
            <span className="badge badge-indigo">{chunks.length} Total Chunks</span>
            <span className="badge badge-cyan">{config.chunkSize} W/Chunk</span>
            <span className="badge badge-emerald">{config.chunkOverlap} W Overlap</span>
          </div>
        </div>
      </div>

      {/* Query Search Simulator Box */}
      <div className="bg-[#090d16] p-4 rounded-2xl border border-indigo-500/30 shadow-lg">
        <form onSubmit={handleTestSearch} className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="w-4 h-4 absolute left-3.5 top-3.5 text-[var(--text-dim)]" />
            <input
              type="text"
              value={testQuery}
              onChange={e => setTestQuery(e.target.value)}
              placeholder="Enter search query to calculate cosine vector similarity scores..."
              className="w-full bg-[#0f172a] border border-[var(--border-subtle)] focus:border-cyan-500 rounded-xl py-3 pl-10 pr-4 text-xs text-white placeholder-[var(--text-dim)] focus:outline-none"
            />
          </div>

          <button type="submit" className="btn-primary text-xs shrink-0 py-3 px-5">
            <Zap className="w-4 h-4" />
            Calculate Vector Scores
          </button>
        </form>
      </div>

      {/* Query Retrieval Ranking Results */}
      {isSearched && searchResults.length > 0 && (
        <div className="animate-slide-up">
          <h3 className="text-xs font-bold text-cyan-400 uppercase tracking-wider mb-3 flex items-center gap-2 font-mono">
            <Percent className="w-4 h-4" /> Top Ranked Retrieved Vector Chunks ({searchResults.length} Matches):
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {searchResults.map((res, idx) => (
              <div
                key={res.chunk.chunkId}
                onClick={() => onSelectChunk(res)}
                className="glass-card p-4 border-indigo-500/30 hover:border-cyan-400 cursor-pointer relative overflow-hidden group"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="badge badge-indigo text-[10px]">Rank #{idx + 1}</span>
                  <span className="font-mono text-xs font-bold text-cyan-400">
                    {(res.score * 100).toFixed(1)}% Match
                  </span>
                </div>

                <h4 className="text-xs font-bold text-white line-clamp-1 mb-1">{res.chunk.docTitle}</h4>
                <p className="text-[11px] text-indigo-300 font-medium mb-2">{res.chunk.section}</p>
                <p className="text-[11px] text-[var(--text-muted)] line-clamp-3 bg-[#090d16] p-2.5 rounded-lg border border-[var(--border-subtle)] font-mono">
                  "{res.chunk.text}"
                </p>

                <div className="mt-3 pt-2 border-t border-[var(--border-subtle)] flex items-center justify-between text-[10px] font-mono text-[var(--text-dim)]">
                  <span>Cosine: {(res.cosineScore * 100).toFixed(1)}%</span>
                  <span>BM25: {(res.bm25Score * 100).toFixed(1)}%</span>
                  <span className="text-cyan-400 group-hover:underline flex items-center gap-0.5">
                    View <ArrowUpRight className="w-3 h-3" />
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* All Indexed Chunks Grid */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider font-mono">
            All Corpus Vector Chunks ({filteredChunks.length} shown)
          </h3>

          <div className="flex gap-1 overflow-x-auto">
            {categories.map(cat => (
              <button
                key={cat}
                onClick={() => setSelectedCategoryFilter(cat)}
                className={`px-2.5 py-1 rounded text-[11px] font-semibold ${
                  selectedCategoryFilter === cat
                    ? 'bg-indigo-600 text-white'
                    : 'bg-white/5 text-[var(--text-muted)] hover:text-white'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-h-[400px] overflow-y-auto pr-1">
          {filteredChunks.map((chunk) => (
            <div key={chunk.chunkId} className="glass-card p-3.5 border-[var(--border-subtle)]">
              <div className="flex items-center justify-between mb-1.5">
                <span className="font-mono text-[10px] text-cyan-400 font-bold">{chunk.chunkId}</span>
                <span className="badge badge-indigo text-[9px]">{chunk.category}</span>
              </div>
              <h4 className="text-xs font-bold text-white line-clamp-1">{chunk.docTitle}</h4>
              <p className="text-[11px] text-indigo-300 line-clamp-1 mb-2">{chunk.section}</p>
              <p className="text-[11px] text-[var(--text-muted)] line-clamp-3 font-mono bg-[#090d16] p-2 rounded border border-[var(--border-subtle)]">
                {chunk.text}
              </p>
              <div className="mt-2 text-[10px] font-mono text-[var(--text-dim)] flex items-center justify-between">
                <span>{chunk.tokenCount} Words</span>
                <span>{chunk.tokens.length} Unique Tokens</span>
              </div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
};
