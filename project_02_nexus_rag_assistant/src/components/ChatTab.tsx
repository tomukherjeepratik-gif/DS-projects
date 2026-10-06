import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, 
  Trash2, 
  Download, 
  Sparkles, 
  Bot, 
  User, 
  BookOpen, 
  Layers
} from 'lucide-react';
import type { ChatMessage, SearchResult } from '../types/rag';

interface ChatTabProps {
  messages: ChatMessage[];
  onSendMessage: (query: string) => void;
  onClearHistory: () => void;
  onSelectSource: (source: SearchResult) => void;
  isProcessing: boolean;
}

const SAMPLE_SUGGESTIONS = [
  { label: '💰 B.Tech Fees & Scholarships', query: 'What is the tuition fee for B.Tech and what scholarships are available?' },
  { label: '💻 ACM Chapter Meetings & Hackathon', query: 'When does the ACM Student Chapter meet and what events do they organize?' },
  { label: '👩‍🏫 CS Department HoD & Office Hours', query: 'Who is the HOD of Computer Science and what are her office hours?' },
  { label: '📚 AI405 RAG Course & Prerequisites', query: 'What are the prerequisites for AI405 LLMs & RAG course?' },
  { label: '⏱️ Exam Period Library Timings', query: 'What time does the library close on regular days vs exam periods?' },
  { label: '⚠️ Attendance & Plagiarism Rules', query: 'What is the minimum attendance requirement and plagiarism policy?' },
];

export const ChatTab: React.FC<ChatTabProps> = ({
  messages,
  onSendMessage,
  onClearHistory,
  onSelectSource,
  isProcessing
}) => {
  const [inputQuery, setInputQuery] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isProcessing]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputQuery.trim() || isProcessing) return;
    onSendMessage(inputQuery.trim());
    setInputQuery('');
  };

  const handleExportChat = () => {
    const chatLog = messages.map(m => `[${m.timestamp}] ${m.sender.toUpperCase()}: ${m.text}`).join('\n\n');
    const blob = new Blob([chatLog], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `nexus_rag_chat_log_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-140px)] min-h-[600px] glass-panel p-4 md:p-6 relative overflow-hidden">
      
      {/* Top Controls Bar */}
      <div className="flex items-center justify-between border-b border-[var(--border-subtle)] pb-3.5 mb-4">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></div>
          <h2 className="text-sm font-bold text-white flex items-center gap-2">
            <span>NexusAI Conversational Assistant</span>
            <span className="badge badge-cyan text-[10px]">RAG Vector Grounded</span>
          </h2>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-[var(--text-dim)] hidden sm:inline-block font-mono">
            History: {messages.filter(m => m.sender === 'user').length} turns
          </span>

          {messages.length > 0 && (
            <>
              <button
                onClick={handleExportChat}
                className="btn-secondary text-xs px-2.5 py-1.5"
                title="Export Chat History"
              >
                <Download className="w-3.5 h-3.5" />
                <span className="hidden md:inline">Export</span>
              </button>

              <button
                onClick={onClearHistory}
                className="btn-secondary text-xs px-2.5 py-1.5 text-rose-300 hover:text-rose-200 border-rose-500/20 hover:bg-rose-500/10"
                title="Clear Conversation History"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span className="hidden md:inline">Clear</span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* Messages Feed */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-2">
        
        {/* Welcome Empty State */}
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center py-8 animate-slide-up">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-indigo-600/30 to-cyan-500/30 border border-indigo-500/40 flex items-center justify-center mb-4 text-indigo-400 shadow-[0_0_30px_rgba(99,102,241,0.25)]">
              <Bot className="w-8 h-8 text-indigo-300" />
            </div>

            <h3 className="text-xl font-bold text-white mb-1">
              Ask Anything About Nexus Institute of Technology
            </h3>
            <p className="text-xs text-[var(--text-muted)] max-w-md mb-6 leading-relaxed">
              I am your official RAG-based AI college assistant. I retrieve exact facts from college documents on <strong className="text-indigo-300">courses, fees, clubs, timetables, faculty</strong>, and <strong className="text-cyan-300">rules</strong> with source citations.
            </p>

            {/* Prompt Suggestion Cards */}
            <div className="w-full max-w-2xl text-left">
              <p className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-3 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-indigo-400" /> Suggested Questions:
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                {SAMPLE_SUGGESTIONS.map((item, idx) => (
                  <button
                    key={idx}
                    onClick={() => onSendMessage(item.query)}
                    className="p-3 rounded-xl bg-white/5 border border-[var(--border-subtle)] hover:border-indigo-500/50 hover:bg-indigo-600/10 text-left transition-all group cursor-pointer"
                  >
                    <p className="text-xs font-semibold text-indigo-300 group-hover:text-indigo-200 mb-0.5">
                      {item.label}
                    </p>
                    <p className="text-[11px] text-[var(--text-muted)] line-clamp-1">
                      "{item.query}"
                    </p>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Render Chat Messages */}
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex items-start gap-3 animate-slide-up ${
              msg.sender === 'user' ? 'flex-row-reverse' : 'flex-row'
            }`}
          >
            {/* Avatar */}
            <div
              className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 text-white font-bold text-xs ${
                msg.sender === 'user'
                  ? 'bg-gradient-to-tr from-cyan-600 to-blue-600 shadow-md'
                  : 'bg-gradient-to-tr from-indigo-600 to-purple-600 shadow-md'
              }`}
            >
              {msg.sender === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
            </div>

            {/* Message Card */}
            <div
              className={`max-w-[85%] md:max-w-[75%] p-4 rounded-2xl ${
                msg.sender === 'user'
                  ? 'bg-indigo-600 text-white rounded-tr-none shadow-lg'
                  : 'glass-card border-[var(--border-subtle)] text-slate-100 rounded-tl-none'
              }`}
            >
              <div className="text-xs opacity-60 mb-1 flex items-center gap-2">
                <span>{msg.sender === 'user' ? 'You' : 'NexusAI Assistant'}</span>
                <span>•</span>
                <span className="font-mono">{msg.timestamp}</span>
              </div>

              {/* Text Output with formatting */}
              <div className="text-sm leading-relaxed whitespace-pre-wrap">
                {msg.text}
              </div>

              {/* Sources Citation Bar (For Assistant Messages) */}
              {msg.sender === 'assistant' && msg.sources && msg.sources.length > 0 && (
                <div className="mt-4 pt-3 border-t border-[var(--border-subtle)]">
                  <p className="text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-2 flex items-center gap-1 font-mono">
                    <Layers className="w-3.5 h-3.5 text-indigo-400" /> Click Retrieved Source Chunks to Inspect:
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {msg.sources.map((src, sIdx) => (
                      <button
                        key={sIdx}
                        onClick={() => onSelectSource(src)}
                        className="citation-pill"
                        title={`Click to view chunk ${src.chunk.chunkId} details (${(src.score * 100).toFixed(1)}% match)`}
                      >
                        <BookOpen className="w-3 h-3" />
                        <span>{src.chunk.docTitle}</span>
                        <span className="text-[10px] opacity-75 font-mono">({src.chunk.chunkId})</span>
                        <span className="badge badge-emerald text-[9px] py-0 px-1 font-mono">
                          {(src.score * 100).toFixed(0)}%
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

          </div>
        ))}

        {/* Processing / Thinking Indicator */}
        {isProcessing && (
          <div className="flex items-center gap-3 animate-slide-up">
            <div className="w-8 h-8 rounded-xl bg-indigo-600/30 border border-indigo-500/50 flex items-center justify-center text-indigo-400">
              <Bot className="w-4 h-4 animate-spin" />
            </div>
            <div className="glass-card px-4 py-3 rounded-2xl rounded-tl-none flex items-center gap-3">
              <div className="flex gap-1">
                <span className="w-2 h-2 rounded-full bg-indigo-400 animate-bounce"></span>
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-bounce [animation-delay:0.2s]"></span>
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-bounce [animation-delay:0.4s]"></span>
              </div>
              <span className="text-xs text-[var(--text-muted)] font-mono">
                Searching Vector Index & Synthesizing Grounded Answer...
              </span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Chat Box */}
      <form onSubmit={handleSubmit} className="mt-4 pt-3 border-t border-[var(--border-subtle)]">
        <div className="relative flex items-center">
          <input
            type="text"
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            placeholder="Ask about courses, fees, ACM club, library hours, faculty or rules..."
            disabled={isProcessing}
            className="w-full bg-[#090d16] border border-[var(--border-subtle)] focus:border-indigo-500 rounded-xl py-3.5 pl-4 pr-12 text-sm text-white placeholder-[var(--text-dim)] focus:outline-none focus:ring-1 focus:ring-indigo-500/50 transition-all"
          />
          <button
            type="submit"
            disabled={!inputQuery.trim() || isProcessing}
            className="absolute right-2 p-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:hover:bg-indigo-600 text-white transition-all shadow-md"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </form>

    </div>
  );
};
