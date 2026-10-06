import React, { useState } from 'react';
import { 
  BookOpen, 
  DollarSign, 
  Users, 
  Calendar, 
  GraduationCap, 
  ShieldAlert, 
  Plus, 
  Search, 
  FileText, 
  Sparkles
} from 'lucide-react';
import type { DocumentItem } from '../types/rag';

interface KnowledgeBaseTabProps {
  documents: DocumentItem[];
  onAddDocument: (doc: DocumentItem) => void;
}

export const KnowledgeBaseTab: React.FC<KnowledgeBaseTabProps> = ({ documents, onAddDocument }) => {
  const [selectedDocId, setSelectedDocId] = useState<string>(documents[0]?.id || '');
  const [activeCategory, setActiveCategory] = useState<string>('All');
  const [searchFilter, setSearchFilter] = useState<string>('');
  const [showAddModal, setShowAddModal] = useState<boolean>(false);

  // New Doc Form state
  const [newTitle, setNewTitle] = useState('');
  const [newCategory, setNewCategory] = useState<'Courses' | 'Fees' | 'Clubs' | 'Timetables' | 'Faculty' | 'Rules'>('Rules');
  const [newContent, setNewContent] = useState('');

  const categories = ['All', 'Courses', 'Fees', 'Clubs', 'Timetables', 'Faculty', 'Rules'];

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'Courses': return <BookOpen className="w-4 h-4 text-indigo-400" />;
      case 'Fees': return <DollarSign className="w-4 h-4 text-emerald-400" />;
      case 'Clubs': return <Users className="w-4 h-4 text-cyan-400" />;
      case 'Timetables': return <Calendar className="w-4 h-4 text-amber-400" />;
      case 'Faculty': return <GraduationCap className="w-4 h-4 text-purple-400" />;
      case 'Rules': return <ShieldAlert className="w-4 h-4 text-rose-400" />;
      default: return <FileText className="w-4 h-4 text-indigo-400" />;
    }
  };

  const filteredDocs = documents.filter(doc => {
    const matchesCategory = activeCategory === 'All' || doc.category === activeCategory;
    const matchesSearch = doc.title.toLowerCase().includes(searchFilter.toLowerCase()) ||
                          doc.content.toLowerCase().includes(searchFilter.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const selectedDoc = documents.find(d => d.id === selectedDocId) || filteredDocs[0] || documents[0];

  const handleCreateDocument = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTitle.trim() || !newContent.trim()) return;

    const newDoc: DocumentItem = {
      id: newTitle.toLowerCase().replace(/[^a-z0-9]/g, '_'),
      title: newTitle.trim(),
      category: newCategory,
      iconName: 'FileText',
      lastUpdated: new Date().toISOString().split('T')[0],
      content: newContent.trim()
    };

    onAddDocument(newDoc);
    setSelectedDocId(newDoc.id);
    setNewTitle('');
    setNewContent('');
    setShowAddModal(false);
  };

  return (
    <div className="glass-panel p-4 md:p-6 min-h-[600px] flex flex-col">
      
      {/* Top Controls Header */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-[var(--border-subtle)] pb-4 mb-5">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-indigo-400" />
            <span>Nexus Knowledge Base & Corpus Manager</span>
          </h2>
          <p className="text-xs text-[var(--text-muted)]">
            Explore authoritative college documents indexed into the RAG vector database
          </p>
        </div>

        <div className="flex items-center gap-3 w-full md:w-auto">
          {/* Search Filter input */}
          <div className="relative flex-1 md:w-64">
            <Search className="w-4 h-4 absolute left-3 top-3 text-[var(--text-dim)]" />
            <input
              type="text"
              placeholder="Search corpus..."
              value={searchFilter}
              onChange={e => setSearchFilter(e.target.value)}
              className="w-full bg-[#090d16] border border-[var(--border-subtle)] rounded-xl py-2 pl-9 pr-3 text-xs text-white placeholder-[var(--text-dim)] focus:outline-none focus:border-indigo-500"
            />
          </div>

          <button
            onClick={() => setShowAddModal(true)}
            className="btn-primary text-xs shrink-0 py-2 px-3"
          >
            <Plus className="w-4 h-4" />
            <span>Add Document</span>
          </button>
        </div>
      </div>

      {/* Category Pills Bar */}
      <div className="flex items-center gap-2 overflow-x-auto pb-3 mb-4">
        {categories.map(cat => (
          <button
            key={cat}
            onClick={() => setActiveCategory(cat)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all whitespace-nowrap ${
              activeCategory === cat
                ? 'bg-indigo-600 text-white shadow-md'
                : 'bg-white/5 border border-[var(--border-subtle)] text-[var(--text-muted)] hover:text-white'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* Main Split View: Left List | Right Document Preview */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 flex-1">
        
        {/* Left Document List */}
        <div className="lg:col-span-4 space-y-2.5 max-h-[550px] overflow-y-auto pr-1">
          {filteredDocs.map(doc => {
            const isSelected = doc.id === selectedDoc?.id;
            return (
              <button
                key={doc.id}
                onClick={() => setSelectedDocId(doc.id)}
                className={`w-full text-left p-3.5 rounded-xl border transition-all ${
                  isSelected
                    ? 'bg-indigo-600/20 border-indigo-500 shadow-md'
                    : 'bg-white/5 border-[var(--border-subtle)] hover:bg-white/10 hover:border-white/20'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    {getCategoryIcon(doc.category)}
                    <span className="text-xs font-bold text-white line-clamp-1">{doc.title}</span>
                  </div>
                  <span className="badge badge-indigo text-[9px] py-0 px-1.5 font-mono">{doc.category}</span>
                </div>
                <p className="text-[11px] text-[var(--text-muted)] line-clamp-2 mt-1">
                  {doc.content.substring(0, 120)}...
                </p>
                <div className="flex items-center justify-between text-[10px] text-[var(--text-dim)] mt-2 font-mono">
                  <span>Updated: {doc.lastUpdated}</span>
                  <span>{doc.content.split(/\s+/).length} Words</span>
                </div>
              </button>
            );
          })}
        </div>

        {/* Right Document Content Reader */}
        <div className="lg:col-span-8 bg-[#090d16] p-5 rounded-2xl border border-[var(--border-subtle)] flex flex-col max-h-[550px] overflow-y-auto">
          {selectedDoc ? (
            <div>
              <div className="flex items-center justify-between border-b border-[var(--border-subtle)] pb-3 mb-4">
                <div className="flex items-center gap-3">
                  {getCategoryIcon(selectedDoc.category)}
                  <div>
                    <h3 className="text-base font-bold text-white">{selectedDoc.title}</h3>
                    <p className="text-xs text-[var(--text-muted)] font-mono">
                      Doc ID: {selectedDoc.id} • Category: {selectedDoc.category}
                    </p>
                  </div>
                </div>
                <span className="badge badge-emerald text-[10px]">Indexed in Vector DB</span>
              </div>

              {/* Document Markdown content display */}
              <div className="text-xs md:text-sm text-slate-200 leading-relaxed font-sans whitespace-pre-wrap">
                {selectedDoc.content}
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-[var(--text-muted)] text-xs">
              Select a document to read
            </div>
          )}
        </div>

      </div>

      {/* Add New Document Modal */}
      {showAddModal && (
        <div className="modal-overlay animate-slide-up" onClick={() => setShowAddModal(false)}>
          <div 
            className="glass-panel max-w-lg w-full p-6 relative border-indigo-500/40 shadow-2xl"
            onClick={e => e.stopPropagation()}
          >
            <h3 className="text-base font-bold text-white mb-1">Add Document to College Corpus</h3>
            <p className="text-xs text-[var(--text-muted)] mb-4">
              Add new college guidelines or course syllabi to automatically chunk and embed into RAG memory.
            </p>

            <form onSubmit={handleCreateDocument} className="space-y-4">
              <div>
                <label className="text-xs font-semibold text-white block mb-1">Document Title</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Internship & Placement Guidelines 2026"
                  value={newTitle}
                  onChange={e => setNewTitle(e.target.value)}
                  className="w-full bg-[#090d16] border border-[var(--border-subtle)] rounded-lg p-2.5 text-xs text-white focus:outline-none focus:border-indigo-500"
                />
              </div>

              <div>
                <label className="text-xs font-semibold text-white block mb-1">Category</label>
                <select
                  value={newCategory}
                  onChange={e => setNewCategory(e.target.value as any)}
                  className="w-full bg-[#090d16] border border-[var(--border-subtle)] rounded-lg p-2.5 text-xs text-white focus:outline-none focus:border-indigo-500"
                >
                  <option value="Courses">Courses & Syllabus</option>
                  <option value="Fees">Fees & Scholarships</option>
                  <option value="Clubs">Clubs & Societies</option>
                  <option value="Timetables">Timetables & Schedule</option>
                  <option value="Faculty">Faculty Directory</option>
                  <option value="Rules">Rules & Policies</option>
                </select>
              </div>

              <div>
                <label className="text-xs font-semibold text-white block mb-1">Document Markdown Content</label>
                <textarea
                  required
                  rows={6}
                  placeholder="# Enter markdown text content here..."
                  value={newContent}
                  onChange={e => setNewContent(e.target.value)}
                  className="w-full bg-[#090d16] border border-[var(--border-subtle)] rounded-lg p-2.5 text-xs text-white focus:outline-none focus:border-indigo-500 font-mono"
                />
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="btn-secondary text-xs"
                >
                  Cancel
                </button>
                <button type="submit" className="btn-primary text-xs">
                  <Sparkles className="w-3.5 h-3.5" />
                  Index Document into RAG
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

    </div>
  );
};
