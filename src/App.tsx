import { useState, useEffect, useMemo } from 'react';
import { Header } from './components/Header';
import { ChatTab } from './components/ChatTab';
import { KnowledgeBaseTab } from './components/KnowledgeBaseTab';
import { ChunkInspectorTab } from './components/ChunkInspectorTab';
import { BenchmarkTab } from './components/BenchmarkTab';
import { SourceChunkModal } from './components/SourceChunkModal';
import { ArchitectureModal } from './components/ArchitectureModal';
import { SettingsModal } from './components/SettingsModal';

import { INITIAL_DOCUMENTS } from './data/collegeData';
import type { DocumentItem, RAGChunk, SearchResult, ChatMessage, RAGConfig } from './types/rag';
import { RAGEngine } from './services/ragEngine';
import { LLMService } from './services/llmService';

const DEFAULT_CONFIG: RAGConfig = {
  topK: 3,
  chunkSize: 140,
  chunkOverlap: 30,
  similarityThreshold: 0.05,
  llmProvider: 'local'
};

export function App() {
  const [activeTab, setActiveTab] = useState<'chat' | 'knowledge' | 'inspector' | 'benchmark'>('chat');
  const [documents, setDocuments] = useState<DocumentItem[]>(() => {
    const saved = localStorage.getItem('nexus_college_docs');
    return saved ? JSON.parse(saved) : INITIAL_DOCUMENTS;
  });

  const [ragConfig, setRagConfig] = useState<RAGConfig>(() => {
    const saved = localStorage.getItem('nexus_rag_config');
    return saved ? JSON.parse(saved) : DEFAULT_CONFIG;
  });

  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    const saved = localStorage.getItem('nexus_chat_history');
    return saved ? JSON.parse(saved) : [];
  });

  const [isProcessing, setIsProcessing] = useState<boolean>(false);

  // Modals state
  const [selectedSourceModal, setSelectedSourceModal] = useState<SearchResult | null>(null);
  const [showArchitectureModal, setShowArchitectureModal] = useState<boolean>(false);
  const [showSettingsModal, setShowSettingsModal] = useState<boolean>(false);

  // Initialize RAG Engine & LLM Service
  const ragEngine = useMemo(() => new RAGEngine(ragConfig), [ragConfig]);
  const llmService = useMemo(() => new LLMService(), []);

  const [chunks, setChunks] = useState<RAGChunk[]>([]);

  // Re-index corpus whenever documents or config change
  const reindexCorpus = () => {
    ragEngine.updateConfig(ragConfig);
    const generatedChunks = ragEngine.ingestDocuments(documents);
    setChunks([...generatedChunks]);
  };

  useEffect(() => {
    reindexCorpus();
  }, [documents, ragConfig]);

  // Persist states to LocalStorage
  useEffect(() => {
    localStorage.setItem('nexus_college_docs', JSON.stringify(documents));
  }, [documents]);

  useEffect(() => {
    localStorage.setItem('nexus_rag_config', JSON.stringify(ragConfig));
  }, [ragConfig]);

  useEffect(() => {
    localStorage.setItem('nexus_chat_history', JSON.stringify(messages));
  }, [messages]);

  // Handle User Message Submission
  const handleSendMessage = async (userText: string) => {
    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const userMsg: ChatMessage = {
      id: `msg-${Date.now()}`,
      sender: 'user',
      text: userText,
      timestamp: now
    };

    const updatedMessages = [...messages, userMsg];
    setMessages(updatedMessages);
    setIsProcessing(true);

    try {
      // Build search query incorporating past context for follow-up turns
      let searchQuery = userText;
      if (messages.length > 0) {
        const lastUserMsg = [...messages].reverse().find(m => m.sender === 'user');
        if (lastUserMsg) {
          searchQuery = `${lastUserMsg.text} ${userText}`;
        }
      }

      // Perform Vector Search
      const searchResults = ragEngine.search(searchQuery, ragConfig.topK);

      // Synthesize Response
      const response = await llmService.generateRAGResponse(
        userText,
        searchResults,
        updatedMessages,
        ragConfig.llmProvider,
        ragConfig.apiKey
      );

      const assistantMsg: ChatMessage = {
        id: `msg-${Date.now() + 1}`,
        sender: 'assistant',
        text: response.text,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        sources: response.sources
      };

      setMessages(prev => [...prev, assistantMsg]);
    } catch (error) {
      console.error('RAG generation error:', error);
      const errorMsg: ChatMessage = {
        id: `msg-${Date.now() + 1}`,
        sender: 'assistant',
        text: 'An error occurred while retrieving vectors or generating response. Please check settings.',
        timestamp: now
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClearHistory = () => {
    setMessages([]);
    localStorage.removeItem('nexus_chat_history');
  };

  const handleAddDocument = (newDoc: DocumentItem) => {
    setDocuments(prev => [newDoc, ...prev]);
  };

  return (
    <div className="min-h-screen bg-[var(--bg-primary)] text-[var(--text-main)] flex flex-col">
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onOpenSettings={() => setShowSettingsModal(true)}
        onOpenArchitecture={() => setShowArchitectureModal(true)}
        docCount={documents.length}
        chunkCount={chunks.length}
      />

      <main className="max-w-7xl mx-auto px-4 md:px-6 flex-1 w-full pb-10">
        {activeTab === 'chat' && (
          <ChatTab
            messages={messages}
            onSendMessage={handleSendMessage}
            onClearHistory={handleClearHistory}
            onSelectSource={setSelectedSourceModal}
            isProcessing={isProcessing}
          />
        )}

        {activeTab === 'knowledge' && (
          <KnowledgeBaseTab
            documents={documents}
            onAddDocument={handleAddDocument}
          />
        )}

        {activeTab === 'inspector' && (
          <ChunkInspectorTab
            chunks={chunks}
            ragEngine={ragEngine}
            config={ragConfig}
            onSelectChunk={setSelectedSourceModal}
          />
        )}

        {activeTab === 'benchmark' && (
          <BenchmarkTab
            ragEngine={ragEngine}
            llmService={llmService}
          />
        )}
      </main>

      {/* Modals */}
      <SourceChunkModal
        result={selectedSourceModal}
        onClose={() => setSelectedSourceModal(null)}
      />

      {showArchitectureModal && (
        <ArchitectureModal onClose={() => setShowArchitectureModal(false)} />
      )}

      {showSettingsModal && (
        <SettingsModal
          config={ragConfig}
          onSaveConfig={setRagConfig}
          onReindex={reindexCorpus}
          onClose={() => setShowSettingsModal(false)}
        />
      )}
    </div>
  );
}
export default App;
