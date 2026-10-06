export interface DocumentItem {
  id: string;
  title: string;
  category: 'Courses' | 'Fees' | 'Clubs' | 'Timetables' | 'Faculty' | 'Rules';
  iconName: string;
  lastUpdated: string;
  content: string;
}

export interface RAGChunk {
  chunkId: string;
  docId: string;
  docTitle: string;
  category: string;
  section: string;
  text: string;
  tokenCount: number;
  tokens: string[];
}

export interface SearchResult {
  chunk: RAGChunk;
  score: number; // 0.0 to 1.0 (Cosine Similarity + Hybrid BM25)
  cosineScore: number;
  bm25Score: number;
  matchedTerms: string[];
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: string;
  sources?: SearchResult[];
  isThinking?: boolean;
}

export interface EvaluationTest {
  id: string;
  category: string;
  question: string;
  expectedKeywords: string[];
  expectedDocId: string;
  status?: 'passed' | 'failed' | 'running' | 'idle';
  actualAnswer?: string;
  retrievedDocId?: string;
  retrievalScore?: number;
  latencyMs?: number;
}

export interface RAGConfig {
  topK: number;
  chunkSize: number; // in words
  chunkOverlap: number; // in words
  similarityThreshold: number; // 0.0 to 1.0
  llmProvider: 'local' | 'openai' | 'gemini';
  apiKey?: string;
}
