import type { DocumentItem, RAGChunk, SearchResult, RAGConfig } from '../types/rag';

export class RAGEngine {
  private chunks: RAGChunk[] = [];
  private idfMap: Map<string, number> = new Map();
  private vectorSpace: Map<string, Map<string, number>> = new Map();
  private config: RAGConfig;

  constructor(config: RAGConfig) {
    this.config = config;
  }

  public updateConfig(newConfig: Partial<RAGConfig>) {
    this.config = { ...this.config, ...newConfig };
  }

  public tokenize(text: string): string[] {
    const cleanText = text.toLowerCase().replace(/[^a-z0-9\s]/g, ' ');
    const stopwords = new Set([
      'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'in', 'to', 'for',
      'of', 'with', 'by', 'as', 'it', 'be', 'are', 'this', 'that', 'from', 'in', 'all',
      'can', 'has', 'have', 'had', 'been', 'what', 'who', 'where', 'when', 'how', 'why'
    ]);
    return cleanText
      .split(/\s+/)
      .filter((w: string) => w.length > 1 && !stopwords.has(w));
  }

  public ingestDocuments(documents: DocumentItem[]): RAGChunk[] {
    const chunkSize = this.config.chunkSize;
    const chunkOverlap = this.config.chunkOverlap;

    this.chunks = [];
    
    documents.forEach(doc => {
      const sections = doc.content.split(/\n(?=#{1,3}\s+)/);
      let chunkIdx = 1;

      sections.forEach(secText => {
        const trimmed = secText.trim();
        if (!trimmed) return;

        const lines = trimmed.split('\n');
        let sectionHeader = doc.title;
        if (lines[0].startsWith('#')) {
          sectionHeader = lines[0].replace(/^#+\s*/, '').trim();
        }

        const words = trimmed.split(/\s+/);
        if (words.length === 0) return;

        let i = 0;
        while (i < words.length) {
          const chunkWords = words.slice(i, i + chunkSize);
          const chunkText = chunkWords.join(' ');
          const chunkId = `${doc.id.toUpperCase()}-${String(chunkIdx).padStart(2, '0')}`;
          
          const tokens = this.tokenize(chunkText);

          this.chunks.push({
            chunkId,
            docId: doc.id,
            docTitle: doc.title,
            category: doc.category,
            section: sectionHeader,
            text: chunkText,
            tokenCount: words.length,
            tokens
          });

          chunkIdx++;
          if (i + chunkSize >= words.length) break;
          i += (chunkSize - chunkOverlap > 0 ? chunkSize - chunkOverlap : chunkSize);
        }
      });
    });

    this.buildIndex();
    return this.chunks;
  }

  private buildIndex() {
    const docCount = this.chunks.length;
    const docFreqMap = new Map<string, number>();

    this.chunks.forEach(chunk => {
      const uniqueTokens = new Set(chunk.tokens);
      uniqueTokens.forEach(t => {
        docFreqMap.set(t, (docFreqMap.get(t) || 0) + 1);
      });
    });

    // Compute IDF
    this.idfMap.clear();
    docFreqMap.forEach((freq, term) => {
      const idf = Math.log((docCount + 1) / (freq + 1)) + 1;
      this.idfMap.set(term, idf);
    });

    // Build TF-IDF Normalized Vectors for each chunk
    this.vectorSpace.clear();
    this.chunks.forEach(chunk => {
      const tfMap = new Map<string, number>();
      chunk.tokens.forEach(t => {
        tfMap.set(t, (tfMap.get(t) || 0) + 1);
      });

      const vector = new Map<string, number>();
      let normSq = 0;

      tfMap.forEach((count, term) => {
        const idf = this.idfMap.get(term) || 1.0;
        const weight = (1 + Math.log(count)) * idf;
        vector.set(term, weight);
        normSq += weight * weight;
      });

      const norm = Math.sqrt(normSq) || 1.0;
      const normalizedVec = new Map<string, number>();
      vector.forEach((w, term) => {
        normalizedVec.set(term, w / norm);
      });

      this.vectorSpace.set(chunk.chunkId, normalizedVec);
    });
  }

  public search(query: string, customTopK?: number): SearchResult[] {
    const topK = customTopK || this.config.topK;
    const queryTokens = this.tokenize(query);
    if (queryTokens.length === 0) return [];

    // Query TF-IDF vector
    const qTfMap = new Map<string, number>();
    queryTokens.forEach(t => qTfMap.set(t, (qTfMap.get(t) || 0) + 1));

    const qVector = new Map<string, number>();
    let qNormSq = 0;

    qTfMap.forEach((count, term) => {
      const idf = this.idfMap.get(term) || 1.0;
      const weight = (1 + Math.log(count)) * idf;
      qVector.set(term, weight);
      qNormSq += weight * weight;
    });

    const qNorm = Math.sqrt(qNormSq) || 1.0;
    const normalizedQVec = new Map<string, number>();
    qVector.forEach((w, term) => {
      normalizedQVec.set(term, w / qNorm);
    });

    const results: SearchResult[] = [];

    this.chunks.forEach(chunk => {
      const chunkVec = this.vectorSpace.get(chunk.chunkId) || new Map();
      
      // 1. Cosine Similarity
      let cosineSim = 0;
      normalizedQVec.forEach((qWeight, term) => {
        if (chunkVec.has(term)) {
          cosineSim += qWeight * (chunkVec.get(term) || 0);
        }
      });

      // 2. BM25 / Exact Term Hits
      const matchedTerms: string[] = [];
      queryTokens.forEach(qt => {
        if (chunk.tokens.includes(qt) && !matchedTerms.includes(qt)) {
          matchedTerms.push(qt);
        }
      });

      const bm25Score = queryTokens.length > 0 ? (matchedTerms.length / queryTokens.length) : 0;

      // 3. Combined Hybrid Score
      const hybridScore = Math.min(1.0, (cosineSim * 0.7) + (bm25Score * 0.3));

      if (hybridScore >= this.config.similarityThreshold) {
        results.push({
          chunk,
          score: parseFloat(hybridScore.toFixed(4)),
          cosineScore: parseFloat(cosineSim.toFixed(4)),
          bm25Score: parseFloat(bm25Score.toFixed(4)),
          matchedTerms
        });
      }
    });

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  public getAllChunks(): RAGChunk[] {
    return this.chunks;
  }
}
