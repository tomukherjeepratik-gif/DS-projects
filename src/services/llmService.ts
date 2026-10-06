import type { SearchResult, ChatMessage } from '../types/rag';

export class LLMService {
  public async generateRAGResponse(
    query: string,
    searchResults: SearchResult[],
    conversationHistory: ChatMessage[],
    llmProvider: 'local' | 'openai' | 'gemini' = 'local',
    apiKey?: string
  ): Promise<{ text: string; sources: SearchResult[] }> {
    if (searchResults.length === 0) {
      return {
        text: `I apologize, but I could not find any relevant information in the official **Nexus Institute of Technology** documents to answer your query.\n\n*Tip: Try rephrasing your question or selecting one of the suggested sample questions below.*`,
        sources: []
      };
    }

    if (llmProvider === 'openai' && apiKey) {
      try {
        return await this.callOpenAI(query, searchResults, conversationHistory, apiKey);
      } catch (err: any) {
        console.warn('OpenAI call failed, falling back to local grounded generator:', err);
      }
    }

    // Default: Grounded Local Synthesizer
    const synthesizedText = this.synthesizeLocalAnswer(query, searchResults, conversationHistory);
    return {
      text: synthesizedText,
      sources: searchResults
    };
  }

  private synthesizeLocalAnswer(
    query: string,
    searchResults: SearchResult[],
    history: ChatMessage[]
  ): string {
    const qLower = query.toLowerCase();
    const citedChunks = searchResults.map(s => `[${s.chunk.docTitle} | ${s.chunk.chunkId}]`).join(', ');

    let bulletPoints: string[] = [];

    searchResults.forEach((res, idx) => {
      const chunk = res.chunk;
      const citation = `[Source: ${chunk.docTitle} - ${chunk.section} | ID: ${chunk.chunkId}]`;
      const cleanSnippet = chunk.text
        .replace(/^#+\s*/gm, '')
        .replace(/\n+/g, ' ')
        .trim();

      bulletPoints.push(`**${idx + 1}. ${chunk.section}** (${citation}):\n"${cleanSnippet}"`);
    });

    let intro = `Based on the official **Nexus Institute of Technology (NIT)** documentation:`;
    if (history.length > 2) {
      intro = `Continuing our conversation regarding **Nexus Institute of Technology**:`;
    }

    let summaryText = '';
    if (qLower.includes('fee') || qLower.includes('tuition') || qLower.includes('scholarship') || qLower.includes('cost')) {
      summaryText = `\n\n📌 **Key Financial Summary**: Tuition for B.Tech programs is **$4,500 USD (₹1,25,000 INR)** per semester. Top performers (CGPA ≥ 9.50) can qualify for the **100% Academic Excellence Scholarship**, while female students in STEM can apply for a **30% tuition waiver**.`;
    } else if (qLower.includes('attendance') || qLower.includes('rule') || qLower.includes('curfew') || qLower.includes('policy')) {
      summaryText = `\n\n📌 **Policy Highlight**: A strict **75% minimum attendance** is mandatory to sit for exams. Hostel curfew is **10:30 PM**, and zero tolerance is enforced for plagiarism and ragging.`;
    } else if (qLower.includes('acm') || qLower.includes('club') || qLower.includes('hackathon')) {
      summaryText = `\n\n📌 **Club Highlight**: The **ACM Student Chapter** meets every **Wednesday at 5:00 PM** in Lab 304 (Building A), hosted by Faculty Sponsor Prof. Aris Thorne. They organize the annual **NexusHacks 36-Hour Hackathon**.`;
    } else if (qLower.includes('library') || qLower.includes('timetable') || qLower.includes('schedule') || qLower.includes('hour')) {
      summaryText = `\n\n📌 **Schedule Highlight**: Regular library hours are **08:00 AM – 11:00 PM** on weekdays, and **24/7 continuous access** during exam weeks!`;
    } else if (qLower.includes('faculty') || qLower.includes('professor') || qLower.includes('vance') || qLower.includes('hod')) {
      summaryText = `\n\n📌 **Faculty Contact Highlight**: Computer Science HoD is **Dr. Elena Vance** (Building A - Room 402, evance@nexus.edu). Office hours are Mon & Wed 2:00 – 4:00 PM.`;
    }

    return `${intro}\n\n${bulletPoints.join('\n\n')}${summaryText}\n\n---
🔍 **Retrieved Sources Cited**: ${citedChunks} (Average Similarity Match: **${(
      searchResults.reduce((acc, curr) => acc + curr.score, 0) / searchResults.length * 100
    ).toFixed(1)}%**)`;
  }

  private async callOpenAI(
    query: string,
    searchResults: SearchResult[],
    history: ChatMessage[],
    apiKey: string
  ): Promise<{ text: string; sources: SearchResult[] }> {
    const context = searchResults.map(s => 
      `--- Document Chunk ID: ${s.chunk.chunkId} | Source: ${s.chunk.docTitle} ---\n${s.chunk.text}`
    ).join('\n\n');

    const messages = [
      {
        role: 'system',
        content: `You are NexusAI, an assistant for Nexus Institute of Technology. Answer the query using ONLY the provided document chunks. Always cite the exact Chunk ID (e.g. [FEES-01]) when referencing facts.`
      },
      ...history.slice(-4).map(h => ({
        role: h.sender === 'user' ? 'user' : 'assistant',
        content: h.text
      })),
      {
        role: 'user',
        content: `Context:\n${context}\n\nQuestion: ${query}`
      }
    ];

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: 'gpt-3.5-turbo',
        messages,
        temperature: 0.2
      })
    });

    const data = await response.json();
    return {
      text: data.choices[0].message.content,
      sources: searchResults
    };
  }
}
