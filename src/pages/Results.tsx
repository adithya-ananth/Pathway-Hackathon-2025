import { useState, useEffect } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { Search, Lightbulb, ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { SimpleLoadingAnimation } from "@/components/SimpleLoadingAnimation";
import { CollapsiblePaperSidebar } from "@/components/CollapsiblePaperSidebar";
import { PaperViewModal } from "@/components/PaperViewModal";
import { useToast } from "@/hooks/use-toast";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface ArxivPaperValue {
  id: string;
  title: string;
  abstract: string;
  authors: string[];
  similarity_score: number;
  url: string;
  primary_category: string;
  file_path: string;
  matched_keywords: string[];
}

interface ArxivPaper {
  _value: ArxivPaperValue;
}

interface ArxivPaperFlat {
  title: string;
  url: string;
  authors: string;
  abstract: string;
}

interface ApiResponse {
  message: string;
  papers: ArxivPaper[];
  flatPapers?: ArxivPaperFlat[];
}

export default function Results() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState<ApiResponse | null>(null);
  const [selectedPaper, setSelectedPaper] = useState<ArxivPaperFlat | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const { toast } = useToast();

  const initialQuery = searchParams.get('q');

  useEffect(() => {
    if (initialQuery) {
      setQuery(initialQuery);
      performSearch(initialQuery);
    }
  }, [initialQuery]);

  const performSearch = async (searchQuery: string) => {
    if (!searchQuery.trim()) return;

    setIsLoading(true);
    setResponse(null);

    // Admin bypass
    if (searchQuery.toLowerCase().trim() === "admin") {
      setTimeout(() => {
        const mockData: ApiResponse = {
          message: "## Answer to: rag\n\n### Executive Summary\nBased on analysis of 5 relevant documents related to your query about rag, with focus on: Retrieval-Augmented Generation, RAG, Information Retrieval, Natural Language Processing.\n\n### Key Insights\nThis is an interdisciplinary topic spanning \"cs.CL\", \"cs.IR\" domains.\n\nThe most relevant aspects identified include: \"information retrieval\", \"natural language processing\", \"rag\", \"retrieval-augmented generation\".\n\nCommon themes across the research include: retrieval, augmented, generation, language, llms.\n\nNotable researchers in this area include: \"Anas Neumann\", \"Shailja Gupta\", \"Rajesh Ranjan\", \"Gautam B\", \"Anupam Purwar\".\n\n### Supporting Documents\n\nDocument 1: \"A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems:   Progress, Gaps, and Future Directions\"\nRelevance Score: 0.299\nAbstract: \"Retrieval-Augmented Generation (RAG) represents a major advancement in natural language processing (NLP), combining large language models (LLMs) with information retrieval systems to enhance factual grounding, accuracy, and contextual relevance. This paper presents a comprehensive systematic rev...\nMatched Terms: \"retrieval-augmented generation\", \"rag\", \"information retrieval\", \"natural language processing\"\n\nDocument 2: \"A Comprehensive Survey of Retrieval-Augmented Generation (RAG):   Evolution, Current Landscape and Future Directions\"\nRelevance Score: 0.284\nAbstract: \"This paper presents a comprehensive study of Retrieval-Augmented Generation (RAG), tracing its evolution from foundational concepts to the current state of the art. RAG combines retrieval mechanisms with generative language models to enhance the accuracy of outputs, addressing key limitations of...\nMatched Terms: \"retrieval-augmented generation\", \"rag\", \"natural language processing\"\n\nDocument 3: \"RAG-Fusion: a New Take on Retrieval-Augmented Generation\"\nRelevance Score: 0.269\nAbstract: \"Infineon has identified a need for engineers, account managers, and customers to rapidly obtain product information. This problem is traditionally addressed with retrieval-augmented generation (RAG) chatbots, but in this study, I evaluated the use of the newly popularized RAG-Fusion method. RAG-...\nMatched Terms: \"retrieval-augmented generation\", \"rag\", \"natural language processing\"\n\nDocument 4: \"An Agile Method for Implementing Retrieval Augmented Generation Tools in   Industrial SMEs\"\nRelevance Score: 0.255\nAbstract: \"Retrieval-Augmented Generation (RAG) has emerged as a powerful solution to mitigate the limitations of Large Language Models (LLMs), such as hallucinations and outdated knowledge. However, deploying RAG-based tools in Small and Medium Enterprises (SMEs) remains a challenge due to their limited r...\nMatched Terms: \"retrieval-augmented generation\", \"rag\", \"natural language processing\"\n\nDocument 5: \"Evaluating the Efficacy of Open-Source LLMs in Enterprise-Specific RAG   Systems: A Comparative Study of Performance and Scalability\"\nRelevance Score: 0.245\nAbstract: \"This paper presents an analysis of open-source large language models (LLMs) and their application in Retrieval-Augmented Generation (RAG) tasks, specific for enterprise-specific data sets scraped from their websites. With the increasing reliance on LLMs in natural language processing, it is cruc...\nMatched Terms: \"retrieval-augmented generation\", \"rag\", \"natural language processing\"\n\n### Conclusion\nThe 5 retrieved documents provide comprehensive coverage of 'rag', spanning multiple research perspectives and methodological approaches. This research spans \"cs.CL\", \"cs.IR\" domains, indicating the interdisciplinary nature of the topic. The focus on Retrieval-Augmented Generation, RAG, Information Retrieval, Natural Language Processing appears well-supported by the current literature, with documents directly addressing these concepts. Key research includes work on \"\"A Systematic Review of Key Retrieval-Augmented Generation (...\", demonstrating active development in this area. \n\n### For Further Research\nConsider exploring the full text of the most relevant documents above, \nparticularly those with the highest relevance scores. \nYou may also want to search for related terms such as: transformers, \"retrieval-augmented generation\", BERT, GPT, language models.",
          papers: [
            {
              _value: {
                id: "2507.18910v1",
                title: "A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems:   Progress, Gaps, and Future Directions",
                abstract: "Retrieval-Augmented Generation (RAG) represents a major advancement in natural language processing (NLP), combining large language models (LLMs) with information retrieval systems to enhance factual grounding, accuracy, and contextual relevance. This paper presents a comprehensive systematic review of RAG, tracing its evolution from early developments in open domain question answering to recent state-of-the-art implementations across diverse applications. The review begins by outlining the motivations behind RAG, particularly its ability to mitigate hallucinations and outdated knowledge in parametric models. Core technical components-retrieval mechanisms, sequence-to-sequence generation models, and fusion strategies are examined in detail. A year-by-year analysis highlights key milestones and research trends, providing insight into RAG's rapid growth. The paper further explores the deployment of RAG in enterprise systems, addressing practical challenges related to retrieval of proprietary data, security, and scalability. A comparative evaluation of RAG implementations is conducted, benchmarking performance on retrieval accuracy, generation fluency, latency, and computational efficiency. Persistent challenges such as retrieval quality, privacy concerns, and integration overhead are critically assessed. Finally, the review highlights emerging solutions, including hybrid retrieval approaches, privacy-preserving techniques, optimized fusion strategies, and agentic RAG architectures. These innovations point toward a future of more reliable, efficient, and context-aware knowledge-intensive NLP systems.",
                authors: [
                  "Agada Joseph Oche",
                  "Ademola Glory Folashade",
                  "Tirthankar Ghosal",
                  "Arpan Biswas"
                ],
                similarity_score: 0.29929313949169933,
                url: "http://arxiv.org/abs/2507.18910v1",
                primary_category: "cs.CL",
                file_path: "papers_text/2507.18910v1.txt",
                matched_keywords: [
                  "retrieval-augmented generation",
                  "rag",
                  "information retrieval",
                  "natural language processing"
                ]
              }
            },
            {
              _value: {
                id: "2410.12837v1",
                title: "A Comprehensive Survey of Retrieval-Augmented Generation (RAG):   Evolution, Current Landscape and Future Directions",
                abstract: "This paper presents a comprehensive study of Retrieval-Augmented Generation (RAG), tracing its evolution from foundational concepts to the current state of the art. RAG combines retrieval mechanisms with generative language models to enhance the accuracy of outputs, addressing key limitations of LLMs. The study explores the basic architecture of RAG, focusing on how retrieval and generation are integrated to handle knowledge-intensive tasks. A detailed review of the significant technological advancements in RAG is provided, including key innovations in retrieval-augmented language models and applications across various domains such as question-answering, summarization, and knowledge-based tasks. Recent research breakthroughs are discussed, highlighting novel methods for improving retrieval efficiency. Furthermore, the paper examines ongoing challenges such as scalability, bias, and ethical concerns in deployment. Future research directions are proposed, focusing on improving the robustness of RAG models, expanding the scope of application of RAG models, and addressing societal implications. This survey aims to serve as a foundational resource for researchers and practitioners in understanding the potential of RAG and its trajectory in natural language processing.",
                authors: [
                  "Shailja Gupta",
                  "Rajesh Ranjan",
                  "Surya Narayan Singh"
                ],
                similarity_score: 0.28409249784249946,
                url: "http://arxiv.org/abs/2410.12837v1",
                primary_category: "cs.CL",
                file_path: "papers_text/2410.12837v1.txt",
                matched_keywords: [
                  "retrieval-augmented generation",
                  "rag",
                  "natural language processing"
                ]
              }
            },
            {
              _value: {
                id: "2402.03367v2",
                title: "RAG-Fusion: a New Take on Retrieval-Augmented Generation",
                abstract: "Infineon has identified a need for engineers, account managers, and customers to rapidly obtain product information. This problem is traditionally addressed with retrieval-augmented generation (RAG) chatbots, but in this study, I evaluated the use of the newly popularized RAG-Fusion method. RAG-Fusion combines RAG and reciprocal rank fusion (RRF) by generating multiple queries, reranking them with reciprocal scores and fusing the documents and scores. Through manually evaluating answers on accuracy, relevance, and comprehensiveness, I found that RAG-Fusion was able to provide accurate and comprehensive answers due to the generated queries contextualizing the original query from various perspectives. However, some answers strayed off topic when the generated queries' relevance to the original query is insufficient. This research marks significant progress in artificial intelligence (AI) and natural language processing (NLP) applications and demonstrates transformations in a global and multi-industry context.",
                authors: [
                  "Zackary Rackauckas"
                ],
                similarity_score: 0.26888617959879,
                url: "http://arxiv.org/abs/2402.03367v2",
                primary_category: "cs.IR",
                file_path: "papers_text/2402.03367v2.txt",
                matched_keywords: [
                  "retrieval-augmented generation",
                  "rag",
                  "natural language processing"
                ]
              }
            },
            {
              _value: {
                id: "2508.21024v1",
                title: "An Agile Method for Implementing Retrieval Augmented Generation Tools in   Industrial SMEs",
                abstract: "Retrieval-Augmented Generation (RAG) has emerged as a powerful solution to mitigate the limitations of Large Language Models (LLMs), such as hallucinations and outdated knowledge. However, deploying RAG-based tools in Small and Medium Enterprises (SMEs) remains a challenge due to their limited resources and lack of expertise in natural language processing (NLP). This paper introduces EASI-RAG, Enterprise Application Support for Industrial RAG, a structured, agile method designed to facilitate the deployment of RAG systems in industrial SME contexts. EASI-RAG is based on method engineering principles and comprises well-defined roles, activities, and techniques. The method was validated through a real-world case study in an environmental testing laboratory, where a RAG tool was implemented to answer operators queries using data extracted from operational procedures. The system was deployed in under a month by a team with no prior RAG experience and was later iteratively improved based on user feedback. Results demonstrate that EASI-RAG supports fast implementation, high user adoption, delivers accurate answers, and enhances the reliability of underlying data. This work highlights the potential of RAG deployment in industrial SMEs. Future works include the need for generalization across diverse use cases and further integration with fine-tuned models.",
                authors: [
                  "Mathieu Bourdin",
                  "Anas Neumann",
                  "Thomas Paviot",
                  "Robert Pellerin",
                  "Samir Lamouri"
                ],
                similarity_score: 0.25531741871208413,
                url: "http://arxiv.org/abs/2508.21024v1",
                primary_category: "cs.CL",
                file_path: "papers_text/2508.21024v1.txt",
                matched_keywords: [
                  "retrieval-augmented generation",
                  "rag",
                  "natural language processing"
                ]
              }
            },
            {
              _value: {
                id: "2406.11424v1",
                title: "Evaluating the Efficacy of Open-Source LLMs in Enterprise-Specific RAG   Systems: A Comparative Study of Performance and Scalability",
                abstract: "This paper presents an analysis of open-source large language models (LLMs) and their application in Retrieval-Augmented Generation (RAG) tasks, specific for enterprise-specific data sets scraped from their websites. With the increasing reliance on LLMs in natural language processing, it is crucial to evaluate their performance, accessibility, and integration within specific organizational contexts. This study examines various open-source LLMs, explores their integration into RAG frameworks using enterprise-specific data, and assesses the performance of different open-source embeddings in enhancing the retrieval and generation process. Our findings indicate that open-source LLMs, combined with effective embedding techniques, can significantly improve the accuracy and efficiency of RAG systems, offering a viable alternative to proprietary solutions for enterprises.",
                authors: [
                  "Gautam B",
                  "Anupam Purwar"
                ],
                similarity_score: 0.24538043628024808,
                url: "http://arxiv.org/abs/2406.11424v1",
                primary_category: "cs.IR",
                file_path: "papers_text/2406.11424v1.txt",
                matched_keywords: [
                  "retrieval-augmented generation",
                  "rag",
                  "natural language processing"
                ]
              }
            }
          ]
        };
        
        // Transform mock data too
        const transformedMockData = {
          ...mockData,
          flatPapers: mockData.papers?.map(paper => ({
            title: paper._value.title,
            url: paper._value.url.replace('/abs/', '/pdf/'),
            authors: paper._value.authors.join(', '),
            abstract: paper._value.abstract
          })) || []
        };
        
        setResponse(transformedMockData);
        setIsLoading(false);
        toast({
          title: "Admin Demo Complete!",
          description: `Loaded ${mockData.papers.length} sample papers for demonstration.`,
        });
      }, 1500);
      return;
    }

    try {
      const res = await fetch("http://localhost:8080/prompt", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt: searchQuery }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data: ApiResponse = await res.json();
      
      // Transform papers to flat structure for easier use
      const transformedData = {
        ...data,
        flatPapers: data.papers?.map(paper => ({
          title: paper._value.title,
          url: paper._value.url.replace('/abs/', '/pdf/'),
          authors: paper._value.authors.join(', '),
          abstract: paper._value.abstract
        })) || []
      };
      
      setResponse(transformedData);
      
      toast({
        title: "Research Complete!",
        description: `Found ${data.papers?.length || 0} relevant papers for your query.`,
      });
    } catch (error) {
      console.error("Error fetching research:", error);
      toast({
        title: "Research Failed",
        description: "Unable to connect to the research API. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!query.trim()) {
      toast({
        title: "Please enter a research topic",
        description: "Enter a topic you'd like to explore to get started.",
        variant: "destructive",
      });
      return;
    }

    // Update URL and perform search
    navigate(`/results?q=${encodeURIComponent(query)}`);
    await performSearch(query);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !isLoading) {
      handleSearch();
    }
  };

  const handlePaperClick = (paper: ArxivPaperFlat) => {
    setSelectedPaper(paper);
    setIsModalOpen(true);
  };

  return (
    <div className="min-h-screen bg-gradient-secondary">
      {/* Header with Search */}
      <header className="bg-card/80 backdrop-blur-xl border-b border-white/10 shadow-elegant sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Button
                variant="ghost"
                onClick={() => navigate('/')}
                className="p-2 hover:bg-muted/50"
              >
                <ChevronLeft className="h-5 w-5" />
              </Button>
              <div className="flex items-center space-x-3">
                <div className="p-2 rounded-lg bg-gradient-primary shadow-glow">
                  <Lightbulb className="h-5 w-5 text-white" />
                </div>
                <h1 className="text-xl font-bold text-foreground">AI Research Assistant</h1>
              </div>
            </div>
            
            <div className="flex items-center gap-4 flex-1 max-w-2xl mx-8">
              <div className="relative flex-1">
                <Input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Search research topics..."
                  className="h-10 pl-4 pr-4 bg-background/80 backdrop-blur border border-muted hover:border-primary/30 focus:border-primary transition-all duration-300"
                  disabled={isLoading}
                />
              </div>
              <Button
                onClick={handleSearch}
                disabled={isLoading || !query.trim()}
                className="h-10 px-6 bg-gradient-primary hover:shadow-glow transition-all duration-300"
              >
                <Search className="h-4 w-4 mr-2" />
                Search
              </Button>
            </div>

            <Button
              variant="ghost"
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="p-2 hover:bg-muted/50"
            >
              <ChevronRight className={`h-5 w-5 transition-transform duration-300 ${sidebarCollapsed ? 'rotate-180' : ''}`} />
            </Button>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        {/* Loading Animation */}
        {isLoading && <SimpleLoadingAnimation />}

        {/* Results Layout */}
        {response && !isLoading && (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Main Content */}
            <div className={`transition-all duration-300 ${sidebarCollapsed ? 'lg:col-span-4' : 'lg:col-span-3'}`}>
              <Card className="p-8 shadow-elegant bg-card/60 backdrop-blur-xl border border-white/10 animate-fade-in-up">
                <div className="mb-6">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="p-2 rounded-lg bg-gradient-primary/10">
                      <Lightbulb className="h-5 w-5 text-research-primary" />
                    </div>
                    <h3 className="text-xl font-bold text-foreground">Research Summary</h3>
                  </div>
                  <div className="h-1 w-20 bg-gradient-primary rounded-full shadow-glow"></div>
                </div>
                <div className="prose prose-slate max-w-none dark:prose-invert">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {response.message}
                  </ReactMarkdown>
                </div>
              </Card>
            </div>

            {/* Papers Sidebar */}
            {!sidebarCollapsed && (
              <div className="lg:col-span-1">
                <CollapsiblePaperSidebar
                  papers={response.flatPapers || []}
                  onPaperClick={handlePaperClick}
                />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Paper View Modal */}
      <PaperViewModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        selectedPaper={selectedPaper}
        allPapers={response?.flatPapers || []}
        onPaperSelect={setSelectedPaper}
        llmResponse={response?.message || ""}
      />
    </div>
  );
}
