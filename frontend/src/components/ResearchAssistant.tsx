import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Search, FileText, Lightbulb } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";

interface ArxivPaper {
  title: string;
  url: string;
  authors: string;
  abstract: string;
}

interface ApiResponse {
  message: string;
  papers: ArxivPaper[];
}

export function ResearchAssistant() {
  const [query, setQuery] = useState("");
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleSearch = async () => {
    if (!query.trim()) {
      toast({
        title: "Please enter a research topic",
        description: "Enter a topic you'd like to explore to get started.",
        variant: "destructive",
      });
      return;
    }

    // Navigate to results page with query
    navigate(`/results?q=${encodeURIComponent(query)}`);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-secondary">
      {/* Header */}
      <header className="bg-card/80 backdrop-blur-xl border-b border-white/10 shadow-elegant sticky top-0 z-10">
        <div className="container mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-3 rounded-xl bg-gradient-primary shadow-glow">
                <Lightbulb className="h-7 w-7 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
                  AI Research Assistant
                </h1>
                <p className="text-muted-foreground font-medium">
                  Explore academic research with AI-powered insights
                </p>
              </div>
            </div>
            <div className="hidden md:flex items-center space-x-2 px-4 py-2 bg-muted/50 rounded-full">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-xs text-muted-foreground font-medium">Live</span>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        {/* Search Interface */}
        <Card className="p-10 mb-8 shadow-elegant bg-card/60 backdrop-blur-xl border border-white/10">
          <div className="max-w-3xl mx-auto text-center">
            <div className="mb-8">
              <div className="relative mb-6">
                <FileText className="h-16 w-16 text-research-primary mx-auto mb-4 drop-shadow-lg" />
                <div className="absolute top-0 left-1/2 transform -translate-x-1/2 w-20 h-20 bg-research-primary/20 rounded-full blur-xl"></div>
              </div>
              <h2 className="text-2xl font-bold text-foreground mb-3 leading-tight">
                What would you like to research today?
              </h2>
              <p className="text-muted-foreground text-lg leading-relaxed max-w-xl mx-auto">
                Enter any topic and get AI-powered insights with relevant academic papers from leading research databases
              </p>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-4 max-w-2xl mx-auto">
              <div className="relative flex-1">
                <Input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="e.g., machine learning in healthcare, quantum computing applications..."
                  className="h-14 text-lg pl-4 pr-4 bg-background/80 backdrop-blur border-2 border-muted hover:border-primary/30 focus:border-primary transition-all duration-300 shadow-inner"
                />
                <div className="absolute inset-0 rounded-md bg-gradient-to-r from-primary/5 to-accent/5 pointer-events-none"></div>
              </div>
              <Button
                onClick={handleSearch}
                disabled={!query.trim()}
                className="h-14 px-8 bg-gradient-primary hover:shadow-glow transition-all duration-300 font-semibold"
                size="lg"
              >
                <Search className="h-5 w-5 mr-2" />
                Research
              </Button>
            </div>
            
            {/* Suggestion Pills */}
            <div className="mt-6 flex flex-wrap justify-center gap-2">
              {["Neural Networks", "Quantum Computing", "Climate Science", "Bioengineering"].map((topic) => (
                <button
                  key={topic}
                  onClick={() => setQuery(topic)}
                  className="px-4 py-2 text-sm bg-muted/50 hover:bg-muted transition-colors rounded-full text-muted-foreground hover:text-foreground"
                >
                  {topic}
                </button>
              ))}
            </div>
          </div>
        </Card>

      </div>
    </div>
  );
}