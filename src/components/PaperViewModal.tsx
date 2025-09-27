import { useState } from "react";
import { X, FileText, Lightbulb, ExternalLink, ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Dialog, DialogContent } from "@/components/ui/dialog";

interface ArxivPaper {
  title: string;
  url: string;
  authors: string;
  abstract: string;
}

interface PaperViewModalProps {
  isOpen: boolean;
  onClose: () => void;
  selectedPaper: ArxivPaper | null;
  allPapers: ArxivPaper[];
  onPaperSelect: (paper: ArxivPaper) => void;
  llmResponse: string;
}

export function PaperViewModal({ 
  isOpen, 
  onClose, 
  selectedPaper, 
  allPapers, 
  onPaperSelect, 
  llmResponse 
}: PaperViewModalProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  if (!selectedPaper) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent 
        className="max-w-[98vw] h-[98vh] p-0 bg-background/95 backdrop-blur-xl border border-white/20 [&>button]:hidden flex flex-col"
      >
        {/* Compact Header with Title and Navigation Tabs */}
        <div className="border-b border-white/10 bg-card/60 flex-shrink-0">
          {/* Title Bar */}
          <div className="flex items-center justify-between px-4 py-2">
            <div className="flex items-center space-x-3">
              <div className="p-1.5 rounded-lg bg-gradient-primary/10">
                <FileText className="h-4 w-4 text-research-primary" />
              </div>
              <div>
                <h2 className="text-sm font-bold text-foreground">Paper Viewer</h2>
              </div>
            </div>
            
            <Button
              variant="ghost"
              onClick={onClose}
              className="h-7 w-7 p-0 hover:bg-muted/50"
            >
              <X className="h-3.5 w-3.5" />
            </Button>
          </div>

          {/* Paper Navigation Tabs - Centered */}
          <div className="bg-muted/20">
            <ScrollArea className="w-full">
              <div className="flex justify-center px-2 py-1 gap-2 min-w-max">
                {allPapers.map((paper, index) => (
                  <Button
                    key={index}
                    variant={selectedPaper === paper ? "default" : "ghost"}
                    size="sm"
                    onClick={() => onPaperSelect(paper)}
                    className={`min-w-[180px] justify-start text-left ${
                      selectedPaper === paper 
                        ? "bg-gradient-primary text-white shadow-glow" 
                        : "hover:bg-research-primary/10 hover:text-research-primary hover:border-research-primary/20 transition-all duration-200"
                    }`}
                  >
                    <div className="truncate">
                      <div className="font-medium text-xs truncate">
                        {paper.title.substring(0, 35)}...
                      </div>
                      <div className="text-xs opacity-70 truncate">
                        {paper.authors.split(',')[0]}
                      </div>
                    </div>
                  </Button>
                ))}
              </div>
            </ScrollArea>
          </div>
        </div>

        {/* Split View Content with Collapsible Sidebar */}
        <div className="flex flex-1 min-h-0 overflow-hidden relative">
          {/* Collapsible LLM Response - Left Side */}
          {sidebarOpen && (
            <div className="w-2/5 border-r border-white/10 flex flex-col bg-card/30">
              <div className="p-3 border-b border-white/10 bg-gradient-to-r from-muted/30 to-transparent flex-shrink-0">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Lightbulb className="h-4 w-4 text-research-primary" />
                    <h3 className="font-bold text-foreground text-sm">AI Analysis</h3>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSidebarOpen(false)}
                    className="h-6 w-6 p-0"
                  >
                    <ChevronLeft className="h-3 w-3" />
                  </Button>
                </div>
              </div>
              
              <ScrollArea className="flex-1">
                <div className="p-4">
                  <div className="prose prose-slate max-w-none prose-sm">
                    <p className="text-foreground leading-relaxed whitespace-pre-wrap text-sm">
                      {llmResponse}
                    </p>
                  </div>
                </div>
              </ScrollArea>
            </div>
          )}

          {/* Paper Viewer - Right Side */}
          <div className={`${sidebarOpen ? 'w-3/5' : 'w-full'} flex flex-col bg-background/30 transition-all duration-300`}>
            <div className="p-3 border-b border-white/10 bg-gradient-to-r from-muted/30 to-transparent flex-shrink-0">
              <div className="flex items-center justify-between">
                {!sidebarOpen && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSidebarOpen(true)}
                    className="h-7 gap-1 mr-3"
                  >
                    <ChevronRight className="h-3 w-3" />
                    <Lightbulb className="h-3 w-3" />
                    <span className="text-xs">AI Analysis</span>
                  </Button>
                )}
                <div className="flex-1 min-w-0 pr-3">
                  <h4 className="font-bold text-foreground text-sm leading-tight mb-1 truncate">
                    {selectedPaper.title}
                  </h4>
                  <p className="text-xs text-muted-foreground truncate">
                    {selectedPaper.authors}
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => window.open(selectedPaper.url, '_blank')}
                  className="h-7 gap-1 flex-shrink-0"
                >
                  <ExternalLink className="h-3 w-3" />
                  Open
                </Button>
              </div>
            </div>
            
            <div className="flex-1 bg-background/50 min-h-0 p-3">
              <div className="w-full h-full bg-white rounded-lg overflow-hidden shadow-inner">
                <iframe
                  src={selectedPaper.url}
                  className="w-full h-full border-0 rounded-lg"
                  title={selectedPaper.title}
                  style={{ height: '100%', width: '100%' }}
                />
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}