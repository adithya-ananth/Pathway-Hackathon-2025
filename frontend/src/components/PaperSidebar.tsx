import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileText, ExternalLink, Users } from "lucide-react";

interface ArxivPaper {
  title: string;
  url: string;
  authors: string;
  abstract: string;
}

interface PaperSidebarProps {
  papers: ArxivPaper[];
  selectedPaper: ArxivPaper | null;
  onPaperSelect: (paper: ArxivPaper) => void;
}

export function PaperSidebar({ papers, selectedPaper, onPaperSelect }: PaperSidebarProps) {
  if (!papers || papers.length === 0) {
    return (
      <Card className="p-6 shadow-card">
        <div className="text-center text-muted-foreground">
          <FileText className="h-8 w-8 mx-auto mb-3 opacity-50" />
          <p className="text-sm">No papers found yet</p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="shadow-card animate-fade-in-up">
      <div className="p-4 border-b bg-muted/50">
        <div className="flex items-center space-x-2">
          <FileText className="h-5 w-5 text-research-primary" />
          <h3 className="font-semibold text-foreground">Related Papers</h3>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          {papers.length} papers found
        </p>
      </div>

      <ScrollArea className="h-[600px]">
        <div className="p-2 space-y-2">
          {papers.map((paper, index) => (
            <div key={index} className="relative">
              <Button
                variant={selectedPaper?.url === paper.url ? "default" : "ghost"}
                className={`
                  w-full h-auto p-4 text-left justify-start transition-all duration-200
                  ${selectedPaper?.url === paper.url 
                    ? "bg-gradient-primary text-white shadow-research" 
                    : "hover:bg-muted hover:shadow-card"
                  }
                `}
                onClick={() => onPaperSelect(paper)}
              >
                <div className="w-full">
                  {/* Paper Title */}
                  <h4 className={`
                    font-medium text-sm leading-tight mb-2 line-clamp-3
                    ${selectedPaper?.url === paper.url ? "text-white" : "text-foreground"}
                  `}>
                    {paper.title}
                  </h4>

                  {/* Authors */}
                  <div className="flex items-center space-x-1 mb-2">
                    <Users className="h-3 w-3 opacity-60" />
                    <p className={`
                      text-xs line-clamp-1
                      ${selectedPaper?.url === paper.url ? "text-white/80" : "text-muted-foreground"}
                    `}>
                      {paper.authors}
                    </p>
                  </div>

                  {/* Abstract Preview */}
                  <p className={`
                    text-xs leading-relaxed line-clamp-2
                    ${selectedPaper?.url === paper.url ? "text-white/70" : "text-muted-foreground"}
                  `}>
                    {paper.abstract}
                  </p>

                  {/* Paper Number Badge */}
                  <div className={`
                    absolute top-2 right-2 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold
                    ${selectedPaper?.url === paper.url 
                      ? "bg-white/20 text-white" 
                      : "bg-research-primary text-white"
                    }
                  `}>
                    {index + 1}
                  </div>
                </div>
              </Button>

              {/* External Link Button */}
              <Button
                variant="ghost"
                size="sm"
                className={`
                  absolute bottom-2 right-2 h-6 w-6 p-0 opacity-60 hover:opacity-100
                  ${selectedPaper?.url === paper.url ? "text-white hover:bg-white/20" : ""}
                `}
                onClick={(e) => {
                  e.stopPropagation();
                  window.open(paper.url, '_blank');
                }}
              >
                <ExternalLink className="h-3 w-3" />
              </Button>
            </div>
          ))}
        </div>
      </ScrollArea>

      {/* Footer */}
      <div className="p-3 border-t bg-muted/30">
        <p className="text-xs text-muted-foreground text-center">
          Click a paper to view • {papers.length} papers loaded
        </p>
      </div>
    </Card>
  );
}