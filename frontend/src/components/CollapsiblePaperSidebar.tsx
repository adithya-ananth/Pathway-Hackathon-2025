import { FileText, ExternalLink } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

interface ArxivPaper {
  title: string;
  url: string;
  authors: string;
  abstract: string;
}

interface CollapsiblePaperSidebarProps {
  papers: ArxivPaper[];
  onPaperClick: (paper: ArxivPaper) => void;
}

export function CollapsiblePaperSidebar({ papers, onPaperClick }: CollapsiblePaperSidebarProps) {
  if (!papers.length) {
    return (
      <Card className="p-6 shadow-elegant bg-card/60 backdrop-blur-xl border border-white/10">
        <div className="text-center">
          <FileText className="h-12 w-12 text-muted-foreground mx-auto mb-3" />
          <p className="text-muted-foreground">No papers found for this query.</p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="shadow-elegant bg-card/60 backdrop-blur-xl border border-white/10 animate-fade-in-up">
      <div className="p-4 border-b border-white/10 bg-gradient-to-r from-muted/30 to-transparent">
        <div className="flex items-center space-x-2">
          <FileText className="h-5 w-5 text-research-primary" />
          <h3 className="font-bold text-foreground">Research Papers</h3>
          <span className="text-sm text-muted-foreground">({papers.length})</span>
        </div>
      </div>
      
      <ScrollArea className="h-[600px]">
        <div className="p-4 space-y-3">
          {papers.map((paper, index) => (
            <div
              key={index}
              className="group relative border border-white/10 rounded-lg p-4 bg-background/30 hover:bg-background/50 transition-all duration-300 cursor-pointer"
              onClick={() => onPaperClick(paper)}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                  <h4 className="font-semibold text-foreground text-sm leading-tight mb-2 group-hover:text-research-primary transition-colors">
                    {paper.title}
                  </h4>
                  <p className="text-xs text-muted-foreground mb-2 font-medium">
                    {paper.authors}
                  </p>
                  <p className="text-xs text-muted-foreground leading-relaxed line-clamp-2">
                    {paper.abstract.substring(0, 120)}...
                  </p>
                </div>
                
                <div className="flex flex-col gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-8 w-8 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={(e) => {
                      e.stopPropagation();
                      window.open(paper.url, '_blank');
                    }}
                  >
                    <ExternalLink className="h-3 w-3" />
                  </Button>
                </div>
              </div>
              
              <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-primary/5 to-accent/5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"></div>
            </div>
          ))}
        </div>
      </ScrollArea>
    </Card>
  );
}