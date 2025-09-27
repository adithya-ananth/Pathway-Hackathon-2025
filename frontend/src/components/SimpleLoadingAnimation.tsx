import { Brain, Sparkles } from "lucide-react";

export function SimpleLoadingAnimation() {
  return (
    <div className="flex justify-center items-center py-16">
      <div className="bg-card/60 backdrop-blur-xl rounded-2xl p-12 shadow-elegant border border-white/10 max-w-md w-full mx-4">
        {/* Animated Icons */}
        <div className="flex justify-center mb-8">
          <div className="relative">
            <div className="p-6 rounded-full bg-gradient-primary animate-loading-pulse shadow-glow">
              <Brain className="h-10 w-10 text-white" />
            </div>
            <div className="absolute -top-2 -right-2 p-2 rounded-full bg-accent/20 animate-bounce delay-300">
              <Sparkles className="h-4 w-4 text-accent" />
            </div>
            <div className="absolute inset-0 rounded-full bg-gradient-primary opacity-20 animate-ping scale-110"></div>
          </div>
        </div>

        {/* Loading Message */}
        <div className="text-center mb-8">
          <h3 className="text-xl font-semibold text-foreground mb-3 animate-fade-in-up">
            Analyzing Your Research Query
          </h3>
          <p className="text-muted-foreground animate-fade-in-up delay-150">
            Our AI is processing your request and searching through academic databases...
          </p>
        </div>

        {/* Elegant Progress Indicator */}
        <div className="relative">
          <div className="w-full h-1 bg-muted/50 rounded-full overflow-hidden">
            <div className="h-full bg-gradient-primary animate-loading-slide rounded-full"></div>
          </div>
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-loading-slide"></div>
        </div>

        {/* Floating Dots */}
        <div className="flex justify-center mt-6 space-x-2">
          <div className="w-2 h-2 bg-primary rounded-full animate-bounce"></div>
          <div className="w-2 h-2 bg-primary rounded-full animate-bounce delay-100"></div>
          <div className="w-2 h-2 bg-primary rounded-full animate-bounce delay-200"></div>
        </div>
      </div>
    </div>
  );
}