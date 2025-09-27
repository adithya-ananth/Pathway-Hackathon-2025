import { useEffect, useState } from "react";
import { Brain, FileSearch, Database, Sparkles } from "lucide-react";

const loadingStages = [
  { message: "Analysing prompt", duration: 2000, icon: Brain },
  { message: "Loading papers from arXiv", duration: 8000, icon: FileSearch },
  { message: "Indexing", duration: 8000, icon: Database },
  { message: "Generating output", duration: 12000, icon: Sparkles },
];

export function LoadingAnimation() {
  const [currentStage, setCurrentStage] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    let timeoutId: NodeJS.Timeout;
    let intervalId: NodeJS.Timeout;
    let startTime = Date.now();

    const updateStage = () => {
      if (currentStage < loadingStages.length - 1) {
        setCurrentStage(prev => prev + 1);
        setProgress(0);
        startTime = Date.now();
      }
    };

    const updateProgress = () => {
      const elapsed = Date.now() - startTime;
      const stageDuration = loadingStages[currentStage]?.duration || 1000;
      const newProgress = Math.min((elapsed / stageDuration) * 100, 100);
      setProgress(newProgress);

      if (newProgress >= 100 && currentStage < loadingStages.length - 1) {
        updateStage();
      }
    };

    // Update progress every 100ms
    intervalId = setInterval(updateProgress, 100);

    // Set timeout for current stage
    if (currentStage < loadingStages.length - 1) {
      timeoutId = setTimeout(updateStage, loadingStages[currentStage]?.duration || 1000);
    }

    return () => {
      clearTimeout(timeoutId);
      clearInterval(intervalId);
    };
  }, [currentStage]);

  const currentStageData = loadingStages[currentStage];
  const CurrentIcon = currentStageData?.icon;

  return (
    <div className="flex justify-center items-center py-16">
      <div className="bg-card rounded-xl p-8 shadow-float border animate-research-glow max-w-md w-full mx-4">
        {/* Icon */}
        <div className="flex justify-center mb-6">
          <div className="relative">
            <div className="p-4 rounded-full bg-gradient-primary animate-loading-pulse">
              {CurrentIcon && <CurrentIcon className="h-8 w-8 text-white" />}
            </div>
            <div className="absolute inset-0 rounded-full bg-gradient-primary opacity-30 animate-ping"></div>
          </div>
        </div>

        {/* Current Stage Message */}
        <div className="text-center mb-6">
          <h3 className="text-lg font-semibold text-foreground mb-2 animate-fade-in-up">
            {currentStageData?.message}
          </h3>
          <div className="text-sm text-muted-foreground">
            Stage {currentStage + 1} of {loadingStages.length}
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mb-6">
          <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
            <div
              className="h-full bg-gradient-primary transition-all duration-300 ease-out relative"
              style={{ width: `${progress}%` }}
            >
              <div className="absolute inset-0 bg-gradient-loading animate-loading-slide"></div>
            </div>
          </div>
          <div className="flex justify-between text-xs text-muted-foreground mt-2">
            <span>{Math.round(progress)}%</span>
            <span>
              {(((currentStage * 100) + progress) / loadingStages.length).toFixed(0)}% Complete
            </span>
          </div>
        </div>

        {/* Stage Indicators */}
        <div className="flex justify-center space-x-2">
          {loadingStages.map((stage, index) => {
            const StageIcon = stage.icon;
            return (
              <div
                key={index}
                className={`p-2 rounded-full transition-all duration-300 ${
                  index < currentStage
                    ? "bg-research-success text-white"
                    : index === currentStage
                    ? "bg-research-primary text-white animate-loading-pulse"
                    : "bg-muted text-muted-foreground"
                }`}
              >
                <StageIcon className="h-4 w-4" />
              </div>
            );
          })}
        </div>

        {/* Processing Text */}
        <div className="text-center mt-4">
          <p className="text-xs text-muted-foreground">
            Processing your research query...
          </p>
        </div>
      </div>
    </div>
  );
}