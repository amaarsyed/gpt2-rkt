import { Pause, Play, SkipBack, SkipForward } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

/**
 * The visualizer's transport: step back, play or pause the walk through this
 * section's stages, step forward, and where in the section the pulse is.
 */
export function PlayBar({
  playing,
  disabled,
  label,
  position,
  onToggle,
  onStep,
  speed,
  onSpeed,
}: {
  playing: boolean;
  disabled: boolean;
  label: string;
  position: string | null;
  onToggle: () => void;
  onStep: (dir: 1 | -1) => void;
  speed: number;
  onSpeed: () => void;
}) {
  return (
    <div className="sc-transport" role="group" aria-label="Playback">
      <Tooltip>
        <TooltipTrigger asChild>
          <Button variant="ghost" size="icon-sm" className="sc-transport-btn" aria-label="Previous stage" disabled={disabled} onClick={() => onStep(-1)}>
            <SkipBack />
          </Button>
        </TooltipTrigger>
        <TooltipContent>Previous stage</TooltipContent>
      </Tooltip>

      <Button
        variant="ghost"
        size="sm"
        className="sc-transport-main"
        aria-label={playing ? 'Pause' : 'Play'}
        aria-pressed={playing}
        disabled={disabled}
        onClick={onToggle}
      >
        {playing ? <Pause /> : <Play />}
      </Button>

      <Tooltip>
        <TooltipTrigger asChild>
          <Button variant="ghost" size="icon-sm" className="sc-transport-btn" aria-label="Next stage" disabled={disabled} onClick={() => onStep(1)}>
            <SkipForward />
          </Button>
        </TooltipTrigger>
        <TooltipContent>Next stage</TooltipContent>
      </Tooltip>

      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            className="sc-transport-btn sc-transport-speed"
            aria-label={`Playback speed ${speed}×, click to change`}
            onClick={onSpeed}
          >
            {speed}×
          </Button>
        </TooltipTrigger>
        <TooltipContent>Speed</TooltipContent>
      </Tooltip>

      <span className="sc-transport-now" aria-live="polite">
        <span className="sc-transport-label">{label}</span>
        {position ? <span className="sc-transport-pos">{position}</span> : null}
      </span>
    </div>
  );
}
