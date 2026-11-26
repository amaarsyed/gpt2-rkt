import { lazy, Suspense, useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { Network, X } from 'lucide-react';
import { TickRuler } from '@/components/ui/tick-ruler';
import { TooltipProvider } from '@/components/ui/tooltip';
import { stillness } from '../figures/shared';
import type { VizApi } from '../visualizer/ExecutionVisualizer';
import { advance as advanceStage, LAST_STEP, retreat as retreatStage, stageLabel } from '../visualizer/executionState';
import { useVizCore } from '../visualizer/useVizCore';
import { NodeMath } from './NodeMath';
import { CARDS, STEP_CARD } from './stackCards';
import { PlayBar } from './PlayBar';
import { StepProvider, useStepTracker } from './Step';
import { TermFocusContext } from './TermFocus';
import { SCENES, TOC, type StepId, type StepScene } from './steps';
import './shadcn.css';
import './scrolly.css';

/** Who moves the graph: the text as it scrolls, or the reader. */
type Driver = 'paper' | 'free';

const ExecutionVisualizer = lazy(() => import('../visualizer/ExecutionVisualizer'));
const STEP_IDS = Object.keys(SCENES) as StepId[];
const STAGE_KEY = 'gpt2rkt:stage';

/**
 * The write-up on the left, the hero visualizer pinned on the right.
 * The paper leads by default; the dock hands the wheel to the reader.
 */
export function ScrollyLayout({ theme, children }: { theme: string; children: ReactNode }) {
  const core = useVizCore();
  const tracker = useStepTracker(STEP_IDS[0] ?? 'intro');
  const [driver, setDriver] = useState<Driver>('paper');
  const api = useRef<VizApi | null>(null);
  const { setState, trace, state, run } = core;
  const active = tracker.active;
  const scene: StepScene = SCENES[active];
  const following = driver === 'paper';

  // On a stacked page (phones, portrait tablets) the visualizer sits behind a toggle, so the
  // text gets the screen. Closed by default; the choice holds for the visit. Wider screens
  // always show it side by side, and the CSS hides the toggle there.
  const [showStage, setShowStage] = useState(() => {
    try { return sessionStorage.getItem(STAGE_KEY) === 'open'; } catch { return false; }
  });
  const toggleStage = () => setShowStage((open) => {
    const next = !open;
    try { sessionStorage.setItem(STAGE_KEY, next ? 'open' : 'closed'); } catch { /* private mode */ }
    return next;
  });

  // A .rkt term hovered in the text points the graph at its module node.
  const [termFocus, setTermFocus] = useState<string | null>(null);
  useEffect(() => {
    setState((current) => ({ ...current, hoveredNodeId: termFocus }));
  }, [termFocus, setState]);

  const take = useCallback(() => setDriver('free'), []);
  const follow = useCallback(() => setDriver('paper'), []);

  // Crossing into a new block of text hands the wheel back to the paper.
  const lastActive = useRef(active);
  useEffect(() => {
    if (lastActive.current === active) return;
    lastActive.current = active;
    setDriver('paper');
  }, [active]);

  // Following the paper: set the scene for this step.
  useEffect(() => {
    if (!following) return;
    setState((current) => ({
      ...current,
      mode: scene.mode,
      activeStageId: scene.stages[0] ?? current.activeStageId,
      stepIndex: scene.sweep ? scene.sweep[0] : (scene.step ?? current.stepIndex),
      playing: false,
      inspect: false,
      selectedNodeId: null,
      hoveredNodeId: null,
      pair: null,
      hoverPair: null,
    }));
  }, [following, scene, setState]);

  // Then walk the pulse through this step's stages, or sweep its recorded steps.
  // The play bar pauses it, or steps it one stage at a time.
  // While following the text, the pulse walks the section's stages on its own.
  const [autoplay] = useState(() => !stillness());
  const advance = useCallback((dir: 1 | -1) => {
    setState((current) => {
      if (scene.sweep) {
        const [from, to] = scene.sweep;
        const next = current.stepIndex + dir;
        return { ...current, stepIndex: next > to ? from : next < from ? to : next };
      }
      const at = Math.max(0, scene.stages.indexOf(current.activeStageId));
      const n = scene.stages.length;
      const stage = scene.stages[(at + dir + n) % n];
      return stage ? { ...current, activeStageId: stage } : current;
    });
  }, [scene, setState]);
  const canStep = scene.stages.length > 1 || Boolean(scene.sweep);

  // A node's card being open pauses the walk, so the graph stays put while it's read.
  const reading = state.inspect;
  useEffect(() => {
    if (!following || !trace || !autoplay || !canStep || reading) return;
    const timer = window.setInterval(() => advance(1), (scene.sweep ? 700 : 1500) / state.speed);
    return () => window.clearInterval(timer);
  }, [following, scene, trace, autoplay, canStep, advance, state.speed, reading]);

  // The play bar runs the whole recorded pass, every stage of every token, on its own.
  // Scrolling into the next block of text hands the graph back to the paper.
  const playWhole = () => {
    if (!trace) return;
    if (!state.playing) setDriver('free');
    run();
  };
  const stepWhole = (dir: 1 | -1) => {
    if (!trace) return;
    setDriver('free');
    setState((current) => {
      const base = current.mode === 'explore' ? { ...current, mode: 'execute' as const } : current;
      return dir > 0 ? advanceStage(base, trace) : retreatStage(base, trace);
    });
  };


  // The sentence under the graph writes itself as the page is read: one more generated
  // word per stretch of scroll, complete just before the end card.
  // It stops at the last full stop, so it never ends on a half-started sentence.
  const total = (() => {
    if (!trace) return 0;
    const prompt = trace.run.promptTokenIds.length;
    const made = Math.max(0, trace.run.tokens.length - prompt);
    for (let i = trace.run.tokens.length - 1; i >= prompt; i -= 1) {
      if (/[.!?]\s*$/.test(trace.run.tokens[i]?.text ?? '')) return i - prompt + 1;
    }
    return made;
  })();
  const [revealCount, setRevealCount] = useState(0);
  useEffect(() => {
    if (!total) return;
    let frame = 0;
    // The page's scrollable height only changes on resize or reflow, so read it then,
    // not on every scroll frame (reading scrollHeight can force a layout).
    let room = document.documentElement.scrollHeight - window.innerHeight;
    const remeasure = () => { room = document.documentElement.scrollHeight - window.innerHeight; };
    const sizes = new ResizeObserver(() => { remeasure(); onScroll(); });
    sizes.observe(document.body);
    const measure = () => {
      frame = 0;
      const progress = room > 0 ? Math.min(1, window.scrollY / (room * 0.94)) : 1;
      setRevealCount(Math.round(progress * total));
    };
    const onScroll = () => { if (!frame) frame = window.requestAnimationFrame(measure); };
    measure();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
      sizes.disconnect();
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [total]);

  // The file this step's code card shows, ringed in the graph.
  const cardId = STEP_CARD[active];
  const litModule = CARDS.find((card) => card.id === cardId)?.moduleId ?? null;


  return (
    <TooltipProvider delayDuration={250}>
      <TermFocusContext.Provider value={setTermFocus}>
      <StepProvider value={tracker}>
        <div className="sc-grid" data-driver={driver} data-stage={showStage ? 'open' : 'closed'}>
          <div className="sc-text">
            {children}
          </div>

          <aside className="sc-stage" aria-label="The recorded forward pass, following the text">
            <div className="sc-stage-inner">
              <PlayBar
                playing={state.playing}
                disabled={!trace}
                label={state.mode === 'explore' ? 'Forward pass' : stageLabel(state.activeStageId)}
                position={state.mode === 'explore' ? null : `token ${state.stepIndex + 1} / ${LAST_STEP + 1}`}
                onToggle={playWhole}
                onStep={stepWhole}
                speed={state.speed}
                onSpeed={() => setState((current) => ({
                  ...current,
                  speed: current.speed === 1 ? 2 : current.speed === 2 ? 4 : 1,
                }))}
              />

              <div className="sc-viz viz-hero" data-viz-root="" data-no-pop="" data-theme={theme}>
                <Suspense fallback={<p className="sc-wait">Reading the recorded run.</p>}>
                  <ExecutionVisualizer
                    core={core}
                    chrome="stage"
                    focusId={termFocus ?? (following ? scene.focus : null)}
                    onUserInteract={take}
                    apiRef={api}
                    revealCount={following ? revealCount : null}
                    litId={termFocus || state.inspect ? null : litModule}
                  />
                </Suspense>
                <NodeMath core={core} />
              </div>


              {/* The playground dock (PlaygroundDock.tsx) is off for now. */}

            </div>
          </aside>
        </div>

        {/* The ruler keeps the length the write-up's rail has always had. */}
        <TickRuler length={(height) => Math.min(height * 0.46, 380)} sections={TOC} />

        <button
          type="button"
          className="sc-stage-toggle"
          aria-pressed={showStage}
          onClick={toggleStage}
        >
          {showStage ? <X size={14} aria-hidden="true" /> : <Network size={14} aria-hidden="true" />}
          {showStage ? 'Hide visualizer' : 'Visualizer'}
        </button>

        {following ? null : (
          <button type="button" className="sc-return" onClick={follow}>
            <span aria-hidden="true">↺</span> You're exploring · back to the text
          </button>
        )}
      </StepProvider>
      </TermFocusContext.Provider>
    </TooltipProvider>
  );
}
