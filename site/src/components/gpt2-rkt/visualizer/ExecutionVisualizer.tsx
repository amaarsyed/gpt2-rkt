import { Fragment, useEffect, useRef, useState, type KeyboardEvent, type MutableRefObject, type PointerEvent, type ReactNode } from 'react';
import { activeQueryRow, strongestKey, causalRow, tokenText } from './attentionMath';
import { LAST_LAYER, LAST_STEP } from './executionState';
import {
  jumpStep,
  stageLabel,
  stepFacts,
  tokenRibbon,
  type AttentionPair,
  type Speed,
} from './executionState';
import { executionGraph, nodeById } from './graphModel';
import { createExecutionCanvas, type ExecutionCanvas } from './canvasRenderer';
import { FloatWindow, type WindowTab } from './FloatWindow';
import { InspectDock } from './InspectDock';
import { SourcePane } from './SourcePane';
import { CODE_NOTES, MODULE_TIP, STAGE_TIP, STAGE_WHAT } from './stageCopy';
import { isBlockStage } from './traceTypes';
import { useVizCore, type VizCore } from './useVizCore';
import { TOUR_KEY, VizTour } from './VizTour';
import './visualizer.css';

/** What each node colour stands for, said on the node's hover tip. */
const ROLE_NAME = {
  ink: 'token',
  residual: 'residual',
  attention: 'attention',
  memory: 'cache',
  future: 'futures',
  module: 'Racket file',
} as const;

const SPEEDS: Speed[] = [1, 2, 4];

export interface ExecutionVisualizerProps {
  /** Lifted state. Omitted, the visualizer owns its own, exactly as in the hero. */
  core?: VizCore;
  /** 'stage' drops the built-in toolbar and scrubber; an outside dock drives it instead. */
  chrome?: 'full' | 'stage';
  /** A node for the camera to frame, eased, without opening Inspect. */
  focusId?: string | null;
  /** Called when the reader takes the wheel: clicks a node or steps with the keys. */
  onUserInteract?: () => void;
  /** Lets an outside dock fit the camera or go fullscreen. */
  apiRef?: MutableRefObject<VizApi | null>;
  /** A node to ring, such as the module whose file the code stack shows. */
  litId?: string | null;
  /** In 'stage', how many generated words the sentence shows. Omitted, it follows the recorded step. */
  revealCount?: number | null;
}

export interface VizApi {
  fit: () => void;
  fullscreen: () => void;
}

export default function ExecutionVisualizer({ core, chrome = 'full', focusId: guideId = null, onUserInteract, apiRef, revealCount = null, litId = null }: ExecutionVisualizerProps = {}) {
  const own = useVizCore(!core);
  const { trace, error, state, setState, run, setMode, toggleInspect } = core ?? own;
  const stage = chrome === 'stage';
  const stateRef = useRef(state);
  const traceRef = useRef(trace);
  const guideRef = useRef(guideId);
  const interactRef = useRef(onUserInteract);
  guideRef.current = guideId;
  const litRef = useRef(litId);
  litRef.current = litId;
  interactRef.current = onUserInteract;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewRef = useRef<ExecutionCanvas | null>(null);
  const spotsRef = useRef<(spots: { id: string; x: number; y: number }[]) => void>(() => undefined);
  const [spots, setSpots] = useState<{ id: string; x: number; y: number }[]>([]);
  const [focusId, setFocusId] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState(false);
  const [tab, setTab] = useState<WindowTab>('what');
  const [fullscreen, setFullscreen] = useState(false);
  const [fallbackFull, setFallbackFull] = useState(false);
  const [tour, setTour] = useState(false);
  stateRef.current = state;
  traceRef.current = trace;
  spotsRef.current = (next) => {
    setSpots((prev) => {
      if (prev.length === next.length && prev.every((item, index) => {
        const other = next[index];
        return other && item.id === other.id && Math.abs(item.x - other.x) < 1.5 && Math.abs(item.y - other.y) < 1.5;
      })) return prev;
      return next;
    });
  };

  useEffect(() => {
    if (stage) return;
    try {
      if (!localStorage.getItem(TOUR_KEY)) setTour(true);
    } catch { /* private mode still runs the canvas */ }
  }, [stage]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const view = createExecutionCanvas(canvas, {
      getState: () => stateRef.current,
      onHover: (id) => {
        setState((current) => (current.hoveredNodeId === id ? current : { ...current, hoveredNodeId: id }));
      },
      onSelect: (id) => {
        if (id) interactRef.current?.();
        const node = executionGraph().nodes.find((item) => item.id === id) ?? null;
        setState((current) => ({
          ...current,
          selectedNodeId: id,
          playing: node ? false : current.playing,
          activeStageId: node?.stageId ?? current.activeStageId,
          inspect: node ? true : current.inspect,
          mode: node ? 'execute' : current.mode,
          pair: node?.stageId && node.stageId === current.activeStageId ? current.pair : null,
        }));
        if (node) {
          setCollapsed(false);
          setTab('what');
        }
      },
      onPositions: (next) => spotsRef.current(next),
      getFocus: () => guideRef.current,
      getLit: () => litRef.current,
    });
    viewRef.current = view;
    view.wake();
    return () => {
      view.destroy();
      viewRef.current = null;
    };
  }, []);

  useEffect(() => { viewRef.current?.wake(); }, [state, fallbackFull, guideId, litId]);

  // Opening Inspect from anywhere, the dock included, shows the window on its first tab.
  useEffect(() => {
    if (!state.inspect) return;
    setCollapsed(false);
    setTab('what');
  }, [state.inspect]);

  useEffect(() => {
    let full = false;
    let swallow = false;
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      if (document.querySelector('[data-tour-open]')) return;
      if (stateRef.current.inspect) {
        event.preventDefault();
        event.stopPropagation();
        swallow = false;
        setCollapsed(false);
        setState((current) => (current.inspect ? { ...current, inspect: false, hoverPair: null } : current));
        return;
      }
      if (document.fullscreenElement || full || swallow) {
        event.preventDefault();
        if (document.fullscreenElement) void document.exitFullscreen().catch(() => undefined);
        full = false;
        swallow = false;
        return;
      }
      if (fallbackFull) {
        event.preventDefault();
        setFallbackFull(false);
      }
    };
    const onFull = () => {
      const active = Boolean(document.fullscreenElement);
      if (full && !active) swallow = true;
      full = active;
      setFullscreen(active);
    };
    window.addEventListener('keydown', onKey, true);
    document.addEventListener('fullscreenchange', onFull);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      document.removeEventListener('fullscreenchange', onFull);
    };
  }, [fallbackFull]);

  const facts = trace ? stepFacts(trace, state) : null;
  const ribbon = trace ? tokenRibbon(trace, state.stepIndex) : null;
  const generated = ribbon?.generated ?? [];

  const onVizKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!trace) return;
    const target = event.target;
    if (target instanceof HTMLElement) delete target.dataset.pointer;
    const field = target instanceof HTMLElement && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable);
    if (field) return;
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault();
      interactRef.current?.();
      const dir = event.key === 'ArrowRight' ? 1 : -1;
      setState((current) => jumpStep(current, trace, current.stepIndex + dir));
      return;
    }
    if (event.key !== ' ') return;
    const native = target instanceof HTMLElement && (target.tagName === 'BUTTON' || target.tagName === 'A');
    if (native) return;
    event.preventDefault();
    interactRef.current?.();
    run();
  };

  const markPointer = (event: PointerEvent<HTMLDivElement>) => {
    const node = event.target instanceof Element ? event.target.closest('button, a, canvas') : null;
    if (node instanceof HTMLElement) node.dataset.pointer = '';
  };

  const openInspect = toggleInspect;

  const goFullscreen = async () => {
    const node = canvasRef.current?.closest('[data-viz-root]');
    if (!(node instanceof HTMLElement) || typeof node.requestFullscreen !== 'function') {
      setFallbackFull((value) => !value);
      return;
    }
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if (fallbackFull) setFallbackFull(false);
      else await node.requestFullscreen();
    } catch {
      setFallbackFull(true);
    }
  };

  if (apiRef) apiRef.current = { fit: () => viewRef.current?.fit(), fullscreen: () => void goFullscreen() };

  const closeTour = () => {
    try { localStorage.setItem(TOUR_KEY, '1'); } catch { /* still closes */ }
    setTour(false);
  };

  const tipId = focusId ?? state.hoveredNodeId;
  const tip = tipId ? (STAGE_TIP[tipId] ?? MODULE_TIP[tipId]) : null;
  const tipSpot = spots.find((item) => item.id === tipId);
  const tipNode = tipId ? nodeById(executionGraph(), tipId) : null;
  const queryRow = activeQueryRow(state.stepIndex);
  const shownPair: AttentionPair | null = state.inspect && state.activeStageId === 'attend' && trace
    ? (state.hoverPair ?? state.pair ?? { q: queryRow, k: strongestKey(causalRow(trace, state.activeLayer, state.head, queryRow)) })
    : null;
  const showLayer = state.mode !== 'explore' && isBlockStage(state.activeStageId);
  const selected = nodeById(executionGraph(), state.selectedNodeId);
  const windowStage = selected?.stageId ?? state.activeStageId;
  const windowState = windowStage === state.activeStageId ? state : { ...state, activeStageId: windowStage };
  const windowFacts = trace ? stepFacts(trace, windowState) : facts;
  const what = (state.selectedNodeId && (STAGE_WHAT[state.selectedNodeId] ?? MODULE_TIP[state.selectedNodeId]))
    || STAGE_WHAT[windowStage]
    || STAGE_TIP[windowStage];

  return (
    <div
      className={`viz${fallbackFull ? ' is-fallback-full' : ''}${stage ? ' is-stage' : ''}`}
      data-viz=""
      data-no-pop=""
      role="region"
      aria-label="Recorded GPT-2 forward pass"
      onKeyDown={onVizKey}
      onPointerDown={markPointer}
    >
      {stage ? null : <div className="viz-top">
        <div className="viz-modes" role="group" aria-label="Canvas mode" data-tour="modes" data-no-pop="">
          <button type="button" data-no-pop="" aria-pressed={state.mode === 'explore' && !state.inspect} onClick={() => setMode('explore')}>Explore</button>
          <button type="button" data-no-pop="" aria-pressed={state.mode === 'execute' && !state.inspect} onClick={() => setMode('execute')}>Execute</button>
          <button type="button" data-no-pop="" aria-pressed={state.inspect} onClick={openInspect} disabled={!trace}>Inspect</button>
        </div>
        <div className="viz-iconbar" role="toolbar" aria-label="Playback" data-tour="controls" data-no-pop="">
          <IconButton label={state.playing ? 'Pause' : 'Play'} primary onClick={run} disabled={!trace}>
            {state.playing ? <PauseIcon /> : <PlayIcon />}
          </IconButton>
          <IconButton label="Step back" onClick={() => trace && setState((current) => jumpStep(current, trace, current.stepIndex - 1))} disabled={!trace}>
            <StepBackIcon />
          </IconButton>
          <IconButton label="Step forward" onClick={() => trace && setState((current) => jumpStep(current, trace, current.stepIndex + 1))} disabled={!trace}>
            <StepForwardIcon />
          </IconButton>
          <IconButton
            label={`Speed ${state.speed}×`}
            onClick={() => setState((current) => ({ ...current, speed: SPEEDS[(SPEEDS.indexOf(current.speed) + 1) % SPEEDS.length] ?? 1 }))}
          >
            <span className="viz-speed">{state.speed}×</span>
          </IconButton>
          {showLayer ? (
            <select
              className="viz-layer"
              aria-label="Transformer layer"
              data-no-pop=""
              value={state.activeLayer}
              onChange={(event) => setState((current) => ({ ...current, activeLayer: Number(event.target.value) }))}
            >
              {Array.from({ length: LAST_LAYER + 1 }, (_, layer) => <option key={layer} value={layer}>L{layer}</option>)}
            </select>
          ) : null}
          <IconButton label="Fit" onClick={() => viewRef.current?.fit()}><FitIcon /></IconButton>
          <IconButton label="Reset" onClick={() => setState((current) => ({ ...current, stepIndex: 0, activeStageId: 'bpe-merge', playing: false }))}>
            <ResetIcon />
          </IconButton>
          <IconButton label={fullscreen || fallbackFull ? 'Exit fullscreen' : 'Fullscreen'} pressed={fullscreen || fallbackFull} onClick={() => void goFullscreen()}>
            <FullIcon />
          </IconButton>
          <IconButton label="Replay the tour" onClick={() => setTour(true)}><HelpIcon /></IconButton>
        </div>
      </div>}

      <div className="viz-caption">
        <p>gpt2-rkt · one token's forward pass <span className="viz-hint">click a node, then ← → to step</span></p>
        <span className="viz-count" data-readout="">{String(state.stepIndex).padStart(2, '0')} / {String(LAST_STEP).padStart(2, '0')}</span>
      </div>

      <div className="viz-stage" data-tour="nodes">
        <canvas
          ref={canvasRef}
          className="viz-canvas"
          data-viz-canvas=""
          tabIndex={0}
          aria-label="Execution canvas. Explore lays out the Racket modules. Execute follows one recorded forward pass."
        />
        {spots.map((spot) => {
          const text = STAGE_TIP[spot.id] ?? MODULE_TIP[spot.id];
          if (!text) return null;
          return (
            <button
              key={spot.id}
              type="button"
              className="viz-hit"
              style={{ left: spot.x, top: spot.y }}
              data-node={spot.id}
              data-no-pop=""
              aria-label={text}
              onFocus={() => setFocusId(spot.id)}
              onBlur={() => setFocusId((current) => (current === spot.id ? null : current))}
            />
          );
        })}
        {tip && tipSpot ? (
          <div className="viz-tip" role="tooltip" data-role={tipNode?.role} style={{ left: tipSpot.x, top: tipSpot.y }}>
            {tipNode ? (
              <span className="viz-tip-head">
                <i aria-hidden="true" />
                <b>{tipNode.label}</b>
                <span className="viz-tip-kind">{ROLE_NAME[tipNode.role]}</span>
              </span>
            ) : null}
            <span className="viz-tip-text">{tip}</span>
          </div>
        ) : null}
        {state.inspect && collapsed && !stage ? (
          <button type="button" className="viz-pill" data-tour="window" data-no-pop="" onClick={() => setCollapsed(false)}>
            {stageLabel(state.activeStageId)}
          </button>
        ) : null}
      </div>

      <div className="viz-legend" data-tour="legend">
        <span><i data-swatch="residual" /> residual</span>
        <span><i data-swatch="attention" /> attention</span>
        <span><i data-swatch="future" /> future</span>
        <span><i data-swatch="memory" /> cache</span>
      </div>

      <div className="viz-bottom" data-tour="scrubber">
        {stage ? (
          <p className="wu-sentence viz-sentence" aria-live="polite">
            {(trace?.run.tokens ?? []).map((token, index) => {
              const text = trace ? tokenText(trace, index) : token.text;
              const promptCount = trace ? trace.run.promptTokenIds.length : 0;
              const made = index < promptCount ? -1 : index - promptCount;
              const litCount = revealCount ?? generated.filter((item) => item.lit).length;
              const role = shownPair && index === shownPair.q ? 'read' : '';
              const state = role
                || (made < 0
                  ? (litCount === 0 ? 'new' : 'done')
                  : made < litCount - 1 ? 'done' : made === litCount - 1 ? 'new' : 'hidden');
              return (
                <Fragment key={`${token.id}-${index}`}>
                  {text.startsWith(' ') ? ' ' : null}
                  <span className="wu-word" data-state={state}>{text.trim()}</span>
                </Fragment>
              );
            })}
          </p>
        ) : (
          <p className="viz-ribbon" aria-live="polite">
            {(trace?.run.tokens ?? [{ id: 0, text: ribbon?.prompt ?? 'Once upon a time' }]).map((token, index) => {
              const promptCount = trace ? trace.run.promptTokenIds.length : 0;
              const lit = !trace || index < promptCount || generated[index - promptCount]?.lit;
              const role = shownPair && index === shownPair.q ? 'query' : shownPair && index === shownPair.k ? 'key' : '';
              return (
                <span key={`${token.id}-${index}`} className={role ? `is-${role}` : lit ? 'is-lit' : undefined}>
                  {trace ? tokenText(trace, index) : token.text}
                </span>
              );
            })}
          </p>
        )}
        {error ? <p className="viz-status">{error}</p> : null}
        {stage ? null : <div className="viz-scrub-row">
          <label className="viz-scrub">
            <input
              type="range"
              min={0}
              max={LAST_STEP}
              step={1}
              value={state.stepIndex}
              disabled={!trace}
              aria-label="Recorded step"
              data-no-pop=""
              onChange={(event) => {
                if (!trace) return;
                setState((current) => jumpStep(current, trace, Number(event.target.value)));
              }}
            />
          </label>
          <span className="viz-scrub-read">{state.stepIndex} / {LAST_STEP}</span>
          <span className="viz-keys">← → step · Space play</span>
        </div>}
      </div>
      {state.inspect && !collapsed && !stage ? (
        <FloatWindow
          title={selected?.label ?? stageLabel(state.activeStageId)}
          tab={tab}
          onTab={setTab}
          onClose={() => setState((current) => ({ ...current, inspect: false, hoverPair: null }))}
          onCollapse={() => setCollapsed(true)}
          onPlace={() => viewRef.current?.fit()}
        >
          {tab === 'what' ? (
            <div className="viz-what">
              <p>{what}</p>
              <p className="viz-what-live">{error ?? windowFacts?.sentence ?? 'Reading the recorded run.'}</p>
              {windowFacts?.stageMs != null ? <p className="viz-what-ms">{windowFacts.stageMs.toFixed(2)} ms</p> : null}
            </div>
          ) : null}
          {tab === 'math' && trace && facts ? (
            <InspectDock
              trace={trace}
              state={windowState}
              schedule={windowFacts?.schedule ?? null}
              stageMs={windowFacts?.stageMs ?? null}
              onHead={(head) => setState((current) => ({ ...current, head }))}
              onPhase={(mathPhase) => setState((current) => ({ ...current, mathPhase }))}
              onPair={(q, k) => setState((current) => ({ ...current, pair: { q, k } }))}
              onHoverPair={(q, k) => setState((current) => ({ ...current, hoverPair: q === null || k === null ? null : { q, k } }))}
            />
          ) : null}
          {tab === 'code' && trace ? (
            <SourcePane
              stage={windowStage}
              callSnippet={trace.sourceMap[windowStage]?.snippet ?? null}
              notes={CODE_NOTES[windowStage] ?? []}
            />
          ) : null}
        </FloatWindow>
      ) : null}
      <VizTour
        open={tour && !stage}
        onClose={closeTour}
        onStep={(index) => {
          if (index !== 4) return;
          setCollapsed(false);
          setState((current) => (current.inspect ? current : {
            ...current,
            inspect: true,
            mode: 'execute',
            playing: false,
            activeStageId: 'attend',
            selectedNodeId: 'attend',
          }));
        }}
      />
    </div>
  );
}

export function IconButton({
  label,
  onClick,
  disabled,
  pressed,
  primary,
  children,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  pressed?: boolean;
  primary?: boolean;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      data-no-pop=""
      className={primary ? 'is-primary' : undefined}
      aria-label={label}
      aria-pressed={pressed}
      disabled={disabled}
      onClick={onClick}
    >
      {children}
      <span className="viz-icon-tip" role="tooltip">{label}</span>
    </button>
  );
}

export function PlayIcon() {
  return <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M5 3.5v9l8-4.5-8-4.5z" /></svg>;
}
export function PauseIcon() {
  return <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4.5 3h2.2v10H4.5zM9.3 3h2.2v10H9.3z" /></svg>;
}
export function StepBackIcon() {
  return <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 3h1.6v10H4zM12.5 3.5 6.2 8l6.3 4.5V3.5z" /></svg>;
}
export function StepForwardIcon() {
  return <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M10.4 3H12v10h-1.6zM3.5 3.5 9.8 8 3.5 12.5V3.5z" /></svg>;
}
export function FitIcon() {
  return <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 6.5V3h3.5M10 3h3v3.5M13 9.5V13H9.5M6 13H3V9.5" /></svg>;
}
export function ResetIcon() {
  return <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8a4.5 4.5 0 1 0 1.2-3.1M3.2 2.8v2.8h2.8" /></svg>;
}
export function FullIcon() {
  return <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 6V3h3M10 3h3v3M13 10v3h-3M6 13H3v-3" /></svg>;
}
export function HelpIcon() {
  return <svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6" /><path d="M6.3 6.2a1.7 1.7 0 1 1 2.2 1.6c-.5.2-.7.5-.7 1V9.2M8 11.2v.6" /></svg>;
}
