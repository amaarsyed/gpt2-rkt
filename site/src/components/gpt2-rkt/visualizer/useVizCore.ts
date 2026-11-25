import { useCallback, useEffect, useState, type Dispatch, type SetStateAction } from 'react';
import { LAST_STEP, advance, initialState, jumpStep, stageOrder, type ExecutionState } from './executionState';
import { loadTrace } from './traceLoader';
import type { ExecutionTrace } from './traceTypes';

/** The visualizer's state and the recorded trace. The hero owns one; the scroll-led layout lifts it up. */
export interface VizCore {
  trace: ExecutionTrace | null;
  error: string | null;
  state: ExecutionState;
  setState: Dispatch<SetStateAction<ExecutionState>>;
  run: () => void;
  step: (dir: 1 | -1) => void;
  setMode: (mode: ExecutionState['mode']) => void;
  toggleInspect: () => void;
}

function atEnd(current: ExecutionState, book: ExecutionTrace) {
  const order = stageOrder(current, book);
  return current.stepIndex === LAST_STEP && current.activeStageId === order[order.length - 1];
}

export function useVizCore(enabled = true): VizCore {
  const [trace, setTrace] = useState<ExecutionTrace | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [state, setState] = useState<ExecutionState>(initialState);

  useEffect(() => {
    if (!enabled) return;
    let live = true;
    loadTrace()
      .then((next) => { if (live) setTrace(next); })
      .catch((reason: unknown) => {
        if (live) setError(reason instanceof Error ? reason.message : 'The trace could not be read.');
      });
    return () => { live = false; };
  }, [enabled]);

  useEffect(() => {
    if (!enabled || !state.playing || !trace) return;
    const timer = window.setTimeout(() => setState((current) => advance(current, trace)), 780 / state.speed);
    return () => window.clearTimeout(timer);
  }, [enabled, state.playing, state.speed, state.stepIndex, state.activeStageId, trace]);

  const run = useCallback(() => {
    if (!trace) return;
    setState((current) => {
      if (current.playing) return { ...current, playing: false };
      if (atEnd(current, trace)) return { ...current, mode: 'execute', stepIndex: 0, activeStageId: 'bpe-merge', playing: true };
      return { ...current, mode: 'execute', playing: true };
    });
  }, [trace]);

  const step = useCallback((dir: 1 | -1) => {
    if (!trace) return;
    setState((current) => jumpStep(current, trace, current.stepIndex + dir));
  }, [trace]);

  const setMode = useCallback((mode: ExecutionState['mode']) => {
    setState((current) => ({
      ...current,
      mode,
      playing: mode === 'explore' ? false : current.playing,
      inspect: false,
      hoverPair: null,
    }));
  }, []);

  const toggleInspect = useCallback(() => {
    setState((current) => {
      if (current.inspect) return { ...current, inspect: false, hoverPair: null };
      return {
        ...current,
        inspect: true,
        mode: 'execute',
        playing: false,
        activeStageId: 'attend',
        activeLayer: 0,
        head: 0,
        mathPhase: 2,
        selectedNodeId: 'attend',
        pair: null,
        hoverPair: null,
      };
    });
  }, []);

  return { trace, error, state, setState, run, step, setMode, toggleInspect };
}
