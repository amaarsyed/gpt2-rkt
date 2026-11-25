import { useEffect, useRef, useState } from 'react';

const STEPS = [
  { target: 'nodes', line: 'Each node is one GPT-2 stage.' },
  { target: 'legend', line: 'Color is the kind of wire.' },
  { target: 'scrubber', line: 'One token generated per step.' },
  { target: 'modes', line: 'Explore, Execute, or Inspect.' },
  { target: 'window', line: 'The card explains this stage.' },
  { target: 'controls', line: '← → step. Space plays.' },
] as const;

export const TOUR_KEY = 'gpt2-viz-tour';

interface Props {
  open: boolean;
  onClose: () => void;
  onStep?: (index: number) => void;
}

export function VizTour({ open, onClose, onStep }: Props) {
  const [step, setStep] = useState(0);
  const [box, setBox] = useState<{ top: number; left: number; width: number; height: number; root: number; stageTop: number; stageBottom: number } | null>(null);
  const dialog = useRef<HTMLDivElement>(null);
  const stepNotify = useRef(onStep);
  stepNotify.current = onStep;
  const current = STEPS[step] ?? STEPS[0];

  useEffect(() => {
    if (!open) return;
    stepNotify.current?.(step);
    const place = () => {
      const root = dialog.current?.closest('.viz');
      const target = root?.querySelector(`[data-tour="${current.target}"]`);
      if (!(root instanceof HTMLElement) || !(target instanceof HTMLElement)) {
        setBox(null);
        return;
      }
      const outer = root.getBoundingClientRect();
      const inner = target.getBoundingClientRect();
      const stage = root.querySelector('.viz-stage')?.getBoundingClientRect();
      setBox({
        top: inner.top - outer.top - 4,
        left: inner.left - outer.left - 4,
        width: inner.width + 8,
        height: inner.height + 8,
        root: outer.width,
        stageTop: stage ? stage.top - outer.top : 0,
        stageBottom: stage ? stage.bottom - outer.top : outer.height,
      });
    };
    place();
    const frame = window.requestAnimationFrame(() => window.requestAnimationFrame(place));
    const later = window.setTimeout(place, 80);
    window.addEventListener('resize', place);
    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(later);
      window.removeEventListener('resize', place);
    };
  }, [open, step, current.target]);

  useEffect(() => {
    if (!open) return;
    const root = dialog.current;
    if (!root) return;
    const items = () => [...root.querySelectorAll<HTMLButtonElement>('button')].filter((node) => !node.disabled);
    root.querySelector<HTMLElement>('[data-tour-next]')?.focus({ preventScroll: true });
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        onClose();
        return;
      }
      if (event.key === 'Enter' && !(event.target instanceof HTMLElement && event.target.dataset.tour === 'skip')) {
        event.preventDefault();
        if (step === STEPS.length - 1) onClose();
        else setStep((value) => value + 1);
        return;
      }
      if (event.key !== 'Tab') return;
      const list = items();
      if (!list.length) return;
      const first = list[0];
      const last = list[list.length - 1];
      if (!first || !last) return;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus({ preventScroll: true });
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus({ preventScroll: true });
      }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [open, step, onClose]);

  if (!open) return null;
  const last = step === STEPS.length - 1;
  const next = () => {
    if (last) onClose();
    else setStep((value) => value + 1);
  };
  const call = (() => {
    if (!box) return { left: 12, top: 36 };
    if (current.target === 'scrubber') {
      return {
        left: Math.max(8, box.root - 252),
        top: Math.max(8, box.top - 22),
      };
    }
    const below = box.top + box.height + 6;
    const inStage = below > box.stageTop + 4 && below < box.stageBottom - 22 && box.height < 180;
    return {
      left: inStage ? Math.max(8, Math.min(box.left, Math.max(8, box.root - 280))) : 12,
      top: inStage ? below : box.stageTop + 8,
    };
  })();

  return (
    <div className="viz-tour" data-tour-open="">
      {box ? <div className="viz-tour-spot" style={box} /> : null}
      <div
        className="viz-tour-call"
        style={call}
        ref={dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby="viz-tour-title"
        data-no-pop=""
      >
        <p id="viz-tour-title">{current.line}</p>
        <p className="viz-tour-actions">
          <button type="button" data-no-pop="" data-tour-next="" onClick={next}>{last ? 'Done' : 'Next'}</button>
          <span aria-hidden="true">·</span>
          <button type="button" data-no-pop="" data-tour="skip" onClick={onClose}>Skip</button>
        </p>
      </div>
    </div>
  );
}
