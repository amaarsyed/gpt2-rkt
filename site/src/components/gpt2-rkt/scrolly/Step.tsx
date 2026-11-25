import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import type { StepId } from './steps';

interface StepRegistry {
  active: StepId;
  observe: (el: HTMLElement) => () => void;
}

const StepContext = createContext<StepRegistry | null>(null);

/**
 * Picks the step that crosses the reading line, a band around 45% down the viewport.
 * One IntersectionObserver for the whole page. Before the first step reaches the line, the intro is active.
 */
export function useStepTracker(initial: StepId): StepRegistry {
  const [active, setActive] = useState<StepId>(initial);
  const observer = useRef<IntersectionObserver | null>(null);
  const pending = useRef(new Set<HTMLElement>());

  useEffect(() => {
    const io = new IntersectionObserver(
      (entries) => {
        const hit = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
        const id = hit?.target instanceof HTMLElement ? hit.target.dataset.step : undefined;
        if (id) setActive(id as StepId);
      },
      { rootMargin: '-44% 0px -54% 0px', threshold: 0 },
    );
    observer.current = io;
    for (const el of pending.current) io.observe(el);
    pending.current.clear();
    return () => {
      io.disconnect();
      observer.current = null;
    };
  }, []);

  const observe = (el: HTMLElement) => {
    if (observer.current) observer.current.observe(el);
    else pending.current.add(el);
    return () => {
      observer.current?.unobserve(el);
      pending.current.delete(el);
    };
  };

  return { active, observe };
}

export function StepProvider({ value, children }: { value: StepRegistry; children: ReactNode }) {
  return <StepContext.Provider value={value}>{children}</StepContext.Provider>;
}

/** A block of the write-up that drives the pinned visualizer while it sits on the reading line. */
export function Step({ id, children, className }: { id: StepId; children: ReactNode; className?: string }) {
  const registry = useContext(StepContext);
  const ref = useRef<HTMLDivElement>(null);
  const observe = registry?.observe;

  useEffect(() => {
    const el = ref.current;
    if (!el || !observe) return;
    return observe(el);
    // The registry's observe is stable for the page's lifetime.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const on = registry?.active === id;
  return (
    <div
      ref={ref}
      id={id}
      data-step={id}
      data-active={on ? '' : undefined}
      className={`wu-step${className ? ` ${className}` : ''}`}
    >
      {children}
    </div>
  );
}
