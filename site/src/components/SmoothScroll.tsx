import { useEffect, useState, type ReactNode } from 'react';
import { ReactLenis, useLenis } from 'lenis/react';
import 'lenis/dist/lenis.css';

const OPTIONS = {
  duration: 1.25,
  lerp: 0.07,
  smoothWheel: true,
  autoRaf: true,
  syncTouch: false,
} as const;

type SmoothScrollProps = {
  children: ReactNode;
};

/** Slightly slow, inertia-style page scrolling via Lenis. */
export function SmoothScroll({ children }: SmoothScrollProps) {
  const [enabled, setEnabled] = useState(() => {
    if (typeof window === 'undefined') return true;
    return !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  });

  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = () => setEnabled(!mq.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  if (!enabled) return children;

  return (
    <ReactLenis root options={OPTIONS}>
      {children}
    </ReactLenis>
  );
}

/** Pause Lenis while overlays lock body scroll (e.g. hello intro). */
export function useLenisLock(locked: boolean) {
  const lenis = useLenis();

  useEffect(() => {
    if (!lenis) return;
    if (locked) lenis.stop();
    else lenis.start();
    return () => {
      lenis.start();
    };
  }, [lenis, locked]);
}
