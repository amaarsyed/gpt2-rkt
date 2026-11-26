import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import './dial-scroll.css';

/**
 * A scroll rail in the style of DialKit's sliders (joshpuckett/dialkit): a rounded track
 * on a faint surface, a fill for how far you've read, a thin handle, and hashmarks at
 * each section that light up while you're on it. Drag or click it to scroll; it also
 * takes the keyboard. Replaces the browser scrollbar on mouse and trackpad screens only
 * (dial-scroll.css); touch screens keep their own.
 */
export function DialScroll({
  marks = 'section[id]',
  label = 'Page position',
  className = '',
  onSeek,
}: {
  /** querySelectorAll for the elements whose tops get a hashmark */
  marks?: string;
  label?: string;
  className?: string;
  /** How to jump to a scroll position. Defaults to window.scrollTo; pass a smooth-scroll
      library's own jump (e.g. Lenis's scrollTo with immediate) so the two don't fight. */
  onSeek?: (top: number) => void;
}) {
  const root = useRef<HTMLDivElement | null>(null);
  const track = useRef<HTMLDivElement | null>(null);
  const value = useRef<HTMLSpanElement | null>(null);
  const room = useRef(0);
  const [ticks, setTicks] = useState<number[]>([]);
  const [dragging, setDragging] = useState(false);
  const [scrollable, setScrollable] = useState(true);

  useEffect(() => {
    let frame = 0;
    // Direct to the DOM on scroll: no React render per frame.
    const paint = () => {
      frame = 0;
      const p = room.current > 0 ? Math.min(1, Math.max(0, window.scrollY / room.current)) : 0;
      root.current?.style.setProperty('--p', p.toFixed(4));
      const pct = String(Math.round(p * 100));
      if (value.current) value.current.textContent = pct;
      track.current?.setAttribute('aria-valuenow', pct);
    };
    const measure = () => {
      room.current = document.documentElement.scrollHeight - window.innerHeight;
      setScrollable(room.current > 8);
      const at = [...document.querySelectorAll<HTMLElement>(marks)]
        .map((el) => (el.getBoundingClientRect().top + window.scrollY) / Math.max(1, room.current))
        .filter((t) => t > 0.005 && t < 0.995);
      setTicks((old) => (old.length === at.length && old.every((t, i) => Math.abs(t - (at[i] ?? 0)) < 0.002) ? old : at));
      paint();
    };
    const onScroll = () => { if (!frame) frame = requestAnimationFrame(paint); };
    measure();
    const sizes = new ResizeObserver(measure);
    sizes.observe(document.body);
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', measure);
    return () => {
      sizes.disconnect();
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', measure);
      if (frame) cancelAnimationFrame(frame);
    };
  }, [marks]);

  const go = (top: number) => {
    const y = Math.min(room.current, Math.max(0, top));
    if (onSeek) onSeek(y);
    else window.scrollTo({ top: y, behavior: 'instant' as ScrollBehavior });
  };

  const seek = (clientY: number) => {
    const box = track.current?.getBoundingClientRect();
    if (!box) return;
    go(((clientY - box.top) / box.height) * room.current);
  };

  const onDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(true);
    seek(event.clientY);
  };
  const onMove = (event: PointerEvent<HTMLDivElement>) => { if (dragging) seek(event.clientY); };
  const onUp = (event: PointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    setDragging(false);
  };

  const onKey = (event: KeyboardEvent<HTMLDivElement>) => {
    const page = window.innerHeight * 0.9;
    const by: Record<string, number> = { ArrowDown: 80, ArrowUp: -80, PageDown: page, PageUp: -page };
    if (event.key in by) go(window.scrollY + (by[event.key] ?? 0));
    else if (event.key === 'Home') go(0);
    else if (event.key === 'End') go(room.current);
    else return;
    event.preventDefault();
  };

  return (
    <div ref={root} className={`dial-scroll ${className}`} data-active={dragging ? '' : undefined} hidden={!scrollable}>
      <div
        ref={track}
        className="dial-scroll-track"
        role="scrollbar"
        aria-label={label}
        aria-orientation="vertical"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={0}
        tabIndex={0}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
        onKeyDown={onKey}
      >
        <div className="dial-scroll-marks" aria-hidden="true">
          {ticks.map((t, i) => <i key={i} style={{ top: `${(t * 100).toFixed(2)}%` }} />)}
        </div>
        <div className="dial-scroll-fill" aria-hidden="true" />
        <div className="dial-scroll-handle" aria-hidden="true" />
      </div>
      <span ref={value} className="dial-scroll-value" aria-hidden="true">0</span>
    </div>
  );
}
