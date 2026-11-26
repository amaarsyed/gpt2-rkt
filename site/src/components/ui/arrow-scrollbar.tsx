import { useEffect, useRef, useState } from 'react';
import './arrow-scrollbar.css';

type Bezier = readonly [number, number, number, number];
type Timing = { duration: number; ease: Bezier };

/**
 * The scrollbar's settings, in the shape of a DialKit "Scrollbar" panel
 * (useDialKit('Scrollbar', SCROLLBAR) would drive them live).
 */
export const SCROLLBAR = {
  arrow: { arrowLength: 30, wingSpread: 8, bobAmplitude: 3, bobPeriod: 2, hitPadding: 10 },
  line: { length: 220, dotSpacing: 10 },
  tracking: { maxExtension: 50, extensionFalloff: 0.6, colorFalloff: 0.3, smoothingTau: 0.05, hitPadding: 10 },
  timing: {
    compressed: { duration: 0.15, ease: [0.41, 1, 0.68, 1] },
    extended: { duration: 0.35, ease: [0.65, 0, 0.35, 1] },
    split: { duration: 0.2, ease: [0.33, 1, 0.68, 1] },
    tracking: { duration: 0.2, ease: [0.33, 1, 0.68, 1] },
  } satisfies Record<string, Timing>,
  appearance: { dotColor: '#6e6868', hoverColor: '#ffffff', strokeWidth: 4 },
};

const RIGHT = 40; // the line's distance from the window's right edge
const BOTTOM = 40; // the arrow tip's distance from the bottom

type Seg = [number, number, number, number]; // x1 y1 x2 y2
type Shape = { left: Seg; right: Seg; parts: Seg[] };

/** cubic-bezier(x1, y1, x2, y2) as a function of progress, solved by bisection. */
function bezier([x1, y1, x2, y2]: Bezier) {
  const at = (a: number, b: number, t: number) => 3 * a * t * (1 - t) ** 2 + 3 * b * t * t * (1 - t) + t ** 3;
  return (x: number) => {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    let lo = 0;
    let hi = 1;
    let t = x;
    for (let i = 0; i < 20; i += 1) {
      if (at(x1, x2, t) < x) lo = t;
      else hi = t;
      t = (lo + hi) / 2;
    }
    return at(y1, y2, t);
  };
}

const mix = (a: number, b: number, t: number) => a + (b - a) * t;

function hex(color: string): [number, number, number] {
  const h = color.replace('#', '');
  const full = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
  const n = Number.parseInt(full, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/**
 * The five shapes the scrollbar moves through, top of page to scrolled:
 * an arrow, the arrow folded to a line, the line grown to full length,
 * the line broken into dots, and the dots tracking the reader.
 */
function shapes(width: number, height: number, length: number, count: number) {
  const x = width - RIGHT;
  const tip = height - BOTTOM;
  const top = height / 2 - length / 2;
  const bottom = height / 2 + length / 2;
  const { arrowLength: arrow, wingSpread: wing } = SCROLLBAR.arrow;
  const run = (from: number, to: number): Seg[] => {
    const step = (to - from) / count;
    return Array.from({ length: count }, (_, i) => [x, from + i * step, x, from + (i + 1) * step]);
  };
  const step = length / count;
  const dots: Seg[] = Array.from({ length: count }, (_, i) => {
    const c = top + (i + 0.5) * step;
    return [x, c - 0.005, x, c + 0.005];
  });
  const shaft = run(tip - arrow, tip);
  const atTip: Seg = [x, tip, x, tip];
  const lastDot = bottom - step / 2;
  const hidden: Seg = [x, lastDot, x, lastDot];
  const atBottom: Seg = [x, bottom, x, bottom];
  const list: Shape[] = [
    { left: [x - wing, tip - wing, x, tip], right: [x + wing, tip - wing, x, tip], parts: shaft },
    { left: atTip, right: atTip, parts: shaft },
    { left: atBottom, right: atBottom, parts: run(top, bottom) },
    { left: hidden, right: hidden, parts: dots },
    { left: hidden, right: hidden, parts: dots },
  ];
  return { list, x, top, step, tip };
}

/**
 * A scrollbar that is an arrow at the top of the page and a column of dots once you scroll.
 * Each dot stands for an equal stretch of the page; the dots around where you are reach
 * out to the left, a hovered dot lights its neighbours, and a click goes there. The arrow
 * bobs and scrolls one screen down. Mouse and trackpad screens only (the CSS hides it,
 * and keeps the native scrollbar, on touch).
 */
export function ArrowScrollbar({
  length,
  onSeek,
}: {
  /** The dot column's length in px, or a function of the window height. Default 220. */
  length?: number | ((windowHeight: number) => number);
  /** How to scroll to a position. Defaults to window.scrollTo; pass a smooth-scroll library's own. */
  onSeek?: (top: number) => void;
}) {
  const svg = useRef<SVGSVGElement | null>(null);
  const seekRef = useRef(onSeek);
  seekRef.current = onSeek;
  const lengthRef = useRef(length);
  lengthRef.current = length;
  const [count, setCount] = useState(() => Math.max(2, Math.round(SCROLLBAR.line.length / SCROLLBAR.line.dotSpacing)));

  useEffect(() => {
    const el = svg.current;
    if (!el) return;
    const lines = [...el.querySelectorAll<SVGLineElement>('line')];
    const [leftWing, rightWing, ...parts] = lines;
    const hits = [...el.querySelectorAll<SVGRectElement>('.asb-hit')];
    const arrowHit = el.querySelector<SVGRectElement>('.asb-arrow-hit');
    if (!leftWing || !rightWing || !arrowHit) return;

    const { tracking, arrow, appearance } = SCROLLBAR;
    const timings = [SCROLLBAR.timing.compressed, SCROLLBAR.timing.extended, SCROLLBAR.timing.split, SCROLLBAR.timing.tracking]
      .map((t) => ({ duration: t.duration, ease: bezier(t.ease) }));
    const still = window.matchMedia('(prefers-reduced-motion: reduce)');
    const dotRGB = hex(appearance.dotColor);

    let geo = shapes(1, 1, 1, count);
    let room = 0;
    const measure = () => {
      const h = window.innerHeight;
      const want = lengthRef.current;
      const len = typeof want === 'function' ? want(h) : want ?? SCROLLBAR.line.length;
      const next = Math.max(2, Math.round(len / SCROLLBAR.line.dotSpacing));
      if (next !== count) { setCount(next); return false; }
      geo = shapes(document.documentElement.clientWidth, h, len, count);
      room = document.documentElement.scrollHeight - h;
      // Hit strips: each dot's band of the line, reaching as far left as a dot can.
      const pad = tracking.hitPadding;
      hits.forEach((rect, i) => {
        rect.setAttribute('x', String(geo.x - tracking.maxExtension - pad));
        rect.setAttribute('y', String(geo.top + i * geo.step));
        rect.setAttribute('width', String(tracking.maxExtension + pad * 2));
        rect.setAttribute('height', String(geo.step));
      });
      const ap = arrow.hitPadding;
      arrowHit.setAttribute('x', String(geo.x - arrow.wingSpread - ap));
      arrowHit.setAttribute('y', String(geo.tip - arrow.arrowLength - ap - arrow.bobAmplitude));
      arrowHit.setAttribute('width', String((arrow.wingSpread + ap) * 2));
      arrowHit.setAttribute('height', String(arrow.arrowLength + ap * 2 + arrow.bobAmplitude * 2));
      return true;
    };
    if (!measure()) return;

    const fraction = () => (room > 0 ? Math.min(1, Math.max(0, window.scrollY / room)) : 0);
    const focusDot = () => Math.round(fraction() * (count - 1));
    const go = (top: number) => {
      const y = Math.min(room, Math.max(0, top));
      if (seekRef.current) seekRef.current(y);
      else window.scrollTo({ top: y, behavior: still.matches ? 'auto' : 'smooth' });
    };

    // t runs 0 (arrow) to 4 (tracking dots), one unit per shape change.
    const LAST = timings.length;
    let t = window.scrollY > 0 ? LAST : 0;
    let target = t;
    let bob = 0;
    let last = 0;
    let frame = 0;
    let hovered: number | null = null;
    let arrowHover = false;
    const reach = new Float64Array(count);

    const inkFor = (factor: number) => {
      // On a light page the configured white would vanish: light up toward the page's ink.
      const fg = getComputedStyle(el).getPropertyValue('--fg').trim();
      const glow = fg.startsWith('#') && hex(fg).reduce((a, b) => a + b, 0) < 384 ? hex(fg) : hex(appearance.hoverColor);
      return `rgb(${dotRGB.map((c, k) => Math.round(mix(c, glow[k] ?? c, factor))).join(', ')})`;
    };

    const paintColors = () => {
      lines.forEach((line, i) => {
        if (target === 0) {
          line.style.stroke = arrowHover ? inkFor(1) : '';
          return;
        }
        const dot = i - 2;
        line.style.stroke = hovered === null || dot < 0 ? '' : inkFor(tracking.colorFalloff ** Math.abs(dot - hovered));
      });
    };

    const set = (line: SVGLineElement, a: Seg, b: Seg, k: number, left = 0, dy = 0) => {
      line.setAttribute('x1', String(mix(a[0], b[0], k) - left));
      line.setAttribute('y1', String(mix(a[1], b[1], k) + dy));
      line.setAttribute('x2', String(mix(a[2], b[2], k)));
      line.setAttribute('y2', String(mix(a[3], b[3], k) + dy));
    };

    const draw = () => {
      const seg = Math.min(Math.max(Math.floor(t), 0), LAST - 1);
      const from = geo.list[seg]!;
      const to = geo.list[seg + 1]!;
      const k = timings[seg]!.ease(t - seg);
      const reachScale = seg === LAST - 1 ? k : 0;
      const bobScale = seg === 0 && !still.matches ? 1 - k : 0;
      const dy = bobScale * arrow.bobAmplitude * Math.sin((2 * Math.PI * bob) / arrow.bobPeriod);
      set(leftWing, from.left, to.left, k, 0, dy);
      set(rightWing, from.right, to.right, k, 0, dy);
      parts.forEach((line, i) => set(line, from.parts[i]!, to.parts[i]!, k, reachScale * (reach[i] ?? 0), dy));
    };

    const tick = (now: number) => {
      frame = 0;
      const dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
      last = now;
      let busy = false;
      // Shape changes, one segment at a time, each at its own duration.
      if (t !== target) {
        if (still.matches) t = target;
        else {
          const seg = Math.min(Math.max(target > t ? Math.floor(t) : Math.ceil(t) - 1, 0), LAST - 1);
          const step = dt / timings[seg]!.duration;
          t = target > t ? Math.min(target, t + step) : Math.max(target, t - step);
        }
        busy = t !== target;
      }
      // The dots near the reader reach left, smoothed toward their targets.
      const focus = focusDot();
      const alpha = still.matches ? 1 : 1 - Math.exp(-dt / tracking.smoothingTau);
      for (let i = 0; i < count; i += 1) {
        const want = tracking.maxExtension * tracking.extensionFalloff ** Math.abs(i - focus);
        const now2 = (reach[i] ?? 0) + (want - (reach[i] ?? 0)) * alpha;
        if (Math.abs(want - now2) < 0.05) reach[i] = want;
        else { reach[i] = now2; busy = true; }
      }
      // The arrow bobs while it is an arrow.
      if (t < 1 && !still.matches) { bob += dt; busy = true; }
      draw();
      if (busy) frame = requestAnimationFrame(tick);
      else last = 0;
    };
    const wake = () => { if (!frame) frame = requestAnimationFrame(tick); };

    const onScroll = () => {
      const next = window.scrollY > 0 ? LAST : 0;
      if (next !== target) {
        target = next;
        el.dataset.state = next === 0 ? 'idle' : 'tracking';
        if (next !== 0) arrowHover = false;
        else hovered = null;
        paintColors();
      }
      wake();
    };
    const onResize = () => { if (measure()) { draw(); wake(); } };

    const enters = hits.map((rect, i) => {
      const enter = () => { hovered = i; paintColors(); };
      const leave = () => { if (hovered === i) { hovered = null; paintColors(); } };
      const click = () => go((count > 1 ? i / (count - 1) : 1) * room);
      rect.addEventListener('mouseenter', enter);
      rect.addEventListener('mouseleave', leave);
      rect.addEventListener('click', click);
      return () => {
        rect.removeEventListener('mouseenter', enter);
        rect.removeEventListener('mouseleave', leave);
        rect.removeEventListener('click', click);
      };
    });
    const arrowEnter = () => { arrowHover = true; paintColors(); };
    const arrowLeave = () => { arrowHover = false; paintColors(); };
    const arrowClick = () => go(window.scrollY + window.innerHeight);
    arrowHit.addEventListener('mouseenter', arrowEnter);
    arrowHit.addEventListener('mouseleave', arrowLeave);
    arrowHit.addEventListener('click', arrowClick);

    el.dataset.state = target === 0 ? 'idle' : 'tracking';
    const sizes = new ResizeObserver(onResize);
    sizes.observe(document.body);
    window.addEventListener('resize', onResize);
    window.addEventListener('scroll', onScroll, { passive: true });
    draw();
    wake();

    return () => {
      enters.forEach((off) => off());
      arrowHit.removeEventListener('mouseenter', arrowEnter);
      arrowHit.removeEventListener('mouseleave', arrowLeave);
      arrowHit.removeEventListener('click', arrowClick);
      sizes.disconnect();
      window.removeEventListener('resize', onResize);
      window.removeEventListener('scroll', onScroll);
      if (frame) cancelAnimationFrame(frame);
    };
  }, [count]);

  const { appearance } = SCROLLBAR;
  return (
    <svg
      ref={svg}
      className="asb"
      data-state="idle"
      aria-hidden="true"
      style={{ ['--asb-dot' as string]: appearance.dotColor, ['--asb-stroke' as string]: appearance.strokeWidth }}
    >
      <line />
      <line />
      {Array.from({ length: count }, (_, i) => <line key={`l${i}`} />)}
      {Array.from({ length: count }, (_, i) => <rect key={`h${i}`} className="asb-hit" />)}
      <rect className="asb-arrow-hit" />
    </svg>
  );
}
