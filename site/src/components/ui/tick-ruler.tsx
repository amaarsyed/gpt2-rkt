import { useEffect, useRef, useState } from 'react';
import './tick-ruler.css';

type Bezier = readonly [number, number, number, number];

/**
 * The ruler's settings, in the shape of a DialKit "Scrollbar" panel
 * (useDialKit('Scrollbar', RULER) would drive them live).
 */
export const RULER = {
  cue: { ticks: 3, gap: 7, bobPeriod: 2, hitPadding: 10 },
  line: { length: 220, dotSpacing: 10, majorEvery: 5 },
  ticks: { minor: 5, major: 12, stroke: 2.5 },
  tracking: { maxExtension: 50, extensionFalloff: 0.6, colorFalloff: 0.3, smoothingTau: 0.05, hitPadding: 10 },
  timing: {
    fold: { duration: 0.15, ease: [0.41, 1, 0.68, 1] as Bezier },
    unfold: { duration: 0.35, ease: [0.65, 0, 0.35, 1] as Bezier },
    readout: { duration: 0.2, ease: [0.33, 1, 0.68, 1] as Bezier },
  },
  appearance: { tickColor: '#6e6868', accentFallback: '#f5c400' },
};

const RIGHT = 22; // where every tick's right end sits, from the window's right edge
const BOTTOM = 40; // the cue's lowest tick, from the bottom

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

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

function rgb(color: string): [number, number, number] | null {
  const c = color.trim();
  if (c.startsWith('#')) {
    const h = c.slice(1);
    const full = h.length === 3 ? h.split('').map((x) => x + x).join('') : h.slice(0, 6);
    const n = Number.parseInt(full, 16);
    return Number.isNaN(n) ? null : [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const m = c.match(/rgba?\(([^)]+)\)/);
  if (!m?.[1]) return null;
  const [r, g, b] = m[1].split(/[ ,/]+/).map(Number);
  return r === undefined || g === undefined || b === undefined ? null : [r, g, b];
}

export type RulerSection = { id: string; label: string };

/**
 * A scrollbar drawn as a ruler. Each tick is an equal stretch of the page, with a longer
 * tick every fifth. The ticks around where you are swell out to the left in the page's
 * accent, with the percentage beside them while you scroll; hovering a tick names the
 * section it lands in, and a click goes there. At the very top it rests as three ticks
 * that ripple, and unfolds from the bottom once you scroll. Mouse and trackpad only.
 */
export function TickRuler({
  length,
  sections = [],
  onSeek,
}: {
  /** The ruler's length in px, or a function of the window height. Default 220. */
  length?: number | ((windowHeight: number) => number);
  /** Sections to name on hover: each id is an element on the page. */
  sections?: RulerSection[];
  /** How to scroll to a position. Defaults to window.scrollTo; pass a smooth-scroll library's own. */
  onSeek?: (top: number) => void;
}) {
  const svg = useRef<SVGSVGElement | null>(null);
  const seekRef = useRef(onSeek);
  seekRef.current = onSeek;
  const lengthRef = useRef(length);
  lengthRef.current = length;
  const sectionsRef = useRef(sections);
  sectionsRef.current = sections;
  const [count, setCount] = useState(() => Math.max(2, Math.round(RULER.line.length / RULER.line.dotSpacing)));

  useEffect(() => {
    const el = svg.current;
    if (!el) return;
    const ticks = [...el.querySelectorAll<SVGLineElement>('.tr-tick')];
    const cue = [...el.querySelectorAll<SVGLineElement>('.tr-cue')];
    const hits = [...el.querySelectorAll<SVGRectElement>('.tr-hit')];
    const cueHit = el.querySelector<SVGRectElement>('.tr-cue-hit');
    const label = el.querySelector<SVGTextElement>('.tr-label');
    if (!cueHit || !label) return;

    const { tracking, ticks: tk, cue: cueCfg, appearance, line } = RULER;
    const unfoldEase = bezier(RULER.timing.unfold.ease);
    const foldEase = bezier(RULER.timing.fold.ease);
    const readEase = bezier(RULER.timing.readout.ease);
    const still = window.matchMedia('(prefers-reduced-motion: reduce)');
    const base = rgb(appearance.tickColor) ?? [110, 104, 104];

    let x = 0;
    let top = 0;
    let step = 0;
    let cueY = 0;
    let room = 0;
    let marks: { at: number; label: string }[] = [];

    const measure = () => {
      const h = window.innerHeight;
      const want = lengthRef.current;
      const len = typeof want === 'function' ? want(h) : want ?? line.length;
      const next = Math.max(2, Math.round(len / line.dotSpacing));
      if (next !== count) { setCount(next); return false; }
      x = document.documentElement.clientWidth - RIGHT;
      top = h / 2 - len / 2;
      step = len / (count - 1);
      cueY = h - BOTTOM;
      room = document.documentElement.scrollHeight - h;
      marks = sectionsRef.current
        .map((s) => {
          const target = document.getElementById(s.id);
          return target ? { at: (target.getBoundingClientRect().top + window.scrollY - h * 0.3) / Math.max(1, room), label: s.label } : null;
        })
        .filter((m): m is { at: number; label: string } => m !== null)
        .sort((a, b) => a.at - b.at);
      const pad = tracking.hitPadding;
      hits.forEach((rect, i) => {
        rect.setAttribute('x', String(x - tk.major - tracking.maxExtension - pad));
        rect.setAttribute('y', String(top + (i - 0.5) * step));
        rect.setAttribute('width', String(tk.major + tracking.maxExtension + pad * 2));
        rect.setAttribute('height', String(step));
      });
      const cp = cueCfg.hitPadding;
      cueHit.setAttribute('x', String(x - tk.major - cp));
      cueHit.setAttribute('y', String(cueY - (cueCfg.ticks - 1) * cueCfg.gap - cp));
      cueHit.setAttribute('width', String(tk.major + cp * 2));
      cueHit.setAttribute('height', String((cueCfg.ticks - 1) * cueCfg.gap + cp * 2));
      return true;
    };
    if (!measure()) return;

    const accent = () => rgb(getComputedStyle(el).getPropertyValue('--accent')) ?? rgb(appearance.accentFallback)!;
    let accentRGB = accent();
    const tint = (k: number) => `rgb(${base.map((c, j) => Math.round(c + ((accentRGB[j] ?? c) - c) * k)).join(', ')})`;

    const fraction = () => (room > 0 ? clamp01(window.scrollY / room) : 0);
    const sectionAt = (f: number) => {
      let name = '';
      for (const m of marks) if (m.at <= f + 0.001) name = m.label;
      return name || (marks[0]?.label ?? '');
    };
    const go = (y: number) => {
      const to = Math.min(room, Math.max(0, y));
      if (seekRef.current) seekRef.current(to);
      else window.scrollTo({ top: to, behavior: still.matches ? 'auto' : 'smooth' });
    };

    // open: 0 = the resting cue, 1 = the full ruler. reach: each tick's swell, smoothed.
    let open = window.scrollY > 0 ? 1 : 0;
    let goal = open;
    let clock = 0;
    let last = 0;
    let frame = 0;
    let hovered: number | null = null;
    let cueHover = false;
    let lastScroll = -10;
    let readout = 0; // 0..1, the label's fade
    const reach = new Float64Array(count);

    const draw = () => {
      // The cue: three ticks that ripple, growing in turn like something being pulled.
      const fold = foldEase(clamp01(open * 3));
      cue.forEach((tick, k) => {
        const wave = still.matches ? 0.5 : 0.5 + 0.5 * Math.sin((2 * Math.PI * clock) / cueCfg.bobPeriod - k * 0.9);
        const len = (tk.minor + (tk.major - tk.minor) * wave) * (1 - fold);
        const y = cueY - k * cueCfg.gap;
        tick.setAttribute('x1', String(x - len));
        tick.setAttribute('x2', String(x));
        tick.setAttribute('y1', String(y));
        tick.setAttribute('y2', String(y));
        tick.style.opacity = String(1 - fold);
        tick.style.stroke = cueHover ? tint(1) : '';
      });
      // The ruler unfolds from its bottom tick up, each tick a little after the one below.
      const spread = 0.9;
      const focus = fraction() * (count - 1);
      ticks.forEach((tick, i) => {
        const fromBottom = (count - 1 - i) / Math.max(1, count - 1);
        const a = unfoldEase(clamp01((open * (1 + spread) - fromBottom * spread)));
        const major = i % line.majorEvery === 0 || i === count - 1;
        const rest = major ? tk.major : tk.minor;
        const len = (rest + (reach[i] ?? 0)) * a;
        const y = top + i * step;
        tick.setAttribute('x1', String(x - len));
        tick.setAttribute('x2', String(x));
        tick.setAttribute('y1', String(y));
        tick.setAttribute('y2', String(y));
        tick.style.opacity = String(a);
        const near = tracking.colorFalloff ** Math.abs(i - focus);
        const lit = hovered === null ? 0 : tracking.colorFalloff ** Math.abs(i - hovered);
        tick.style.stroke = tint(Math.max(near > 0.05 ? near : 0, lit));
      });
      // The readout: the percentage while scrolling, or the hovered tick's section.
      const showing = hovered !== null ? hovered : Math.round(focus);
      const at = showing;
      const len = ((at % line.majorEvery === 0 ? tk.major : tk.minor) + (reach[at] ?? 0));
      label.setAttribute('x', String(x - len - 10));
      label.setAttribute('y', String(top + at * step));
      label.textContent = hovered !== null
        ? sectionAt(hovered / (count - 1))
        : `${Math.round(fraction() * 100)}%`;
      const r = readEase(readout) * open;
      label.style.opacity = String(r);
      label.style.transform = `translateX(${(1 - readEase(readout)) * 6}px)`;
    };

    const tick = (now: number) => {
      frame = 0;
      const dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
      last = now;
      clock += dt;
      let busy = false;
      if (open !== goal) {
        if (still.matches) open = goal;
        else {
          const total = RULER.timing.unfold.duration * 1.9; // the unfold's own time, plus the stagger
          open = goal > open ? Math.min(goal, open + dt / total) : Math.max(goal, open - dt / (RULER.timing.fold.duration * 2));
        }
        busy = true;
      }
      const focus = fraction() * (count - 1);
      const alpha = still.matches ? 1 : 1 - Math.exp(-dt / tracking.smoothingTau);
      for (let i = 0; i < count; i += 1) {
        const want = tracking.maxExtension * tracking.extensionFalloff ** Math.abs(i - focus);
        const cur = reach[i] ?? 0;
        const nxt = cur + (want - cur) * alpha;
        if (Math.abs(want - nxt) < 0.05) reach[i] = want;
        else { reach[i] = nxt; busy = true; }
      }
      // The readout shows while scrolling or hovering, and fades a moment after.
      const wantRead = hovered !== null || now / 1000 - lastScroll < 0.9 ? 1 : 0;
      if (readout !== wantRead) {
        const d = dt / RULER.timing.readout.duration;
        readout = wantRead > readout ? Math.min(1, readout + d) : Math.max(0, readout - d);
        busy = true;
      }
      if (wantRead === 1 && hovered === null) busy = true; // keep watching for the fade-out
      if (open < 1 && !still.matches) busy = true; // the cue ripples while it shows
      draw();
      if (busy) frame = requestAnimationFrame(tick);
      else last = 0;
    };
    const wake = () => { if (!frame) frame = requestAnimationFrame(tick); };

    const onScroll = () => {
      lastScroll = performance.now() / 1000;
      const next = window.scrollY > 0 ? 1 : 0;
      if (next !== goal) {
        goal = next;
        el.dataset.state = next ? 'ruler' : 'cue';
        if (!next) hovered = null;
        else cueHover = false;
      }
      wake();
    };
    const onResize = () => { if (measure()) { draw(); wake(); } };
    const onTheme = () => { accentRGB = accent(); draw(); };

    const offs = hits.map((rect, i) => {
      const enter = () => { hovered = i; wake(); };
      const leave = () => { if (hovered === i) { hovered = null; wake(); } };
      const click = () => go((i / (count - 1)) * room);
      rect.addEventListener('mouseenter', enter);
      rect.addEventListener('mouseleave', leave);
      rect.addEventListener('click', click);
      return () => {
        rect.removeEventListener('mouseenter', enter);
        rect.removeEventListener('mouseleave', leave);
        rect.removeEventListener('click', click);
      };
    });
    const cueEnter = () => { cueHover = true; draw(); };
    const cueLeave = () => { cueHover = false; draw(); };
    const cueClick = () => go(window.scrollY + window.innerHeight);
    cueHit.addEventListener('mouseenter', cueEnter);
    cueHit.addEventListener('mouseleave', cueLeave);
    cueHit.addEventListener('click', cueClick);

    el.dataset.state = goal ? 'ruler' : 'cue';
    const sizes = new ResizeObserver(onResize);
    sizes.observe(document.body);
    const themes = new MutationObserver(onTheme);
    themes.observe(document.documentElement, { attributes: true, subtree: true, attributeFilter: ['data-theme', 'class'] });
    window.addEventListener('resize', onResize);
    window.addEventListener('scroll', onScroll, { passive: true });
    draw();
    wake();

    return () => {
      offs.forEach((off) => off());
      cueHit.removeEventListener('mouseenter', cueEnter);
      cueHit.removeEventListener('mouseleave', cueLeave);
      cueHit.removeEventListener('click', cueClick);
      sizes.disconnect();
      themes.disconnect();
      window.removeEventListener('resize', onResize);
      window.removeEventListener('scroll', onScroll);
      if (frame) cancelAnimationFrame(frame);
    };
  }, [count]);

  return (
    <svg
      ref={svg}
      className="tr"
      data-state="cue"
      aria-hidden="true"
      style={{ ['--tr-tick' as string]: RULER.appearance.tickColor, ['--tr-stroke' as string]: RULER.ticks.stroke }}
    >
      {Array.from({ length: RULER.cue.ticks }, (_, i) => <line key={`c${i}`} className="tr-cue" />)}
      {Array.from({ length: count }, (_, i) => <line key={`t${i}`} className="tr-tick" />)}
      <text className="tr-label" textAnchor="end" dominantBaseline="middle" />
      {Array.from({ length: count }, (_, i) => <rect key={`h${i}`} className="tr-hit" />)}
      <rect className="tr-cue-hit" />
    </svg>
  );
}
