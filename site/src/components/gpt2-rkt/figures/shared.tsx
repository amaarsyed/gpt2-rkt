// Parts every bench in this write-up shares: a base plate on feet with a groove,
// corner screws and a label plate. Geometry is computed once per projection.
import {
  corners,
  cylinder,
  depthOf,
  haloOf,
  insetPlan,
  iso,
  lineOnTop,
  pathOf,
  planOutline,
  ring,
  slabOf,
  topTicks,
  type Flat,
  type Plan,
  type Plane,
  type Point3,
  type Projection,
  type SolidPaths,
} from '../../iso/iso-kit';
import { Dots, Line, Solid } from '../../iso/react/draw';

export const BASE_H = 10;
export const FOOT_H = 5;

export const byDepth = (points: Plane[], P: Projection, z = 0) =>
  [...points].sort((a, b) => depthOf([a[0], a[1], z], P) - depthOf([b[0], b[1], z], P));

export const join = (paths: string[]) => paths.filter(Boolean).join('');

export const seg = (a: Point3, b: Point3, P: Projection) => pathOf([iso(a, P), iso(b, P)]);

/** A projection fitted to the given points, with a fixed azimuth. */
export function fit(points: Point3[], width: number, height: number, azimuth: number, pad = 30, elevation = 30): Projection {
  const unit: Projection = { origin: [0, 0], scale: 1, azimuth, elevation };
  const flat = points.map((p) => iso(p, unit));
  const xs = flat.map(([x]) => x);
  const ys = flat.map(([, y]) => y);
  const left = Math.min(...xs);
  const right = Math.max(...xs);
  const up = Math.min(...ys);
  const down = Math.max(...ys);
  const scale = Math.min((width - pad * 2) / (right - left), (height - pad * 2) / (down - up));
  return {
    origin: [width / 2 - ((left + right) / 2) * scale, height / 2 - ((up + down) / 2) * scale],
    scale,
    azimuth,
    elevation,
  };
}

export const planCorners = ({ x, y, w, d }: Plan, z0: number, z1: number): Point3[] => {
  const out: Point3[] = [];
  for (const px of [x, x + w]) for (const py of [y, y + d]) for (const z of [z0, z1]) out.push([px, py, z]);
  return out;
};

export interface BaseArt {
  halo: string;
  feet: SolidPaths[][];
  slab: SolidPaths;
  groove: string;
  screws: Flat[];
  screwRings: string;
  plate: SolidPaths;
  plateLine: string;
  plateRule: string;
  plateScrews: Flat[];
}

/** A heavy bench plate on four feet, top face at z = 0. `label` is where the label plate sits. */
export function baseArt(base: Plan, label: Plan, P: Projection): BaseArt {
  const feet = byDepth(corners(base, 20), P, -BASE_H);
  const screws = corners(base, 9);
  return {
    halo: haloOf(base, -BASE_H - FOOT_H, BASE_H, P),
    feet: feet.map(([x, y]) => [
      cylinder(x, y, 7, -BASE_H - FOOT_H, 1.6, P, 24),
      cylinder(x, y, 5.4, -BASE_H - FOOT_H + 1.6, FOOT_H - 1.6, P, 24),
    ]),
    slab: slabOf(base, -BASE_H, BASE_H, P, 8, 1.6),
    groove: planOutline(insetPlan(base, 4.5), 0, P),
    screws: screws.map(([x, y]) => iso([x, y, 0], P)),
    screwRings: join(screws.map(([x, y]) => ring(x, y, 2.2, 0, P, 16))),
    plate: slabOf(label, 0, 0.8, P, 4),
    plateLine: planOutline(insetPlan(label, 1.6), 0.8, P),
    plateRule: join(
      [0, 1, 2].map((row) =>
        lineOnTop(
          [label.x + 7, label.y + 3.2 + row * ((label.d - 6.4) / 2)],
          [label.x + (row === 1 ? label.w * 0.55 : label.w * 0.78), label.y + 3.2 + row * ((label.d - 6.4) / 2)],
          0.8,
          P,
        ),
      ),
    ),
    plateScrews: [
      iso([label.x + 3.2, label.y + label.d / 2, 0.8], P),
      iso([label.x + label.w - 3.2, label.y + label.d / 2, 0.8], P),
    ],
  };
}

export function BaseView({ art, k }: { art: BaseArt; k: string }) {
  return (
    <g key={k}>
      <path className="iso-halo" d={art.halo} />
      {art.feet.map((parts, i) => (
        <g key={i}>
          <Solid paths={parts[0]} tone="lo" />
          <Solid paths={parts[1]} tone="mid" />
        </g>
      ))}
      <Solid paths={art.slab} tone="mid" />
      <Line d={art.groove} tone="faint" />
      <Line d={art.screwRings} tone="lo" />
      <Dots points={art.screws} size={0.55} tone="mid" />
      <Solid paths={art.plate} tone="lo" crease="none" />
      <Line d={art.plateLine} tone="faint" />
      <Line d={art.plateRule} tone="lo" free />
      <Dots points={art.plateScrews} size={0.45} tone="mid" />
    </g>
  );
}

/** A thin ruler strip lying on the plate along x, with ticks rooted on its far edge. */
export function rulerArt(strip: Plan, step: number, every: number, P: Projection) {
  const t = topTicks(strip.x + 2, strip.x + strip.w - 2, step, every, strip.y + 0.6, 0.6, [2, 3.6], P);
  return { strip: slabOf(strip, 0, 0.6, P, 3), minor: t.minor, major: t.major };
}

export function RulerView({ art }: { art: ReturnType<typeof rulerArt> }) {
  return (
    <g>
      <Solid paths={art.strip} tone="lo" crease="none" />
      <Line d={art.minor} tone="lo" />
      <Line d={art.major} tone="mid" />
    </g>
  );
}

/** After a mouse press, focus the figure so the arrow keys work, without the keyboard focus ring. */
export const focusFromPointer = (svg: SVGSVGElement | null) => {
  if (!svg) return;
  svg.dataset.pointer = '';
  svg.focus({ preventScroll: true });
};

/** A key press (or leaving the figure) brings the focus ring back for keyboard users. */
export const clearPointerFocus = (e: { currentTarget: Element }) => {
  delete (e.currentTarget as SVGSVGElement).dataset.pointer;
};

/** Calm on/off for prefers-reduced-motion, read inside effects only. */
export const stillness = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export const ease = (current: number, target: number, dt: number, seconds: number, calm: boolean) => {
  if (calm) return target;
  const next = current + (target - current) * (1 - Math.exp(-dt / seconds));
  return Math.abs(target - next) < 1e-3 ? target : next;
};

export const fmt = (n: number) => n.toLocaleString('en-US');
