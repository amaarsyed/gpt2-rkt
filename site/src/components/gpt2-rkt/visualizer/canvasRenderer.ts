import { ease, stillness } from '../figures/shared';
import type { ExecutionState } from './executionState';
import {
  executionGraph,
  moduleIdForStage,
  neighbors,
  nodeById,
  ROLE_VAR,
  type EdgeKind,
  type ExecutionGraph,
  type GraphEdge,
  type GraphNode,
  type NodeRole,
} from './graphModel';

export interface CanvasHooks {
  getState: () => ExecutionState;
  onHover: (id: string | null) => void;
  onSelect: (id: string | null) => void;
  onPositions?: (spots: { id: string; x: number; y: number }[]) => void;
  /** Optional: a node to frame without opening Inspect. The scroll-led write-up uses it. */
  getFocus?: () => string | null;
  /** Optional: a node to ring in the accent, such as the .rkt file whose code is showing. */
  getLit?: () => string | null;
}

interface LiveNode {
  node: GraphNode;
  x: number;
  y: number;
}

interface Particle {
  edge: number;
  t: number;
  speed: number;
}

interface Camera {
  x: number;
  y: number;
  k: number;
}

const BLOCK_IDS = new Set([
  'block-ln1',
  'block-qkv',
  'kv-write',
  'attend',
  'block-proj-res',
  'block-ln2',
  'mlp-gelu',
  'block-mlp-res',
]);

export interface ExecutionCanvas {
  destroy: () => void;
  fit: () => void;
  wake: () => void;
}

export function createExecutionCanvas(canvas: HTMLCanvasElement, hooks: CanvasHooks): ExecutionCanvas {
  const graph = executionGraph();
  const live: LiveNode[] = graph.nodes.map((node) => ({ node, x: node.explore.x, y: node.explore.y }));
  const particles: Particle[] = [];
  const camera: Camera = { x: graph.width / 2, y: graph.height / 2, k: 0.82 };
  const target: Camera = { ...camera };
  let modeBlend = 0;
  let width = 0;
  let height = 0;
  let dpr = 1;
  let raf = 0;
  let last = performance.now();
  let spawn = 0;
  let hoverId: string | null = null;
  let frameKey = '';
  let heldMode: ExecutionState['mode'] | '' = '';
  let wasGuided = false;
  let dragging = false;
  let moved = false;
  let pointer = { x: 0, y: 0, cx: 0, cy: 0 };
  let visible = false;
  let queued = false;
  let ready = false;
  const paints = canvas as HTMLCanvasElement & { paintCount: number };
  paints.paintCount = 0;

  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return { destroy: () => undefined, fit: () => undefined, wake: () => undefined };
  }
  const context = ctx;
  let posted = 0;
  // Colours and theme are read from CSS once and kept until the theme or size changes,
  // instead of a getComputedStyle call on every frame.
  let palette: Palette | null = null;
  let darkTheme = true;
  const colorsNow = () => {
    if (!palette) {
      palette = readColors(canvas);
      const themeNode = canvas.closest('[data-theme]') ?? document.documentElement;
      darkTheme = themeNode.getAttribute('data-theme') !== 'light';
    }
    return palette;
  };

  const resize = () => {
    const rect = canvas.getBoundingClientRect();
    width = Math.max(1, rect.width);
    height = Math.max(1, rect.height);
    // Sharp on retina, but never more than ~3.5M device pixels: a full-height canvas at 2x
    // is cleared and redrawn every animated frame, which weaker GPUs can't keep up with.
    dpr = Math.min(window.devicePixelRatio || 1, 2, Math.sqrt(3_500_000 / Math.max(1, width * height)));
    palette = null;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    if (width > 40) restage(hooks.getState(), camera.k === 0.82 || stillness());
    if (ready) wake();
  };

  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
  resize();

  const screenToWorld = (sx: number, sy: number) => ({
    x: (sx - width / 2) / camera.k + camera.x,
    y: (sy - height / 2) / camera.k + camera.y,
  });

  const hit = (sx: number, sy: number): string | null => {
    const world = screenToWorld(sx, sy);
    let best: string | null = null;
    let bestD = Infinity;
    for (const item of live) {
      const pad = (item.node.radius + 8) / camera.k;
      const d = Math.hypot(world.x - item.x, world.y - item.y);
      if (d <= pad && d < bestD) {
        best = item.node.id;
        bestD = d;
      }
    }
    return best;
  };

  const local = (event: PointerEvent | WheelEvent) => {
    const rect = canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  };

  const onPointerDown = (event: PointerEvent) => {
    if (event.button !== 0) return;
    canvas.focus({ preventScroll: true });
    const point = local(event);
    dragging = true;
    moved = false;
    pointer = { x: point.x, y: point.y, cx: camera.x, cy: camera.y };
    canvas.setPointerCapture(event.pointerId);
    wake();
  };

  const onPointerMove = (event: PointerEvent) => {
    const point = local(event);
    if (dragging) {
      const dx = point.x - pointer.x;
      const dy = point.y - pointer.y;
      if (dx * dx + dy * dy > 16) moved = true;
      if (moved) {
        camera.x = pointer.cx - dx / camera.k;
        camera.y = pointer.cy - dy / camera.k;
        target.x = camera.x;
        target.y = camera.y;
      }
      wake();
      return;
    }
    const next = hit(point.x, point.y);
    if (next !== hoverId) {
      hoverId = next;
      hooks.onHover(next);
      wake();
    }
    canvas.style.cursor = next ? 'pointer' : 'grab';
  };

  const onPointerUp = (event: PointerEvent) => {
    if (!dragging) return;
    dragging = false;
    const point = local(event);
    if (!moved) hooks.onSelect(hit(point.x, point.y));
    canvas.style.cursor = hit(point.x, point.y) ? 'pointer' : 'grab';
    wake();
  };

  const onPointerLeave = () => {
    if (dragging) return;
    if (hoverId !== null) {
      hoverId = null;
      hooks.onHover(null);
    }
  };

  const onWheel = (event: WheelEvent) => {
    const armed = document.activeElement === canvas || event.ctrlKey || event.metaKey;
    if (!armed) return;
    event.preventDefault();
    event.stopPropagation();
    const point = local(event);
    const world = screenToWorld(point.x, point.y);
    const dy = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY;
    const next = clamp(camera.k * Math.exp(-dy * 0.00115), 0.42, 2.5);
    camera.k = next;
    camera.x = world.x - (point.x - width / 2) / next;
    camera.y = world.y - (point.y - height / 2) / next;
    target.k = camera.k;
    target.x = camera.x;
    target.y = camera.y;
    wake();
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerUp);
  canvas.addEventListener('pointerleave', onPointerLeave);
  canvas.addEventListener('wheel', onWheel, { passive: false });

  function frameFor(state: ExecutionState): Camera {
    const placed = graph.nodes.map((node) => {
      const spot = state.mode === 'execute' ? node.execute : node.explore;
      return { node, x: spot.x, y: spot.y };
    });
    const guided = hooks.getFocus?.() ?? null;
    if (guided) {
      const focus = placed.find((item) => item.node.id === guided)
        ?? placed.find((item) => item.node.stageId === state.activeStageId)
        ?? placed[0];
      const ids = new Set(focus ? neighbors(graph, focus.node.id) : []);
      if (focus && BLOCK_IDS.has(focus.node.id)) {
        for (const id of BLOCK_IDS) ids.add(id);
      }
      const group = placed.filter((item) => ids.has(item.node.id));
      const core = group.length ? group : placed;
      const floor = core.reduce((max, item) => Math.max(max, item.y), 0);
      const seen = new Set(core.map((item) => item.node.id));
      const framed = [...core];
      const narrow = width < 560;
      if (!narrow) {
        for (const item of placed) {
          if (seen.has(item.node.id) || item.node.role === 'module') continue;
          if (item.y <= floor + item.node.radius) framed.push(item);
        }
      }
      const shown = narrow ? framed.filter((item) => item.node.role !== 'module') : framed;
      const box = shown.length ? shown : framed;
      const inBlock = box.some((item) => BLOCK_IDS.has(item.node.id));
      const padX = narrow ? 36 : inBlock ? 72 : 64;
      const padTop = narrow ? 8 : inBlock ? 56 : 48;
      const padBottom = narrow ? 16 : inBlock ? 88 : 56;
      let minX = Infinity;
      let maxX = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;
      for (const item of box) {
        minX = Math.min(minX, item.x - item.node.radius);
        maxX = Math.max(maxX, item.x + item.node.radius);
        minY = Math.min(minY, item.y - item.node.radius);
        maxY = Math.max(maxY, item.y + item.node.radius);
      }
      const left = minX - padX;
      const right = maxX + padX;
      const top = minY - padTop;
      const bottom = maxY + padBottom;
      const view = freeRect();
      const boxW = Math.max(right - left, 1);
      const boxH = Math.max(bottom - top, 1);
      const fit = Math.min(view.width / boxW, view.height / boxH);
      let k = clamp(fit, Math.min(0.42, fit), 2.2);
      // Scroll-led framing never zooms past the graph's full width, so no node is cut off at the sides.
      if (guided) {
        let allMinX = Infinity;
        let allMaxX = -Infinity;
        for (const item of placed) {
          allMinX = Math.min(allMinX, item.x - item.node.radius - 40);
          allMaxX = Math.max(allMaxX, item.x + item.node.radius + 40);
        }
        k = Math.min(k, (view.width - 24) / Math.max(allMaxX - allMinX, 1));
      }
      const viewCx = view.x + view.width / 2;
      const viewCy = view.y + view.height / 2;
      let cx = (left + right) / 2;
      let cy = (top + bottom) / 2;
      // Scroll-led framing only: when the whole graph fits at this zoom, centre it, so the space
      // above and below (or either side) stays even instead of following the focused group.
      if (guided) {
        let allMinX = Infinity;
        let allMaxX = -Infinity;
        let allMinY = Infinity;
        let allMaxY = -Infinity;
        for (const item of placed) {
          allMinX = Math.min(allMinX, item.x - item.node.radius);
          allMaxX = Math.max(allMaxX, item.x + item.node.radius);
          allMinY = Math.min(allMinY, item.y - item.node.radius);
          allMaxY = Math.max(allMaxY, item.y + item.node.radius + 18);
        }
        if ((allMaxY - allMinY) * k <= view.height - 32) cy = (allMinY + allMaxY) / 2;
        if ((allMaxX - allMinX) * k <= view.width - 32) cx = (allMinX + allMaxX) / 2;
      }
      return {
        x: cx - (viewCx - width / 2) / k,
        y: cy - (viewCy - height / 2) / k,
        k,
      };
    }
    const padX = 64;
    const padY = 40;
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const item of placed) {
      minX = Math.min(minX, item.x - item.node.radius);
      maxX = Math.max(maxX, item.x + item.node.radius);
      minY = Math.min(minY, item.y - item.node.radius);
      maxY = Math.max(maxY, item.y + item.node.radius);
    }
    const left = minX - padX;
    const right = maxX + padX;
    const top = minY - padY;
    const bottom = maxY + padY;
    const boxW = Math.max(right - left, 1);
    const boxH = Math.max(bottom - top, 1);
    const insetX = 16;
    const insetBottom = 8;
    const fit = Math.min((width - insetX * 2) / boxW, Math.max(height - insetBottom, 120) / boxH);
    const k = clamp(fit, Math.min(0.42, fit), 2.5);
    const cx = (left + right) / 2;
    const cy = (top + bottom) / 2;
    const centerY = Math.max(height - insetBottom, 120) / 2;
    return { x: cx, y: cy - (centerY - height / 2) / k, k };
  }

  function freeRect() {
    return { x: 0, y: 0, width, height };
  }

  function restage(state: ExecutionState, snap: boolean) {
    const guided = hooks.getFocus?.() ?? null;
    const focus = guided ? `g:${guided}` : '-';
    const sized = `${Math.round(width)}|${Math.round(height)}`;
    const key = `${state.mode}|${focus}|${sized}`;
    if (key === frameKey) return;
    // While a node's card is open the graph holds still: only a resize, a mode change or the
    // Fit button moves the camera, never the click that opened the card.
    if (state.inspect && frameKey.startsWith(`${state.mode}|`) && frameKey.endsWith(`|${sized}`)) return;
    frameKey = key;
    const next = frameFor(state);
    target.x = next.x;
    target.y = next.y;
    target.k = next.k;
    const modeChanged = heldMode !== '' && heldMode !== state.mode;
    heldMode = state.mode;
    const glide = Boolean(guided) || wasGuided;
    wasGuided = Boolean(guided);
    if (snap || (!modeChanged && !glide)) {
      camera.x = next.x;
      camera.y = next.y;
      camera.k = next.k;
    }
    if (snap) {
      modeBlend = state.mode === 'execute' ? 1 : 0;
      for (const item of live) {
        const dest = nodeDest(state, item.node);
        item.x = dest.x;
        item.y = dest.y;
      }
    }
  }

  function fit() {
    frameKey = '';
    restage(hooks.getState(), stillness());
    wake();
  }

  function nodeDest(state: ExecutionState, node: GraphNode) {
    return state.mode === 'execute' ? node.execute : node.explore;
  }

  function settled(state: ExecutionState) {
    const blend = state.mode === 'execute' ? 1 : 0;
    if (Math.abs(modeBlend - blend) > 0.012) return false;
    if (Math.abs(camera.x - target.x) > 1.25) return false;
    if (Math.abs(camera.y - target.y) > 1.25) return false;
    if (Math.abs(camera.k - target.k) > 0.012) return false;
    for (const item of live) {
      const dest = nodeDest(state, item.node);
      if (Math.abs(item.x - dest.x) > 1.25 || Math.abs(item.y - dest.y) > 1.25) return false;
    }
    return true;
  }

  function snapPose(state: ExecutionState) {
    modeBlend = state.mode === 'execute' ? 1 : 0;
    camera.x = target.x;
    camera.y = target.y;
    camera.k = target.k;
    for (const item of live) {
      const dest = nodeDest(state, item.node);
      item.x = dest.x;
      item.y = dest.y;
    }
  }

  function shouldContinue(state: ExecutionState, calm: boolean) {
    if (!visible) return false;
    if (dragging) return true;
    if (calm) return false;
    if (state.playing && state.mode === 'execute') return true;
    return !settled(state);
  }

  function stop() {
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
    queued = false;
  }

  function wake() {
    if (!visible || queued) return;
    queued = true;
    raf = requestAnimationFrame(frame);
  }

  function frame(now: number) {
    queued = false;
    raf = 0;
    if (!visible) return;
    const dt = Math.min(0.05, Math.max(0, (now - last) / 1000));
    last = now;
    const calm = stillness();
    const state = hooks.getState();
    restage(state, calm);
    if (calm) {
      modeBlend = state.mode === 'execute' ? 1 : 0;
      camera.x = target.x;
      camera.y = target.y;
      camera.k = target.k;
      for (const item of live) {
        const dest = nodeDest(state, item.node);
        item.x = dest.x;
        item.y = dest.y;
      }
      particles.length = 0;
    } else {
      modeBlend = ease(modeBlend, state.mode === 'execute' ? 1 : 0, dt, 0.42, false);
      camera.x = ease(camera.x, target.x, dt, 0.28, false);
      camera.y = ease(camera.y, target.y, dt, 0.28, false);
      camera.k = ease(camera.k, target.k, dt, 0.28, false);
      for (const item of live) {
        const dest = nodeDest(state, item.node);
        item.x = ease(item.x, dest.x, dt, 0.34, false);
        item.y = ease(item.y, dest.y, dt, 0.34, false);
      }
      if (settled(state)) snapPose(state);
    }
    stepParticles(state, calm, dt, now);
    paint(context, state, calm, now);
    paints.paintCount += 1;
    if (shouldContinue(state, calm)) wake();
  }

  function stepParticles(state: ExecutionState, calm: boolean, dt: number, now: number) {
    if (calm || !state.playing || state.mode !== 'execute' || modeBlend < 0.72) {
      particles.length = 0;
      return;
    }
    const hot = hotEdges(state);
    const cap = 22;
    const every = 0.09;
    spawn += dt;
    if (spawn > every && particles.length < cap && hot.length) {
      spawn = 0;
      const edge = hot[Math.floor(now / 90) % hot.length] ?? hot[0];
      if (edge !== undefined) particles.push({ edge, t: 0, speed: 0.55 + (edge % 5) * 0.06 });
    }
    for (let i = particles.length - 1; i >= 0; i -= 1) {
      const particle = particles[i];
      if (!particle) continue;
      particle.t += particle.speed * dt;
      if (particle.t > 1) particles.splice(i, 1);
    }
  }

  function hotEdges(state: ExecutionState): number[] {
    const ids: number[] = [];
    graph.edges.forEach((edge, index) => {
      if (edge.kind === 'module') return;
      const targetNode = live.find((item) => item.node.id === edge.target);
      const sourceNode = live.find((item) => item.node.id === edge.source);
      const stage = state.activeStageId;
      if (targetNode?.node.stageId === stage || sourceNode?.node.stageId === stage) ids.push(index);
    });
    return ids;
  }

  function paint(context: CanvasRenderingContext2D, state: ExecutionState, calm: boolean, now: number) {
    const colors = colorsNow();
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);
    context.fillStyle = colors.bg;
    context.fillRect(0, 0, width, height);

    context.setTransform(
      dpr * camera.k,
      0,
      0,
      dpr * camera.k,
      dpr * (width / 2 - camera.x * camera.k),
      dpr * (height / 2 - camera.y * camera.k),
    );

    const focusId = state.hoveredNodeId;
    const hot = focusId ? neighbors(graph, focusId) : null;
    const unit = 1 / camera.k;

    drawBlockFrame(context, state, colors, unit);
    graph.edges.forEach((edge) => drawEdge(context, edge, state, colors, hot, calm, now, unit));
    for (const particle of particles) drawParticle(context, particle, colors, unit);
    for (const item of live) drawNode(context, item, state, colors, hot, calm, now, unit);
    const litId = hooks.getLit?.() ?? null;
    const lit = litId ? live.find((item) => item.node.id === litId) : null;
    if (lit) {
      context.beginPath();
      context.arc(lit.x, lit.y, lit.node.radius + 7 * unit, 0, Math.PI * 2);
      context.strokeStyle = colors.accent;
      context.lineWidth = 2 * unit;
      context.stroke();
    }
    if (hooks.onPositions && (calm || now - posted > 120)) {
      posted = now;
      hooks.onPositions(live.map((item) => ({
        id: item.node.id,
        x: (item.x - camera.x) * camera.k + width / 2,
        y: (item.y - camera.y) * camera.k + height / 2,
      })));
    }
  }

  function drawBlockFrame(context: CanvasRenderingContext2D, state: ExecutionState, colors: Palette, unit: number) {
    if (modeBlend < 0.08) return;
    const members = live.filter((item) => BLOCK_IDS.has(item.node.id));
    if (!members.length) return;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const item of members) {
      minX = Math.min(minX, item.x);
      minY = Math.min(minY, item.y);
      maxX = Math.max(maxX, item.x);
      maxY = Math.max(maxY, item.y);
    }
    const padX = 58;
    const padTop = 42;
    // Labels sit ~36px under the node centers. 86px keeps Proj+Res, MLP/GELU,
    // and MLP+Res inside the dashed lens, with room for the residual arc below them.
    const padBottom = 86;
    context.save();
    context.globalAlpha = modeBlend * 0.45;
    context.strokeStyle = colors.accent;
    context.lineWidth = 1 * unit;
    context.setLineDash([4 * unit, 5 * unit]);
    roundRect(
      context,
      minX - padX,
      minY - padTop,
      maxX - minX + padX * 2,
      maxY - minY + padTop + padBottom,
      16,
    );
    context.stroke();
    context.setLineDash([]);
    context.font = screenFont(500, 10, camera.k);
    context.textAlign = 'left';
    context.textBaseline = 'bottom';
    const lensLabel = `L${state.activeLayer}`;
    const lensX = minX - padX + 8;
    const lensY = minY - padTop - 6;
    context.lineJoin = 'round';
    context.lineWidth = 3 * unit;
    context.strokeStyle = colors.bg;
    context.strokeText(lensLabel, lensX, lensY);
    context.fillStyle = colors.accent;
    context.fillText(lensLabel, lensX, lensY);
    context.restore();
  }

  function drawEdge(
    context: CanvasRenderingContext2D,
    edge: ExecutionGraph['edges'][number],
    state: ExecutionState,
    colors: Palette,
    hot: Set<string> | null,
    calm: boolean,
    now: number,
    unit: number,
  ) {
    const source = live.find((item) => item.node.id === edge.source);
    const targetNode = live.find((item) => item.node.id === edge.target);
    if (!source || !targetNode) return;
    const involved = !hot || hot.has(edge.source) || hot.has(edge.target);
    let alpha = involved ? 0.85 : 0.42;
    if (edge.kind === 'module') alpha *= 1 - modeBlend * 0.55;
    if (alpha < 0.02) return;
    const active = edge.kind !== 'module' && (
      targetNode.node.stageId === state.activeStageId || source.node.stageId === state.activeStageId
    );
    const curve = bend(source, targetNode, bendAmount(edge, modeBlend));
    context.save();
    context.globalAlpha = alpha;
    context.strokeStyle = edgeColor(edge.kind, targetNode.node.role, colors);
    context.lineWidth = (active ? 2.1 : edge.kind === 'residual' ? 1.5 : 1.15) * unit;
    context.setLineDash(edge.kind === 'residual' ? [7 * unit, 5 * unit] : []);
    if (active && !calm && modeBlend > 0.5) context.lineDashOffset = -((now / 28) % 24) * unit;
    context.beginPath();
    context.moveTo(source.x, source.y);
    context.quadraticCurveTo(curve.x, curve.y, targetNode.x, targetNode.y);
    context.stroke();
    context.restore();
  }

  function drawParticle(context: CanvasRenderingContext2D, particle: Particle, colors: Palette, unit: number) {
    const edge = graph.edges[particle.edge];
    if (!edge) return;
    const source = live.find((item) => item.node.id === edge.source);
    const targetNode = live.find((item) => item.node.id === edge.target);
    if (!source || !targetNode) return;
    const curve = bend(source, targetNode, bendAmount(edge, modeBlend));
    const point = quadratic(source, curve, targetNode, particle.t);
    context.save();
    context.fillStyle = edge.kind === 'residual' ? colors.accent : edge.kind === 'memory' ? colors.red : colors.fg;
    context.globalAlpha = 0.9;
    context.beginPath();
    context.arc(point.x, point.y, 3.3 * unit, 0, Math.PI * 2);
    context.fill();
    context.restore();
  }

  function drawNode(
    context: CanvasRenderingContext2D,
    item: LiveNode,
    state: ExecutionState,
    colors: Palette,
    hot: Set<string> | null,
    calm: boolean,
    now: number,
    unit: number,
  ) {
    const { node } = item;
    const breath = calm || modeBlend > 0.35 ? 0 : Math.sin(now / 900 + node.explore.x * 0.01) * 2.4 * (1 - modeBlend);
    const x = item.x;
    const y = item.y + breath;
    const dim = hot && !hot.has(node.id);
    const dark = darkTheme;
    const selectedNode = nodeById(graph, state.selectedNodeId);
    const selected = state.selectedNodeId === node.id;
    const active = modeBlend > 0.55 && node.stageId === state.activeStageId;
    const linked = node.role === 'module' && modeBlend > 0.5 && (
      node.id === moduleIdForStage(state.activeStageId)
      || node.id === moduleIdForStage(selectedNode?.stageId ?? null)
      || (selectedNode?.role === 'module' && selectedNode.id === node.id)
    );
    const color = roleColor(node.role, colors);
    const radius = node.radius * (node.role === 'module' ? 1 - modeBlend * 0.12 : 1);
    const inactive = dark && modeBlend > 0.6 && !active && !selected && !linked;
    context.save();
    context.globalAlpha = dim ? 0.42 : inactive ? 0.45 : 1;
    if (selected) {
      context.strokeStyle = colors.accent;
      context.lineWidth = 1.4 * unit;
      context.setLineDash([3.5 * unit, 3 * unit]);
      context.beginPath();
      context.arc(x, y, radius + 7 * unit, 0, Math.PI * 2);
      context.stroke();
      context.globalAlpha = dim ? 0.42 : 0.45;
      context.beginPath();
      context.arc(x, y, radius + 12 * unit, 0, Math.PI * 2);
      context.stroke();
      context.setLineDash([]);
      context.globalAlpha = dim ? 0.42 : inactive ? 0.45 : 1;
    }
    if (active) {
      context.strokeStyle = color;
      context.globalAlpha = dim ? 0.42 : 0.35;
      context.lineWidth = 6 * unit;
      context.beginPath();
      context.arc(x, y, radius + 3 * unit, 0, Math.PI * 2);
      context.stroke();
      context.globalAlpha = dim ? 0.42 : inactive ? 0.45 : 1;
    }
    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    if (node.role === 'module') {
      if (linked) {
        context.fillStyle = colors.accent;
        context.globalAlpha = 0.45;
        context.fill();
        context.globalAlpha = dim ? 0.42 : inactive ? 0.45 : 1;
        context.strokeStyle = colors.accent;
        context.lineWidth = 2.2 * unit;
      } else {
        context.strokeStyle = dark ? colors.fg : color;
        context.lineWidth = 1.5 * unit;
      }
      context.stroke();
    } else {
      context.fillStyle = color;
      context.fill();
      context.fillStyle = colors.bg;
      context.beginPath();
      context.arc(x, y, radius * 0.38, 0, Math.PI * 2);
      context.fill();
    }
    // On a phone-width canvas the file labels crowd the stage labels, so a file
    // keeps its dot and only names itself while it is the one in play.
    const narrow = width < 520;
    if (narrow && node.role === 'module' && !selected && !linked && !(hot && hot.has(node.id))) {
      context.restore();
      return;
    }
    const label = node.label;
    const labelX = x;
    const labelY = y + radius + 6 * unit;
    context.globalAlpha = dim ? 0.88 : 1;
    context.font = screenFont(node.role === 'module' ? 560 : 620, node.role === 'module' ? 12 : 14, camera.k, narrow ? 10.5 : 13);
    context.textAlign = 'center';
    context.textBaseline = 'top';
    context.lineJoin = 'round';
    context.miterLimit = 2;
    context.lineWidth = Math.min(3.2 * unit, 3);
    context.strokeStyle = colors.bg;
    context.strokeText(label, labelX, labelY);
    context.fillStyle = colors.fg;
    context.fillText(label, labelX, labelY);
    context.restore();
  }

  const figure = canvas.closest('[data-viz-root]') ?? canvas;
  const io = new IntersectionObserver((entries) => {
    const entry = entries[entries.length - 1];
    visible = Boolean(entry?.isIntersecting);
    if (!visible) stop();
    else wake();
  });
  io.observe(figure);

  const onTheme = () => { palette = null; wake(); };
  const theme = new MutationObserver(onTheme);
  const themeRoot = canvas.closest('.wu-root');
  if (themeRoot) theme.observe(themeRoot, { attributes: true, attributeFilter: ['data-theme'] });
  theme.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'style'] });

  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  motion.addEventListener('change', onTheme);

  ready = true;
  resize();

  return {
    fit,
    wake,
    destroy: () => {
      stop();
      ro.disconnect();
      io.disconnect();
      theme.disconnect();
      motion.removeEventListener('change', onTheme);
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('pointercancel', onPointerUp);
      canvas.removeEventListener('pointerleave', onPointerLeave);
      canvas.removeEventListener('wheel', onWheel);
    },
  };
}

const UI = '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif';

function screenFont(weight: number, worldPx: number, zoom: number, floor = 11) {
  const safe = Math.max(zoom, 0.2);
  const size = Math.max(floor, worldPx * safe) / safe;
  return `${weight} ${size.toFixed(2)}px ${UI}`;
}

interface Palette {
  bg: string;
  fg: string;
  muted: string;
  accent: string;
  red: string;
  green: string;
  blue: string;
  anatomyMuted: string;
}

function readColors(canvas: HTMLCanvasElement): Palette {
  const style = getComputedStyle(canvas);
  return {
    bg: pick(style, '--bg', '#121210'),
    fg: pick(style, '--fg', '#ecebe6'),
    muted: pick(style, '--anatomy-muted', '#7a7871'),
    accent: pick(style, '--accent', '#f5c400'),
    red: pick(style, '--anatomy-red', '#ff6b6b'),
    green: pick(style, '--anatomy-green', '#6be0a0'),
    blue: pick(style, '--anatomy-blue', '#6b9bff'),
    anatomyMuted: pick(style, '--anatomy-muted', '#7a7871'),
  };
}

function pick(style: CSSStyleDeclaration, name: string, fallback: string): string {
  const value = style.getPropertyValue(name).trim();
  return value || fallback;
}

function roleColor(role: NodeRole, colors: Palette): string {
  const variable = ROLE_VAR[role];
  if (variable === '--accent') return colors.accent;
  if (variable === '--anatomy-blue') return colors.blue;
  if (variable === '--anatomy-green') return colors.green;
  if (variable === '--anatomy-red') return colors.red;
  if (variable === '--fg') return colors.fg;
  return colors.anatomyMuted;
}

function edgeColor(kind: EdgeKind, role: NodeRole, colors: Palette): string {
  if (kind === 'residual') return colors.accent;
  if (kind === 'memory') return colors.red;
  if (kind === 'module') return colors.anatomyMuted;
  return roleColor(role, colors);
}

function bendAmount(edge: GraphEdge, blend: number): number {
  // In execute, drop the LN2 residual under the MLP/GELU label instead of through it.
  if (edge.id === 'r-mlp') return edge.bend + blend * 72;
  return edge.bend;
}

function bend(source: LiveNode, target: LiveNode, amount: number) {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const len = Math.hypot(dx, dy) || 1;
  return {
    x: (source.x + target.x) / 2 + (-dy / len) * amount,
    y: (source.y + target.y) / 2 + (dx / len) * amount,
  };
}

function quadratic(source: LiveNode, control: { x: number; y: number }, target: LiveNode, t: number) {
  const u = 1 - t;
  return {
    x: u * u * source.x + 2 * u * t * control.x + t * t * target.x,
    y: u * u * source.y + 2 * u * t * control.y + t * t * target.y,
  };
}

function roundRect(context: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + w, y, x + w, y + h, r);
  context.arcTo(x + w, y + h, x, y + h, r);
  context.arcTo(x, y + h, x, y, r);
  context.arcTo(x, y, x + w, y, r);
  context.closePath();
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}
