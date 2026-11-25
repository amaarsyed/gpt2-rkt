import { useEffect, useRef } from 'react';
import { attentionWeight } from './attentionMath';
import type { ExecutionTrace } from './traceTypes';

interface Props {
  trace: ExecutionTrace;
  layer: number;
  head: number;
  queryRow: number;
  query: number;
  keyIndex: number;
  phase: 0 | 1 | 2 | 3;
  onPick: (query: number, keyIndex: number) => void;
  onHover: (query: number | null, keyIndex: number | null) => void;
}

const PAD = 14;

function gridLayout(width: number, height: number, count: number, strip: number) {
  const cell = Math.max(4, Math.min((width - PAD * 2) / count, (height - PAD * 2 - strip) / count));
  const grid = cell * count;
  return {
    cell,
    originX: (width - grid) / 2,
    originY: PAD,
    grid,
  };
}

export function AttentionHeatmap({
  trace, layer, head, queryRow, query, keyIndex, phase, onPick, onHover,
}: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const hover = useRef<{ q: number; k: number } | null>(null);
  const n = queryRow + 1;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const context = canvas.getContext('2d');
    if (!context) return;

    const paint = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 3);
      const width = Math.max(1, rect.width);
      const height = Math.max(1, rect.height);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      const style = getComputedStyle(canvas);
      const bg = style.getPropertyValue('--bg').trim() || '#121210';
      const fg = style.getPropertyValue('--fg').trim() || '#ecebe6';
      const blue = style.getPropertyValue('--anatomy-blue').trim() || '#6b9bff';
      const accent = style.getPropertyValue('--accent').trim() || '#f5c400';
      const green = style.getPropertyValue('--anatomy-green').trim() || '#6be0a0';
      const muted = style.getPropertyValue('--anatomy-muted').trim() || '#7a7871';
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, width, height);
      context.fillStyle = bg;
      context.fillRect(0, 0, width, height);
      const strip = phase === 2 ? 26 : 0;
      const { cell, originX, originY, grid } = gridLayout(width, height, n, strip);

      context.save();
      context.globalAlpha = phase === 0 || phase === 3 ? 0.22 : 0.1;
      context.fillStyle = accent;
      context.fillRect(originX, originY + query * cell, grid, cell);
      context.restore();

      context.save();
      context.fillStyle = accent;
      context.fillRect(originX - 7, originY + query * cell + 1, 3, Math.max(2, cell - 2));
      context.restore();

      for (let q = 0; q < n; q += 1) {
        for (let k = 0; k < n; k += 1) {
          const x = originX + k * cell;
          const y = originY + q * cell;
          const size = Math.max(1, cell - 1.25);
          if (k > q) {
            context.save();
            context.globalAlpha = phase === 1 ? 0.22 : 0.1;
            context.fillStyle = phase === 1 ? muted : green;
            context.fillRect(x + 0.5, y + 0.5, size, size);
            if (phase === 1) {
              context.beginPath();
              context.rect(x, y, cell, cell);
              context.clip();
              context.strokeStyle = fg;
              context.globalAlpha = 0.55;
              context.lineWidth = 1;
              context.beginPath();
              const step = 3.4;
              for (let offset = -cell; offset < cell; offset += step) {
                context.moveTo(x + offset, y + cell);
                context.lineTo(x + offset + cell, y);
              }
              context.stroke();
            }
            context.restore();
            continue;
          }
          const weight = attentionWeight(trace, layer, head, q, k);
          let alpha = 0.14 + Math.min(1, weight) * 0.86;
          if ((phase === 0 || phase === 3) && q !== query) alpha *= 0.22;
          if (phase === 1) alpha *= 0.55;
          context.save();
          context.globalAlpha = alpha;
          context.fillStyle = blue;
          context.fillRect(x + 0.5, y + 0.5, size, size);
          context.restore();
        }
      }

      context.save();
      context.strokeStyle = fg;
      context.globalAlpha = 0.28;
      context.lineWidth = 1;
      context.strokeRect(originX + 0.5, originY + 0.5, grid - 1, grid - 1);
      context.restore();

      if (phase === 2) {
        const base = originY + grid + 6;
        const maxBar = 16;
        for (let k = 0; k < n; k += 1) {
          const weight = attentionWeight(trace, layer, head, query, k);
          const bar = Math.max(2, weight * maxBar);
          context.fillStyle = k === keyIndex ? accent : blue;
          context.globalAlpha = k === keyIndex ? 1 : 0.7;
          context.fillRect(originX + k * cell + 1, base + maxBar - bar, Math.max(1, cell - 2), bar);
        }
        context.globalAlpha = 1;
      }

      if (phase === 1) {
        context.save();
        context.fillStyle = fg;
        context.globalAlpha = 0.72;
        context.font = '600 11px ui-monospace, monospace';
        context.textAlign = 'right';
        context.textBaseline = 'top';
        context.fillText('−∞', originX + grid - 5, originY + 4);
        context.restore();
      }

      const ring = (q: number, k: number, color: string, widthPx: number) => {
        context.save();
        context.strokeStyle = color;
        context.lineWidth = widthPx;
        context.strokeRect(
          originX + k * cell + 1,
          originY + q * cell + 1,
          Math.max(1, cell - 2),
          Math.max(1, cell - 2),
        );
        context.restore();
      };
      const hot = hover.current;
      if (hot && (hot.q !== query || hot.k !== keyIndex)) ring(hot.q, hot.k, fg, 1.25);
      ring(query, keyIndex, accent, 1.8);
    };

    paint();
    const ro = new ResizeObserver(paint);
    ro.observe(canvas);
    const root = canvas.closest('.wu-root');
    const schedule = () => requestAnimationFrame(paint);
    const theme = new MutationObserver(schedule);
    if (root) theme.observe(root, { attributes: true, attributeFilter: ['data-theme'] });
    theme.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'style'] });
    (canvas as HTMLCanvasElement & { paintHeat?: () => void }).paintHeat = paint;
    return () => {
      ro.disconnect();
      theme.disconnect();
      delete (canvas as HTMLCanvasElement & { paintHeat?: () => void }).paintHeat;
    };
  }, [trace, layer, head, queryRow, query, keyIndex, n, phase]);

  const cellAt = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = ref.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const { cell, originX, originY } = gridLayout(rect.width, rect.height, n, phase === 2 ? 26 : 0);
    const k = Math.floor((event.clientX - rect.left - originX) / cell);
    const q = Math.floor((event.clientY - rect.top - originY) / cell);
    if (q < 0 || k < 0 || q >= n || k >= n || k > q) return null;
    return { q, k };
  };

  const redraw = () => {
    const canvas = ref.current as (HTMLCanvasElement & { paintHeat?: () => void }) | null;
    canvas?.paintHeat?.();
  };

  return (
    <canvas
      ref={ref}
      className="viz-heat"
      data-heatmap=""
      data-phase={phase}
      role="img"
      aria-label="Quantized causal attention weights"
      onMouseMove={(event) => {
        const cell = cellAt(event);
        const prev = hover.current;
        if (prev?.q === cell?.q && prev?.k === cell?.k) return;
        hover.current = cell;
        redraw();
        onHover(cell?.q ?? null, cell?.k ?? null);
      }}
      onMouseLeave={() => {
        hover.current = null;
        redraw();
        onHover(null, null);
      }}
      onClick={(event) => {
        const cell = cellAt(event);
        if (cell) onPick(cell.q, cell.k);
      }}
    />
  );
}
