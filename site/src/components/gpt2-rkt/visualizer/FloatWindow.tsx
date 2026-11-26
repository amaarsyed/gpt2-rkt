import { useLayoutEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react';

export type WindowTab = 'what' | 'math' | 'code';

interface Props {
  title: string;
  tab: WindowTab;
  onTab: (tab: WindowTab) => void;
  onClose: () => void;
  onCollapse: () => void;
  children: ReactNode;
}

const TABS: { id: WindowTab; label: string }[] = [
  { id: 'what', label: 'What it does' },
  { id: 'math', label: 'Math' },
  { id: 'code', label: 'Code' },
];

export function FloatWindow({ title, tab, onTab, onClose, onCollapse, children }: Props) {
  const frame = useRef<HTMLElement>(null);
  const [box, setBox] = useState({ x: 480, y: 8, w: 420, h: 360 });

  useLayoutEffect(() => {
    const shell = frame.current?.offsetParent;
    const node = frame.current;
    if (!(shell instanceof HTMLElement) || !node) return;
    const stage = shell.querySelector('.viz-stage');
    const stageTop = stage instanceof HTMLElement ? stage.offsetTop : 0;
    const stageHeight = stage instanceof HTMLElement ? stage.clientHeight : shell.clientHeight - stageTop;
    const maxW = Math.max(260, shell.clientWidth - 16);
    const maxH = Math.max(200, stageHeight - 16);
    const w = Math.min(420, maxW);
    const h = Math.min(360, maxH);
    const next = {
      x: Math.max(8, shell.clientWidth - w - 12),
      y: stageTop + 8,
      w,
      h,
    };
    node.style.left = `${next.x}px`;
    node.style.top = `${next.y}px`;
    node.style.width = `${next.w}px`;
    node.style.height = `${next.h}px`;
    setBox(next);
  }, []);

  const drag = (event: ReactPointerEvent<HTMLElement>, mode: 'move' | 'resize') => {
    if (window.matchMedia('(max-width: 720px)').matches) return;
    const shell = frame.current?.offsetParent;
    if (!(shell instanceof HTMLElement)) return;
    event.preventDefault();
    const bounds = shell.getBoundingClientRect();
    const start = { px: event.clientX, py: event.clientY, ...box };
    const move = (ev: PointerEvent) => {
      const dx = ev.clientX - start.px;
      const dy = ev.clientY - start.py;
      if (mode === 'move') {
        setBox((current) => ({
          ...current,
          x: clamp(start.x + dx, 8, Math.max(8, bounds.width - current.w - 8)),
          y: clamp(start.y + dy, 8, Math.max(8, bounds.height - current.h - 8)),
        }));
      } else {
        setBox((current) => ({
          ...current,
          w: clamp(start.w + dx, 280, Math.max(280, bounds.width - current.x - 8)),
          h: clamp(start.h + dy, 220, Math.max(220, bounds.height - current.y - 8)),
        }));
      }
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  return (
    <aside
      className="viz-floatwin"
      ref={frame}
      style={{ left: box.x, top: box.y, width: box.w, height: box.h }}
      data-tour="window"
      data-no-pop=""
      aria-label={`${title} inspector`}
    >
      <header
        className="viz-floatwin-bar"
        onPointerDown={(event) => drag(event, 'move')}
      >
        <strong>{title}</strong>
        <button type="button" data-no-pop="" aria-label="Collapse inspector" onClick={onCollapse}>–</button>
        <button type="button" data-no-pop="" aria-label="Close inspector" onClick={onClose}>×</button>
      </header>
      <div className="viz-floatwin-tabs" role="tablist" aria-label="Inspector sections">
        {TABS.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            data-no-pop=""
            aria-selected={tab === item.id}
            onClick={() => onTab(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className="viz-floatwin-body" role="tabpanel">{children}</div>
      <button
        type="button"
        className="viz-floatwin-resize"
        aria-label="Resize inspector"
        data-no-pop=""
        onPointerDown={(event) => drag(event, 'resize')}
      />
    </aside>
  );
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}
