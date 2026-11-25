import { useEffect, useRef, type ReactNode } from 'react';
import { stillness } from '../figures/shared';
import { RACKET_COMMIT, RACKET_EXCERPTS } from './racketSource';
import type { StageId } from './traceTypes';

const KEYS = new Set(['define', 'let', 'let*', 'lambda', 'if', 'cond', 'else', 'unless', 'when', 'for', 'and', 'or', 'begin']);

function permalink(file: string, start: number, end: number): string {
  return `https://github.com/amaarsyed/gpt2-rkt/blob/${RACKET_COMMIT}/${file}#L${start}-L${end}`;
}

function tokens(line: string): ReactNode[] {
  const parts: ReactNode[] = [];
  const pattern = /;.*$|"(?:\\.|[^"\\])*"|'(?:[^\\]|\\.)|#\w+|\b[A-Za-z][\w+!*?-]*|\d+(?:\.\d+)?|[()]|./g;
  let match = pattern.exec(line);
  let index = 0;
  while (match) {
    const text = match[0];
    let kind = '';
    if (text.startsWith(';')) kind = 'comment';
    else if (text.startsWith('"') || text.startsWith("'")) kind = 'string';
    else if (KEYS.has(text)) kind = 'key';
    else if (text === '(' || text === ')') kind = 'paren';
    parts.push(<span key={index} className={kind ? `is-${kind}` : undefined}>{text}</span>);
    index += 1;
    match = pattern.exec(line);
  }
  return parts;
}

export function SourcePane({ stage, callSnippet, notes = [], walkLine = null }: { stage: StageId; callSnippet: string | null; notes?: { line: number; note: string }[]; walkLine?: number | null }) {
  const excerpt = RACKET_EXCERPTS[stage];
  const lines = excerpt.code.split('\n');
  const pre = useRef<HTMLPreElement>(null);
  useEffect(() => {
    const root = pre.current;
    const hot = root?.querySelector('.is-walk') ?? root?.querySelector('.is-hot');
    if (!root || !(hot instanceof HTMLElement)) return;
    const rootRect = root.getBoundingClientRect();
    const hotRect = hot.getBoundingClientRect();
    const top = hotRect.top - rootRect.top + root.scrollTop;
    const next = Math.max(0, top - 36);
    if (stillness()) {
      root.scrollTop = next;
      return;
    }
    const start = root.scrollTop;
    const dist = next - start;
    if (Math.abs(dist) < 1) return;
    const t0 = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - t0) / 180);
      root.scrollTop = start + dist * (1 - (1 - t) ** 3);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [stage, walkLine]);
  return (
    <section className="viz-source" aria-label="Racket source" data-no-pop="">
      <header className="viz-source-head">
        <span>{excerpt.fn}</span>
        <a href={permalink(excerpt.file, excerpt.startLine, excerpt.startLine + lines.length - 1)}>{excerpt.file}:{excerpt.startLine}–{excerpt.startLine + lines.length - 1}</a>
      </header>
      <pre className="viz-code" ref={pre}>
        {lines.map((line, index) => {
          const number = excerpt.startLine + index;
          const hot = number >= excerpt.highlightStart && number <= excerpt.highlightEnd;
          const note = notes.find((item) => item.line === number)?.note;
          return (
            <span key={number} className={[hot ? 'is-hot' : '', number === walkLine ? 'is-walk' : ''].join(' ').trim() || undefined}>
              <span className="viz-code-line">
                <span className="viz-gutter">{number}</span>
                {tokens(line)}
              </span>
              {note ? <span className="viz-line-note">{note}</span> : null}
            </span>
          );
        })}
      </pre>
      {excerpt.callSiteLine !== null && excerpt.callSiteEndLine !== null ? (
        <p className="viz-call">
          <a href={permalink(excerpt.file, excerpt.callSiteLine, excerpt.callSiteEndLine)}>
            call site {excerpt.file}:{excerpt.callSiteLine}–{excerpt.callSiteEndLine}
          </a>
          {callSnippet ? <code>{callSnippet}</code> : null}
        </p>
      ) : null}
    </section>
  );
}
