import { useEffect, useRef, useState } from 'react';
import { RACKET_COMMIT } from '../visualizer/racketSource';
import { paint } from './CodeCard';
import { CARDS, TREE, type StackCard } from './stackCards';

/** A section's file as a single code card. It rises in the first time it scrolls into view. */
export function CodeStack({ cardId }: { cardId: string }) {
  const card = CARDS.find((item) => item.id === cardId) ?? CARDS[0];
  const root = useRef<HTMLElement>(null);
  const [seen, setSeen] = useState(false);

  useEffect(() => {
    const el = root.current;
    if (!el) return;
    const io = new IntersectionObserver(([entry]) => {
      if (entry?.isIntersecting) { setSeen(true); io.disconnect(); }
    }, { rootMargin: '0px 0px -15% 0px' });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  if (!card) return null;
  return (
    <article className="cs-card" ref={root} data-state={seen ? 'front' : 'next'} aria-label={card.title}>
      <header className="cs-head">
        <span className="cs-title">{card.title}</span>
      </header>
      <CardBody card={card} front={seen} />
    </article>
  );
}

function CardBody({ card, front }: { card: StackCard; front: boolean }) {
  const body = useRef<HTMLDivElement>(null);

  // Coming to the front, the card scrolls its first tinted line near the top.
  useEffect(() => {
    if (!front || !body.current) return;
    const first = body.current.querySelector<HTMLElement>('[data-focus]');
    body.current.scrollTo({ top: first ? Math.max(0, first.offsetTop - 28) : 0 });
  }, [front]);

  if (!card.snippet) {
    return (
      <div className="cs-body" ref={body}>
        <pre className="cs-code cs-tree">
          {TREE.map((row) => {
            // The tree glyphs stay plain; the file or folder name links to it on GitHub, at the pinned commit.
            const at = row.line.search(/[A-Za-z]/);
            const lead = at > 0 ? row.line.slice(0, at) : '';
            const name = at > 0 ? row.line.slice(at) : row.line;
            const href = row.path
              ? `https://github.com/amaarsyed/gpt2-rkt/${row.dir ? 'tree' : 'blob'}/${RACKET_COMMIT}/${row.path}`
              : null;
            return (
            <span key={row.line} className="cs-line">
              <span className="cs-src">
                {lead}
                {href ? <a className="cs-file" href={href} target="_blank" rel="noopener noreferrer">{name}</a> : name}
              </span>
              {row.note ? <span className="cs-note">{'; '}{row.note}</span> : null}
            </span>
            );
          })}
        </pre>
      </div>
    );
  }

  const all = card.snippet.chunks.flatMap((chunk) => chunk.code.split('\n')).filter((line) => line.trim());
  const indent = Math.min(...all.map((line) => line.length - line.trimStart().length));
  return (
    <div className="cs-body" ref={body}>
      <pre className="cs-code">
        {card.snippet.chunks.map((chunk, c) => (
          <span key={chunk.start} className="cs-chunk">
            {c > 0 ? <span className="cs-line cs-gap" aria-hidden="true" /> : null}
            {chunk.code.split('\n').map((line, i) => {
              const number = chunk.start + i;
              return (
                <span key={number} className="cs-line" data-focus={card.focus.includes(number) ? '' : undefined}>
                  <span className="cs-ln">{number}</span>
                  <span className="cs-src">{paint(line.slice(Number.isFinite(indent) ? indent : 0).trimEnd())}</span>
                </span>
              );
            })}
          </span>
        ))}
      </pre>
    </div>
  );
}
