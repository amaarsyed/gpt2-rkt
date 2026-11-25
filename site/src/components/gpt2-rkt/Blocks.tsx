import { useEffect, useRef, useState } from 'react';
import { stillness } from './figures/shared';

/** shadcn-style code block: bordered, muted surface, and a small header. */
export function CodeBlock({ code, title = 'Terminal' }: { code: string; title?: string }) {
  return (
    <div className="wu-code">
      <div className="wu-code-head">
        <span className="wu-code-title">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <polyline points="4 17 10 11 4 5" />
            <line x1="12" y1="19" x2="20" y2="19" />
          </svg>
          {title}
        </span>
      </div>
      <pre className="wu-code-pre">
        <code>{code}</code>
      </pre>
    </div>
  );
}

const GENERATE_CMD = `racket main.rkt generate \\
  --model-dir models/gpt2 \\
  --prompt "Once upon a time" \\
  --max-new-tokens 16`;

/** The recorded run, as real text: the command, then the greedy continuation. */
export function GenerateBlock() {
  return (
    <figure className="wu-card">
      <div className="wu-card-title">
        One recorded run <span>Greedy · 16 new tokens</span>
      </div>
      <CodeBlock code={GENERATE_CMD} />
      <p className="wu-output">
        <mark>Once upon a time</mark>, the world was a place of great beauty and great danger. The world was
      </p>
      <figcaption className="wu-tie">↳ Real Racket output, formatted for this figure</figcaption>
    </figure>
  );
}

const TOKENS = ['Once', '␣upon', '␣a', '␣time', ','];
const PROMPT = 4;

/**
 * The timeline, one tick at a time: an empty buffer sized for all five tokens,
 * prefill writing the four prompt columns, a hold, then decode writing only the
 * comma's column while the first four are reused. Then it starts over.
 */
const TICKS = [
  { written: 0, ms: 900 },
  { written: 1, ms: 420 },
  { written: 2, ms: 420 },
  { written: 3, ms: 420 },
  { written: 4, ms: 1500 },
  { written: 5, ms: 2600 },
] as const;

function useCacheTimeline() {
  const ref = useRef<HTMLElement>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (stillness()) {
      setTick(TICKS.length - 1);
      return;
    }
    const el = ref.current;
    if (!el) return;
    let timer = 0;
    let visible = false;
    let current = 0;
    const advance = () => {
      if (!visible) return;
      current = (current + 1) % TICKS.length;
      setTick(current);
      timer = window.setTimeout(advance, TICKS[current].ms);
    };
    const io = new IntersectionObserver(
      ([entry]) => {
        visible = entry.isIntersecting;
        window.clearTimeout(timer);
        if (visible) timer = window.setTimeout(advance, TICKS[current].ms);
      },
      { threshold: 0.4 },
    );
    io.observe(el);
    return () => {
      io.disconnect();
      window.clearTimeout(timer);
    };
  }, []);

  return { ref, written: TICKS[tick].written };
}

function Cell({ kind, n, written }: { kind: 'k' | 'v'; n: number; written: number }) {
  const on = n <= written;
  const added = on && n > PROMPT;
  const reused = written > PROMPT && n <= PROMPT;
  const cls = ['wu-cell', on && 'is-on', added && 'is-added', reused && 'is-reused'].filter(Boolean).join(' ');
  return (
    <td className={cls}>
      <span>
        {kind}
        <sub>{n}</sub>
      </span>
    </td>
  );
}

/** One layer's key/value cache, animated: prefill fills four slots, decode adds the comma's. */
export function CacheTable() {
  const { ref, written } = useCacheTimeline();
  const decoding = written > PROMPT;
  const phase = decoding ? 'Decode · the comma' : written === 0 ? 'Empty buffer' : 'Prefill · the prompt';
  const status = decoding
    ? '4 tokens reused, 1 added'
    : `${written} of ${PROMPT} prompt tokens stored`;

  return (
    <figure ref={ref} className="wu-cache" aria-label="One layer's key/value cache. Prefill stores a key and a value for each of the four prompt tokens; processing the comma reuses those four and adds one more.">
      <div className="wu-card-title">
        One layer’s key/value cache <span>␣ marks a space in the token</span>
      </div>
      <div className="wu-cache-head" aria-live="polite">
        <h3>{phase}</h3>
        <span>{status}</span>
      </div>
      <table className="wu-cache-table">
        <thead>
          <tr>
            <th aria-hidden="true" />
            {TOKENS.map((t, i) => (
              <th key={t} scope="col" className={i < written ? 'is-on' : undefined}>
                <code>{t}</code>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {(['k', 'v'] as const).map((kind) => (
            <tr key={kind}>
              <th scope="row">{kind === 'k' ? 'Key' : 'Value'}</th>
              {TOKENS.map((t, i) => (
                <Cell key={t} kind={kind} n={i + 1} written={written} />
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}
