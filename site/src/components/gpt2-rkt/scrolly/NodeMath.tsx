import { useEffect, useLayoutEffect, useRef, type ReactNode } from 'react';
import { activeQueryRow, causalRow, tokenText } from '../visualizer/attentionMath';
import { stageLabel } from '../visualizer/executionState';
import { executionGraph, nodeById } from '../visualizer/graphModel';
import { isStageId, type ExecutionTrace, type StageId, type TensorSummary } from '../visualizer/traceTypes';
import type { VizCore } from '../visualizer/useVizCore';
import { RKT } from './snippets';

/** Below this width the node card opens as a bottom sheet instead of beside the node. */
const SHEET = '(max-width: 560px)';
import { Tex } from './Tex';

/** What each stage does, said plainly, and the one equation that says it exactly. */
const STAGE_MATH: Record<StageId, { idea: string; tex: string }> = {
  'bpe-merge': {
    idea: 'Start from the raw bytes and keep gluing together the neighbouring pair GPT-2 learned first, until no learned pair is left. What remains are the tokens.',
    tex: String.raw`\text{while a ranked pair } (a, b) \text{ is left:}\quad a\,b \;\to\; ab`,
  },
  'input-embed': {
    idea: 'Look up a learned vector for the token and add one for its position, so the model knows both what the token is and where it sits.',
    tex: String.raw`\sym{xi}{x_i} = \sym{Wte}{W_{te}}[\sym{ti}{t_i}] + \sym{Wpe}{W_{pe}}[\sym{i}{i}]`,
  },
  'block-ln1': {
    idea: 'Re-centre the 768 numbers and scale them to a steady spread, so attention always sees inputs of the same size.',
    tex: String.raw`\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta`,
  },
  'block-qkv': {
    idea: 'One matrix product turns each vector into three: a query (what am I looking for?), a key (what do I offer?), and a value (what do I pass on?).',
    tex: String.raw`[\,\sym{Q}{q}\;\; \sym{K}{k}\;\; \sym{V}{v}\,] = \sym{LN1}{\mathrm{LN}_1}(\sym{x}{x})\, W_{qkv} + b, \qquad 768 \to 3 \times 768`,
  },
  'kv-write': {
    idea: 'Save this token’s key and value in the cache, so every later token can look back at it without recomputing anything.',
    tex: String.raw`K[t] \leftarrow \sym{K}{k_t}, \qquad V[t] \leftarrow \sym{V}{v_t}`,
  },
  attend: {
    idea: 'Score how well this token’s query matches each earlier key, turn the scores into percentages, then blend the values by those percentages.',
    tex: String.raw`\alpha_j = \sym{softmax}{\mathrm{softmax}}_j\!\left(\frac{\sym{Q}{q} \cdot \sym{K}{k_j}}{\sym{sqrtdk}{\sqrt{64}}}\right), \qquad o = \sum_{j \le t} \alpha_j \, \sym{V}{v_j}`,
  },
  'block-proj-res': {
    idea: 'Join the 12 heads’ outputs, project them back to 768 numbers, and add the result onto the running vector. The add keeps what was already there.',
    tex: String.raw`\sym{x}{x} \leftarrow \sym{x}{x} + W_o\,[\,o_1; \dots; o_{12}\,] + b_o`,
  },
  'block-ln2': {
    idea: 'Normalise again, this time for the feed-forward network.',
    tex: String.raw`\sym{LN2}{\mathrm{LN}_2}(x) = \gamma_2 \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_2`,
  },
  'mlp-gelu': {
    idea: 'Widen each vector to 3,072 numbers, bend them with GELU so the block can learn non-linear features, ready to narrow back.',
    tex: String.raw`h = \sym{GELU}{\mathrm{GELU}}(\sym{x}{x}\,\sym{W1}{W_1} + \sym{b1}{b_1}) \in \mathbb{R}^{3072}, \quad \mathrm{GELU}(u) \approx \tfrac{u}{2}\bigl(1 + \tanh(0.798\,(u + 0.0447\,u^3))\bigr)`,
  },
  'block-mlp-res': {
    idea: 'Narrow back to 768 numbers and add onto the running vector, the second residual of the block.',
    tex: String.raw`\sym{x}{x} \leftarrow \sym{x}{x} + h\,\sym{W2}{W_2} + \sym{b2}{b_2}`,
  },
  'final-ln': {
    idea: 'After the twelfth block, normalise the last position’s vector once more before scoring the vocabulary.',
    tex: String.raw`\sym{LN}{\mathrm{LN}}(\sym{hL}{h_L})`,
  },
  'tied-logits': {
    idea: 'Dot the final vector with every token’s own embedding. Tokens that point the same way score higher; softmax turns the scores into probabilities.',
    tex: String.raw`\sym{z}{z} = \sym{Wte}{W_{te}}\,\sym{LN}{\mathrm{LN}}(\sym{hL}{h_L}), \qquad p = \sym{softmax}{\mathrm{softmax}}(\sym{z}{z})`,
  },
  sample: {
    idea: 'Greedy decoding: take the token with the highest score. That token is appended and the whole pass runs again for the next one.',
    tex: String.raw`\sym{tnext}{t_{\text{next}}} = \sym{argmax}{\arg\max_j}\, \sym{z}{z_j}`,
  },
};

function fmt(value: number, digits = 2): string {
  const fixed = value.toFixed(digits);
  return fixed === `-${(0).toFixed(digits)}` ? (0).toFixed(digits) : fixed;
}

function quote(text: string): string {
  const shown = text.replace(/\n/g, '↵').trim();
  return `“${shown === '' ? '·' : shown}”`;
}

function Stats({ tensor, note }: { tensor: TensorSummary | undefined; note?: string }) {
  if (!tensor) return null;
  return (
    <dl className="nm-stats">
      <div><dt>shape</dt><dd>{tensor.shape.join(' × ')}</dd></div>
      <div><dt>mean</dt><dd>{fmt(tensor.mean)}</dd></div>
      <div><dt>spread</dt><dd>{fmt(tensor.std)}</dd></div>
      <div><dt>range</dt><dd>{fmt(tensor.min, 1)} … {fmt(tensor.max, 1)}</dd></div>
      {note ? <p className="nm-stats-note">{note}</p> : null}
    </dl>
  );
}

function Bars({ rows }: { rows: { label: string; value: number; hint?: string; strong?: boolean }[] }) {
  const peak = Math.max(...rows.map((row) => row.value), 1e-9);
  return (
    <ol className="nm-bars">
      {rows.map((row, index) => (
        <li key={`${row.label}-${index}`} data-strong={row.strong ? '' : undefined}>
          <span className="nm-bar-label">{row.label}</span>
          <span className="nm-bar-track" aria-hidden="true">
            <i style={{ width: `${Math.max(2, (row.value / peak) * 100)}%` }} />
          </span>
          <span className="nm-bar-value">{row.hint ?? `${(row.value * 100).toFixed(1)}%`}</span>
        </li>
      ))}
    </ol>
  );
}

/** The recorded run, as numbers a reader can check against the equation. */
function Live({ trace, stage, stepIndex, layer, head }: { trace: ExecutionTrace; stage: StageId; stepIndex: number; layer: number; head: number }) {
  const step = trace.exact.steps[stepIndex];
  const block = step?.blocks[layer];
  const query = activeQueryRow(stepIndex);

  switch (stage) {
    case 'bpe-merge':
      return (
        <div className="nm-tokens">
          {trace.exact.bpe.chunks.flatMap((chunk) => chunk.tokenStrings.map((piece, i) => (
            <span key={`${chunk.text}-${i}`} className="nm-token">
              <b>{piece.replace('Ġ', '␣')}</b>
              <small>{chunk.tokenIds[i]}</small>
            </span>
          )))}
        </div>
      );
    case 'input-embed': {
      const ids = step?.inputIds ?? [];
      const at = Math.max(0, query);
      return (
        <>
          <p className="nm-line">
            This step embeds {ids.length === 1 ? 'one token' : `${ids.length} tokens`}
            {ids.length === 1 ? <> : {quote(tokenText(trace, at))} is token <b>{ids[0]}</b> at position <b>{at}</b>.</> : '.'}
          </p>
          <Stats tensor={step?.['input-embed']} />
        </>
      );
    }
    case 'attend': {
      const weights = causalRow(trace, layer, head, query);
      const rows = weights.map((value, key) => ({ label: quote(tokenText(trace, key)), value, strong: false }));
      const best = rows.reduce((a, b, i) => (b.value > (rows[a]?.value ?? 0) ? i : a), 0);
      if (rows[best]) rows[best].strong = true;
      return (
        <>
          <p className="nm-line">
            How much {quote(tokenText(trace, query))} attends to each earlier token, layer {layer + 1}, head {head + 1}:
          </p>
          <Bars rows={rows} />
        </>
      );
    }
    case 'kv-write': {
      const kv = block?.['kv-write'];
      if (!kv) return <p className="nm-line">The cache grows by one key row and one value row per token, in every block.</p>;
      return (
        <p className="nm-line">
          Rows filled: <b>{kv.filledBefore}</b> → <b>{kv.filledAfter}</b> of {kv.capacity}, each {kv.width} numbers wide.
        </p>
      );
    }
    case 'tied-logits':
    case 'sample': {
      const top = step?.['tied-logits']?.top ?? [];
      if (!top.length) return null;
      return (
        <>
          <p className="nm-line">{stage === 'sample' ? 'The pick is the top row:' : 'The five likeliest next tokens:'}</p>
          <Bars rows={top.slice(0, 5).map((row, i) => ({ label: quote(row.text), value: row.prob, strong: i === 0 }))} />
        </>
      );
    }
    case 'final-ln':
      return <Stats tensor={step?.['final-ln']} note="After normalising: mean near 0, before the learned scale." />;
    default: {
      const tensor = block?.[stage as 'block-ln1'] as TensorSummary | undefined;
      return tensor ? <Stats tensor={tensor} /> : null;
    }
  }
}

/**
 * A clicked node's math, on the visualizer: the idea in a sentence, the equation,
 * and the recorded numbers. It goes away when the pointer clicks elsewhere or leaves.
 */
export function NodeMath({ core }: { core: VizCore }) {
  const { trace, state, setState } = core;
  const card = useRef<HTMLElement>(null);
  const selectedRef = useRef(state.selectedNodeId);
  selectedRef.current = state.selectedNodeId;
  const open = state.inspect && Boolean(state.selectedNodeId);

  useEffect(() => {
    if (!open) return;
    const close = () => setState((current) => ({ ...current, inspect: false, selectedNodeId: null, hoverPair: null }));
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') close(); };
    // A press anywhere but the card closes it, unless that press picked another node.
    const onDown = (event: PointerEvent) => {
      if (card.current?.contains(event.target as Node)) return;
      const before = selectedRef.current;
      window.setTimeout(() => { if (selectedRef.current === before) close(); }, 80);
    };
    // Leaving the visualizer side closes it, after a short grace period.
    const stage = card.current?.closest('.sc-stage');
    let timer = 0;
    const onLeave = () => { timer = window.setTimeout(close, 350); };
    const onEnter = () => window.clearTimeout(timer);
    window.addEventListener('keydown', onKey);
    window.addEventListener('pointerdown', onDown, true);
    stage?.addEventListener('pointerleave', onLeave);
    stage?.addEventListener('pointerenter', onEnter);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('pointerdown', onDown, true);
      stage?.removeEventListener('pointerleave', onLeave);
      stage?.removeEventListener('pointerenter', onEnter);
      window.clearTimeout(timer);
    };
  }, [open, setState]);

  // Open beside the clicked node without covering it. The graph never moves for the card,
  // so the card does the fitting: right of the node, else left, else below or above,
  // narrowing or shortening (the body scrolls) to stay inside the visualizer.
  const nodeId = open ? state.selectedNodeId : null;
  useLayoutEffect(() => {
    const el = card.current;
    if (!nodeId || !el) return;
    const root = el.closest<HTMLElement>('.sc-viz');
    if (!root) return;
    // On a phone the card is a bottom sheet (scrolly.css): no placing beside the node.
    if (window.matchMedia(SHEET).matches) {
      el.style.translate = '';
      el.style.width = '';
      el.style.maxHeight = '';
      el.dataset.placed = '';
      return;
    }
    let frame = 0;
    const place = () => {
      const hit = root.querySelector<HTMLElement>(`.viz-hit[data-node="${CSS.escape(nodeId)}"]`);
      const box = root.getBoundingClientRect();
      const margin = 12;
      const gap = 22;
      const full = Math.min(380, box.width - margin * 2);
      const tall = box.height - margin * 2;
      let ax = box.width / 2;
      let ay = box.height / 3;
      if (hit) {
        const r = hit.getBoundingClientRect();
        ax = r.left + r.width / 2 - box.left;
        ay = r.top + r.height / 2 - box.top;
      }
      // Room on each side of the node, inside the visualizer.
      const room = {
        right: box.width - margin - (ax + gap),
        left: ax - gap - margin,
        below: box.height - margin - (ay + gap),
        above: ay - gap - margin,
      };
      const narrowest = Math.min(full, 280);
      const side: keyof typeof room =
        room.right >= full ? 'right'
          : room.left >= full ? 'left'
            : Math.max(room.right, room.left) >= narrowest ? (room.right >= room.left ? 'right' : 'left')
              : room.below >= 200 || room.above >= 200 ? (room.below >= room.above ? 'below' : 'above')
                : (['right', 'left', 'below', 'above'] as const).reduce((a, b) => (room[b] > room[a] ? b : a));
      const beside = side === 'right' || side === 'left';
      const w = beside ? Math.max(Math.min(full, room[side]), Math.min(full, 220)) : full;
      const maxH = beside ? tall : Math.max(Math.min(tall, room[side]), 140);
      el.style.width = `${Math.round(w)}px`;
      el.style.maxHeight = `${Math.round(maxH)}px`;
      const h = el.offsetHeight;
      let x: number;
      let y: number;
      if (beside) {
        x = side === 'right' ? ax + gap : ax - gap - w;
        y = ay - 28;
      } else {
        x = ax - w / 2;
        y = side === 'below' ? ay + gap : ay - gap - h;
      }
      x = Math.min(Math.max(x, margin), Math.max(margin, box.width - w - margin));
      y = Math.min(Math.max(y, margin), Math.max(margin, box.height - h - margin));
      el.style.translate = `${Math.round(x)}px ${Math.round(y)}px`;
      el.style.transformOrigin = beside
        ? `${side === 'right' ? 'left' : 'right'} ${Math.round(Math.min(Math.max(ay - y, 0), h))}px`
        : `${Math.round(Math.min(Math.max(ax - x, 0), w))}px ${side === 'below' ? 'top' : 'bottom'}`;
      el.dataset.placed = '';
      frame = requestAnimationFrame(place);
    };
    place();
    return () => cancelAnimationFrame(frame);
  }, [nodeId]);

  if (!open || !state.selectedNodeId) return null;
  const node = nodeById(executionGraph(), state.selectedNodeId);
  if (!node) return null;

  let body: ReactNode;
  let meta = '';
  if (node.stageId && isStageId(node.stageId)) {
    const math = STAGE_MATH[node.stageId];
    const step = trace?.exact.steps[state.stepIndex];
    meta = step ? `${step.phase === 'prefill' ? 'prefill' : `token ${state.stepIndex + 1}`}${node.id.startsWith('block') || node.id === 'attend' || node.id === 'kv-write' || node.id === 'mlp-gelu' ? ` · layer ${state.activeLayer + 1}` : ''}` : '';
    body = (
      <>
        <p className="nm-idea">{math.idea}</p>
        <div className="nm-eq"><Tex block>{math.tex}</Tex></div>
        {trace ? (
          <section className="nm-live" aria-label="In the recorded run">
            <h4>In the recorded run</h4>
            <Live trace={trace} stage={node.stageId} stepIndex={state.stepIndex} layer={state.activeLayer} head={state.head} />
          </section>
        ) : null}
      </>
    );
  } else if (node.id.startsWith('mod-')) {
    const key = node.id.slice(4);
    const entry = RKT[key];
    meta = 'Racket file';
    body = <p className="nm-idea">{entry?.what ?? node.sub}</p>;
  } else {
    meta = 'input';
    body = (
      <>
        <p className="nm-idea">The prompt the recorded run starts from. The tokenizer turns it into token ids.</p>
        {trace ? <p className="nm-line"><b>{quote(trace.run.prompt ?? 'Once upon a time')}</b></p> : null}
      </>
    );
  }

  return (
    <aside
      ref={card}
      className="nm"
      data-role={node.role}
      role="dialog"
      aria-label={`${node.label}: the math`}
    >
      <header className="nm-head">
        <i aria-hidden="true" />
        <h3>{node.stageId ? stageLabel(node.stageId) : node.label}</h3>
        {meta ? <span className="nm-meta">{meta}</span> : null}
        <button
          type="button"
          className="nm-close"
          aria-label="Close"
          onClick={() => setState((current) => ({ ...current, inspect: false, selectedNodeId: null }))}
        >
          ×
        </button>
      </header>
      <div className="nm-body" data-lenis-prevent="">{body}</div>
    </aside>
  );
}

