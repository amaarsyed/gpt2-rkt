import { useEffect, useRef, useState, type MouseEvent as ReactMouseEvent, type PointerEvent as ReactPointerEvent } from 'react';
import { createPortal } from 'react-dom';
import katex from 'katex';
import 'katex/dist/katex.min.css';

/**
 * What each symbol in the write-up's equations means. An equation marks a symbol with
 * \sym{key}{...}; hovering it shows this sentence.
 */
const GLOSSARY: Record<string, { name: string; what: string }> = {
  xi: { name: String.raw`x_i`, what: String.raw`The vector for the token at position $i$ as it enters the first block: 768 numbers.` },
  Wte: { name: String.raw`W_{te}`, what: String.raw`The token embedding table, $50257 \times 768$: one row per token. The model reuses it at the end to score tokens.` },
  ti: { name: String.raw`t_i`, what: String.raw`The id of the token at position $i$.` },
  Wpe: { name: String.raw`W_{pe}`, what: String.raw`The position embedding table, $1024 \times 768$: one row per position.` },
  i: { name: String.raw`i`, what: String.raw`The token's position in the text, counting from $0$.` },
  R768: { name: String.raw`\mathbb{R}^{768}`, what: String.raw`768 real numbers: GPT-2 small's width.` },
  Attn: { name: String.raw`\mathrm{Attn}`, what: String.raw`Attention: each position takes a weighted mix of its own and earlier positions' values.` },
  Q: { name: String.raw`Q`, what: String.raw`Queries: one row per position, what that position is looking for.` },
  K: { name: String.raw`K`, what: String.raw`Keys: one row per position, what that position offers to be matched against.` },
  V: { name: String.raw`V`, what: String.raw`Values: one row per position, what that position hands over when attention picks it.` },
  softmax: { name: String.raw`\mathrm{softmax}`, what: String.raw`Turns each row of scores into weights that are positive and add up to $1$: $\;e^{s_j} / \sum_k e^{s_k}$.` },
  QKT: { name: String.raw`QK^{\top}`, what: String.raw`Every query dotted with every key: one score $q_a \cdot k_b$ for each pair of positions.` },
  sqrtdk: { name: String.raw`\sqrt{d_k}`, what: String.raw`Dividing by $\sqrt{64} = 8$ keeps the scores from growing with the size of a head.` },
  dk: { name: String.raw`d_k`, what: String.raw`The width of one attention head: $768 / 12 = 64$.` },
  M: { name: String.raw`M`, what: String.raw`The causal mask: it hides later positions, so a token only attends to itself and earlier tokens.` },
  MLP: { name: String.raw`\mathrm{MLP}`, what: String.raw`The feed-forward network. It runs on each position on its own.` },
  x: { name: String.raw`x`, what: String.raw`One position's vector, 768 numbers, as it moves through the block.` },
  W1: { name: String.raw`W_1`, what: String.raw`Expands each vector: $\mathbb{R}^{768} \to \mathbb{R}^{3072}$.` },
  W2: { name: String.raw`W_2`, what: String.raw`Brings each vector back: $\mathbb{R}^{3072} \to \mathbb{R}^{768}$.` },
  GELU: { name: String.raw`\mathrm{GELU}`, what: String.raw`A smooth nonlinearity, $x\,\Phi(x)$: passes positive inputs through and squashes negative ones toward $0$.` },
  z: { name: String.raw`z`, what: String.raw`The logits: one score for each token in the vocabulary.` },
  LN1: { name: String.raw`\mathrm{LN}_1`, what: String.raw`The block's first layer norm: rescales each vector to mean $0$ and variance $1$, then applies a learned scale and shift, before attention.` },
  LN2: { name: String.raw`\mathrm{LN}_2`, what: String.raw`The block's second layer norm, before the feed-forward network.` },
  b1: { name: String.raw`b_1`, what: String.raw`The first bias, $3072$ numbers, added after $W_1$.` },
  b2: { name: String.raw`b_2`, what: String.raw`The second bias, $768$ numbers, added after $W_2$.` },
  LN: { name: String.raw`\mathrm{LN}`, what: String.raw`The final layer norm: rescales the vector to mean $0$ and variance $1$, then applies a learned scale and shift.` },
  hL: { name: String.raw`h_L`, what: String.raw`The vector at the last position after all twelve blocks.` },
  R50257: { name: String.raw`\mathbb{R}^{50257}`, what: String.raw`One score for each of GPT-2's 50,257 tokens.` },
  tnext: { name: String.raw`t_{\mathrm{next}}`, what: String.raw`The token the model picks next.` },
  argmax: { name: String.raw`\arg\max_j`, what: String.raw`The index $j$ with the highest score. Always taking it is greedy decoding.` },
  rows: { name: String.raw`\text{rows}`, what: String.raw`The rows of the left matrix: one per token being processed. A long prompt has many; one decoding token has one.` },
  n32: { name: String.raw`32`, what: String.raw`The row threshold in nn.rkt, parallel-axis-threshold.` },
};

/** Prose with $…$ spans rendered as inline LaTeX. */
function proseHtml(text: string): string {
  return text
    .split(/(\$[^$]+\$)/g)
    .map((part) => (part.startsWith('$') && part.endsWith('$')
      ? katex.renderToString(part.slice(1, -1), { throwOnError: false, output: 'html' })
      : part.replace(/&/g, '&amp;').replace(/</g, '&lt;')))
    .join('');
}

const OPTIONS = {
  throwOnError: false,
  output: 'html' as const,
  strict: false as const,
  trust: (context: { command: string }) => context.command === '\\htmlData',
  macros: { '\\sym': '\\htmlData{sym=#1}{#2}' },
};

/** LaTeX rendered with KaTeX. `block` centers it on its own line. Marked symbols explain themselves on hover. */
export function Tex({ children, block = false }: { children: string; block?: boolean }) {
  const html = katex.renderToString(children, { ...OPTIONS, displayMode: block });
  const [tip, setTip] = useState<{ key: string; x: number; y: number } | null>(null);
  const Tag = block ? 'div' : 'span';

  // The tip is pinned to where the symbol was; scrolling moves the symbol, so the tip goes.
  useEffect(() => {
    if (!tip) return;
    const hide = () => setTip(null);
    window.addEventListener('scroll', hide, { passive: true });
    return () => window.removeEventListener('scroll', hide);
  }, [tip]);

  const root = useRef<HTMLElement | null>(null);
  // Touch has no hover: a tap opens a symbol's tip, a second tap or a tap elsewhere closes it.
  const touched = useRef(false);

  const tipFor = (target: EventTarget | null) => {
    const el = (target as HTMLElement | null)?.closest<HTMLElement>('[data-sym]');
    const key = el?.dataset.sym;
    if (!el || !key || !GLOSSARY[key]) return null;
    const box = el.getBoundingClientRect();
    // Keep the tip (up to 340px wide, centered on the symbol) inside the window.
    const half = Math.min(170, window.innerWidth / 2 - 12);
    const x = Math.min(Math.max(box.left + box.width / 2, half + 12), window.innerWidth - half - 12);
    return { key, x, y: box.top };
  };

  const onOver = (event: ReactPointerEvent<HTMLElement>) => {
    if (event.pointerType === 'touch') return;
    setTip(tipFor(event.target));
  };

  const onLeave = (event: ReactPointerEvent<HTMLElement>) => {
    if (event.pointerType !== 'touch') setTip(null);
  };

  const onDown = (event: ReactPointerEvent<HTMLElement>) => {
    touched.current = event.pointerType === 'touch';
  };

  const onTap = (event: ReactMouseEvent<HTMLElement>) => {
    if (!touched.current) return;
    const next = tipFor(event.target);
    setTip((current) => (next && current?.key !== next.key ? next : null));
  };

  // A tap anywhere outside this equation closes its tip.
  useEffect(() => {
    if (!tip) return;
    const away = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setTip(null);
    };
    document.addEventListener('pointerdown', away, true);
    return () => document.removeEventListener('pointerdown', away, true);
  }, [tip]);

  // A display equation wider than the column scrolls sideways; flag it so the edge can fade.
  useEffect(() => {
    const el = root.current;
    if (!block || !el) return;
    const check = () => {
      // Only while there is more to the right: scrolled to the end, the edge is solid.
      if (el.scrollLeft + el.clientWidth < el.scrollWidth - 1) el.dataset.overflow = '';
      else delete el.dataset.overflow;
    };
    check();
    const sizes = new ResizeObserver(check);
    sizes.observe(el);
    el.addEventListener('scroll', check, { passive: true });
    return () => {
      sizes.disconnect();
      el.removeEventListener('scroll', check);
    };
  }, [block, html]);

  const entry = tip ? GLOSSARY[tip.key] : null;
  return (
    <>
      <Tag
        className={block ? 'tex-block' : 'tex-inline'}
        data-active-sym={tip?.key}
        ref={(el: HTMLElement | null) => { root.current = el; }}
        onPointerOver={onOver}
        onPointerLeave={onLeave}
        onPointerDown={onDown}
        onClick={onTap}
        dangerouslySetInnerHTML={{ __html: html }}
      />
      {tip && entry
        ? createPortal(
          <span className="tex-tip" role="tooltip" style={{ left: tip.x, top: tip.y }}>
            <span
              className="tex-tip-sym"
              dangerouslySetInnerHTML={{ __html: katex.renderToString(entry.name, { throwOnError: false, output: 'html' }) }}
            />
            <span className="tex-tip-what" dangerouslySetInnerHTML={{ __html: proseHtml(entry.what) }} />
          </span>,
          document.body,
        )
        : null}
    </>
  );
}
