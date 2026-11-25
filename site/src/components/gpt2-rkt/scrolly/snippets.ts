import { RACKET_EXCERPTS } from '../visualizer/racketSource';
import type { StageId } from '../visualizer/traceTypes';
import { CODE_NOTES } from '../visualizer/stageCopy';

export interface SnippetChunk {
  start: number;
  code: string;
}

export interface Snippet {
  file: string;
  chunks: SnippetChunk[];
  /** Trailing comments, keyed by source line number. */
  notes: { line: number; note: string }[];
}

/** One stage's excerpt from the pinned commit, with the visualizer's notes as comments. */
export function stageSnippet(...stages: StageId[]): Snippet {
  const first = stages[0] ?? 'attend';
  return {
    file: RACKET_EXCERPTS[first].file,
    chunks: stages.map((stage) => ({ start: RACKET_EXCERPTS[stage].startLine, code: RACKET_EXCERPTS[stage].code })),
    notes: stages.flatMap((stage) => CODE_NOTES[stage] ?? []),
  };
}

/** Verbatim from gpt2-rkt src/nn.rkt @ 75286c5918a303582b7ce99b5378a2dba47ba5c8. */
export const PARALLEL_SNIPPET: Snippet = {
  file: 'src/nn.rkt',
  chunks: [
    {
      "start": 28,
      "code": ";; Keep small products serial to avoid future setup costs.\n(define parallel-work-threshold 100000)"
    },
    {
      "start": 39,
      "code": ";; Split rows at this size; otherwise split output columns.\n(define parallel-axis-threshold 32)\n;; One block per processor avoids dozens of tiny decode futures.\n(define parallel-block-multiple 1)"
    },
    {
      "start": 67,
      "code": "(define (for-parallel-blocks total cpus proc)\n  (cond\n    [(or (fx< cpus 2) (fx< total parallel-axis-threshold))\n     (proc 0 total)]\n    [else\n     (define blocks (fx* parallel-block-multiple cpus))\n     (define jobs\n       (for/list ([b (in-range blocks)])\n         (define start (quotient (fx* b total) blocks))\n         (define end (quotient (fx* (fx+ b 1) total) blocks))\n         (and (fx< start end)\n              (future (lambda () (proc start end))))))\n     (for ([job (in-list jobs)] #:when job)\n       (touch job))]))"
    }
  ],
  notes: [
    { line: 69, note: 'one core, or a short axis: one thread' },
    { line: 72, note: 'one block per processor' },
    { line: 78, note: 'each block runs in its own future' },
    { line: 80, note: 'wait for every block' },
  ],
};

function range(from: number, to: number): number[] {
  return Array.from({ length: to - from + 1 }, (_, i) => from + i);
}

/** Keep only the given source lines of a snippet; runs of consecutive lines stay together, gaps show as …. */
export function pick(snippet: Snippet, lines: number[]): Snippet {
  const source = new Map<number, string>();
  for (const chunk of snippet.chunks) {
    chunk.code.split('\n').forEach((text, i) => source.set(chunk.start + i, text));
  }
  const wanted = [...new Set(lines)].filter((n) => source.has(n)).sort((a, b) => a - b);
  const chunks: SnippetChunk[] = [];
  for (const n of wanted) {
    const last = chunks[chunks.length - 1];
    const lastEnd = last ? last.start + last.code.split('\n').length - 1 : -2;
    if (last && n === lastEnd + 1) last.code += `\n${source.get(n)}`;
    else chunks.push({ start: n, code: source.get(n) ?? '' });
  }
  return { ...snippet, chunks, notes: snippet.notes.filter((note) => wanted.includes(note.line)) };
}

/** One attention head: the causal mask, then the softmax and the weighted mix, as whole forms. */
export const ATTENTION_SNIPPET: Snippet = pick(stageSnippet('attend'), [36, ...range(51, 73)]);

/** The two thresholds, then the whole futures split. */
export const PARALLEL_SHORT: Snippet = pick(PARALLEL_SNIPPET, [29, 40, ...range(67, 80)]);

/** One entry per Racket file: what it does, and a few real lines of it. */
export interface RktFile {
  file: string;
  what: string;
  snippet: Snippet;
}

/** Verbatim lines from gpt2-rkt @ 75286c5918a303582b7ce99b5378a2dba47ba5c8. */
export const RKT: Record<string, RktFile> = {
  tokenizer: {
    file: 'tokenizer.rkt',
    what: 'Splits text into GPT-2 tokens by merging the best-ranked byte pair until no merge is left.',
    snippet: pick(stageSnippet('bpe-merge'), [117, 118, 119, 120, 121, 122]),
  },
  embed: {
    file: 'embed.rkt',
    what: "Adds each token's 768-number vector to a vector for its position.",
    snippet: pick(stageSnippet('input-embed'), [77, 81, 82]),
  },
  block: {
    file: 'block.rkt',
    what: 'One transformer block: the layer norms, one packed matmul for query, key and value, attention, and the feed-forward, each added back onto the residual.',
    snippet: pick(stageSnippet('block-qkv'), [153, 156, 157, 158]),
  },
  attention: {
    file: 'attention.rkt',
    what: 'The causal attention kernel: scores each query against the keys it can see, softmaxes the scores, and mixes the values.',
    snippet: pick(stageSnippet('attend'), [29, 36, 48, 58, 64]),
  },
  nn: {
    file: 'nn.rkt',
    what: 'The number-crunching: matrix products split across futures, and the GELU the feed-forward uses.',
    snippet: {
      file: 'src/nn.rkt',
      chunks: [{"start": 57, "code": "(define (flgelu x)\n  (define x2 (fl* x x))\n  (define x3 (fl* x2 x))\n  (define inner (fl* gelu-k (fl+ x (fl* gelu-cubic x3))))\n  (fl* 0.5 (fl* x (fl+ 1.0 (fltanh inner)))))"}],
      notes: [{ line: 59, note: "GELU's tanh approximation" }],
    },
  },
  model: {
    file: 'model.rkt',
    what: 'Stacks the twelve blocks, applies the final norm, and scores the vocabulary with the tied token table.',
    snippet: pick(stageSnippet('tied-logits'), [193, 196]),
  },
  generate: {
    file: 'generate.rkt',
    what: 'The entry point: encodes the prompt, runs the decode loop, and reports tokens per second. No temperature means greedy.',
    snippet: {
      file: 'src/generate.rkt',
      chunks: [
        {
          "start": 19,
          "code": ";; Temperature is stored as a flonum. Omitted means greedy decoding."
        },
        {
          "start": 63,
          "code": "(define (generate cfg #:output-port [out #f])"
        },
        {
          "start": 67,
          "code": "  (define ids (encode (gpt2-model-tokenizer model) (generate-config-prompt cfg)))"
        },
        {
          "start": 69,
          "code": "  (define result\n    (model-generate model ids (generate-config-max-new-tokens cfg)\n                    #:temperature (generate-config-temperature cfg)\n                    #:top-k (generate-config-top-k cfg)\n                    #:seed (generate-config-seed cfg)"
        }
      ],
      notes: [],
    },
  },
};
