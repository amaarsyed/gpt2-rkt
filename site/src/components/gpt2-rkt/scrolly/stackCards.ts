import type { StepId } from './steps';
import { PARALLEL_SNIPPET, stageSnippet, type Snippet } from './snippets';

/** The repository as a tree: the files this write-up walks through, with one line on each. */
export const TREE: { line: string; note?: string; path?: string; dir?: boolean }[] = [
  { line: 'gpt2-rkt/' },
  { line: '├── src/', path: 'src', dir: true },
  { line: '│   ├── safetensors.rkt', note: 'reads the 124M weights', path: 'src/safetensors.rkt' },
  { line: '│   ├── tokenizer.rkt', note: 'byte-pair encoding', path: 'src/tokenizer.rkt' },
  { line: '│   ├── embed.rkt', note: 'token + position embeddings', path: 'src/embed.rkt' },
  { line: '│   ├── block.rkt', note: 'one transformer block', path: 'src/block.rkt' },
  { line: '│   ├── attention.rkt', note: 'causal attention, one head', path: 'src/attention.rkt' },
  { line: '│   ├── nn.rkt', note: 'matmul, GELU, futures', path: 'src/nn.rkt' },
  { line: '│   ├── model.rkt', note: 'twelve blocks, final norm, logits', path: 'src/model.rkt' },
  { line: '│   └── generate.rkt', note: 'the decode loop', path: 'src/generate.rkt' },
  { line: '├── tests/', note: 'checks against a reference', path: 'tests', dir: true },
  { line: '└── bench/', note: 'timing runs', path: 'bench', dir: true },
];

export interface StackCard {
  id: string;
  title: string;
  /** The module node this file is in the visualizer, if any. */
  moduleId: string | null;
  snippet: Snippet | null;
  /** Lines to tint, by their number in the file. */
  focus: number[];
}

const range = (from: number, to: number) => Array.from({ length: to - from + 1 }, (_, i) => from + i);

/** The cards in reading order. Each step brings one to the front. */
export const CARDS: StackCard[] = [
  { id: 'tree', title: 'gpt2-rkt/', moduleId: null, snippet: null, focus: [] },
  { id: 'tokenizer', title: 'tokenizer.rkt', moduleId: 'mod-tokenizer', snippet: stageSnippet('bpe-merge'), focus: range(117, 125) },
  { id: 'attention', title: 'attention.rkt', moduleId: 'mod-attention', snippet: stageSnippet('attend'), focus: [36, 48, 58, 64] },
  { id: 'model', title: 'model.rkt', moduleId: 'mod-model', snippet: stageSnippet('tied-logits'), focus: [196] },
  { id: 'block', title: 'block.rkt', moduleId: 'mod-block', snippet: stageSnippet('kv-write'), focus: range(122, 125) },
  { id: 'nn', title: 'nn.rkt', moduleId: 'mod-nn', snippet: PARALLEL_SNIPPET, focus: [29, 40, 69, 72, 78, 80] },
  { id: 'tree-end', title: 'gpt2-rkt/', moduleId: null, snippet: null, focus: [] },
];

export const STEP_CARD: Record<StepId, string> = {
  intro: 'tree',
  'forward-pass': 'tree',
  tokenize: 'tokenizer',
  block: 'attention',
  logits: 'model',
  'kv-cache': 'block',
  'kv-copies': 'block',
  parallelism: 'nn',
  checks: 'nn',
  'read-the-code': 'tree-end',
};
