import type { VizMode } from '../visualizer/executionState';
import type { StageId } from '../visualizer/traceTypes';

/** What the pinned visualizer shows while a block of the write-up sits on the reading line. */
export interface StepScene {
  mode: VizMode;
  /** Stages the pulse walks through, in order, while this step is active. */
  stages: StageId[];
  /** The node the camera frames. Null shows the whole graph. */
  focus: string | null;
  /** Recorded steps to sweep through, inclusive, for the cache chapter. */
  sweep?: [number, number];
  /** The recorded step to hold, when it matters. */
  step?: number;
}

export const SCENES = {
  intro: { mode: 'explore', stages: ['bpe-merge'], focus: null },
  'forward-pass': { mode: 'explore', stages: ['bpe-merge'], focus: 'mod-tokenizer' },
  tokenize: { mode: 'execute', stages: ['bpe-merge', 'input-embed'], focus: 'bpe-merge', step: 0 },
  block: {
    mode: 'execute',
    stages: ['block-ln1', 'block-qkv', 'kv-write', 'attend', 'block-proj-res', 'block-ln2', 'mlp-gelu', 'block-mlp-res'],
    focus: 'attend',
    step: 0,
  },
  logits: { mode: 'execute', stages: ['final-ln', 'tied-logits', 'sample'], focus: 'tied-logits', step: 0 },
  'kv-cache': { mode: 'execute', stages: ['kv-write'], focus: 'kv-write', step: 1 },
  'kv-copies': { mode: 'execute', stages: ['kv-write'], focus: 'kv-write', sweep: [1, 16] },
  parallelism: { mode: 'explore', stages: ['tied-logits'], focus: 'mod-model' },
  checks: { mode: 'execute', stages: ['tied-logits', 'sample'], focus: 'tied-logits', step: 16 },
  'read-the-code': { mode: 'explore', stages: ['bpe-merge'], focus: null },
} satisfies Record<string, StepScene>;

export type StepId = keyof typeof SCENES;

/** The rail's entries: chapter starts, in reading order. */
export const TOC: { id: StepId; label: string }[] = [
  { id: 'intro', label: 'Intro' },
  { id: 'forward-pass', label: 'Forward pass' },
  { id: 'tokenize', label: 'Tokens' },
  { id: 'block', label: 'Block' },
  { id: 'logits', label: 'Logits' },
  { id: 'kv-cache', label: 'KV cache' },
  { id: 'parallelism', label: 'Parallelism' },
  { id: 'read-the-code', label: 'Code' },
];
