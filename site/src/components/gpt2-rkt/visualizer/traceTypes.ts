/** Schema 1.0.0 execution trace. Field names match the Racket exporter. */

export const TRACE_SCHEMA = '1.0.0';
export const TRACE_COMMIT = 'ae29f0007eda76012ed772be4e54b18ecbf56939';
export const TRACE_PAIRS = 30240;
export const TRACE_OUTPUT =
  ', the world was a place of great beauty and great danger. The world was';

export const STAGE_IDS = [
  'bpe-merge',
  'input-embed',
  'block-ln1',
  'block-qkv',
  'kv-write',
  'attend',
  'block-proj-res',
  'block-ln2',
  'mlp-gelu',
  'block-mlp-res',
  'final-ln',
  'tied-logits',
  'sample',
] as const;

export type StageId = (typeof STAGE_IDS)[number];

export const BLOCK_STAGE_IDS = [
  'block-ln1',
  'block-qkv',
  'kv-write',
  'attend',
  'block-proj-res',
  'block-ln2',
  'mlp-gelu',
  'block-mlp-res',
] as const satisfies readonly StageId[];

export type BlockStageId = (typeof BLOCK_STAGE_IDS)[number];

export interface TensorSummary {
  shape: number[];
  min: number;
  max: number;
  mean: number;
  std: number;
  l2: number;
}

export interface LogitTop {
  id: number;
  text: string;
  logit: number;
  prob: number;
}

export interface TiedLogits extends TensorSummary {
  top?: LogitTop[];
}

export interface KvWrite {
  filledBefore: number;
  filledAfter: number;
  capacity: number;
  width: number;
  rowsWritten: number;
  elementBytes: number;
  strideBytes: number;
  bytesWritten: number;
  kByteOffset: number;
  vByteOffset: number;
  srcStrideElements: number;
  dstStrideElements: number;
  zeroCopy: boolean;
  layout: string;
}

export interface BlockRecord {
  layer: number;
  'block-ln1'?: TensorSummary;
  'block-qkv'?: TensorSummary;
  'kv-write'?: KvWrite;
  attend?: TensorSummary;
  'block-proj-res'?: TensorSummary;
  'block-ln2'?: TensorSummary;
  'mlp-gelu'?: TensorSummary & { cProj?: TensorSummary };
  'block-mlp-res'?: TensorSummary;
}

export interface SampleRecord {
  id: number;
  text: string;
  decoding: string;
  cumulative: string;
}

export interface ExactStep {
  phase: 'prefill' | 'decode';
  inputIds: number[];
  'input-embed'?: TensorSummary;
  blocks: BlockRecord[];
  'final-ln'?: TensorSummary;
  'tied-logits'?: TiedLogits;
  sample?: SampleRecord | null;
}

export interface BpeMerge {
  pair: [string, string];
  rank: number;
  merged: string;
  pieces: string[];
}

export interface BpeChunk {
  text: string;
  byteEncoded: string;
  merges: BpeMerge[] | null;
  pieces: string[];
  tokenIds: number[];
  tokenStrings: string[];
  cached: boolean;
}

export interface SourcePin {
  commitSha: string;
  file: string;
  line: number;
  endLine: number;
  fn: string;
  snippet: string;
}

export interface ScheduleChunk {
  i0: number;
  i1: number;
  j0: number;
  j1: number;
}

export interface ScheduleRecord {
  phase: string;
  layer: number | null;
  stage: string;
  ops: number;
  axis: 'serial' | 'row' | 'col' | string;
  processorCount: number;
  splitCount: number;
  workerCount: number;
  thresholds: { work: number; axis: number };
  chunks: ScheduleChunk[];
}

export interface StageSpan {
  phase: string | null;
  step: number | null;
  layer: number | null;
  stage: string;
  ms: number;
}

export interface HistogramItem {
  stage: string;
  layer: number | null;
  phase: string;
  step: number | null;
  bins: number;
  min: number;
  max: number;
  quantization: string;
  counts: number[];
}

export interface TokenRecord {
  id: number;
  text: string;
}

export interface ExecutionTrace {
  schemaVersion: string;
  commitSha: string;
  model: {
    repo: string;
    revision: string;
    nLayer: number;
    nHead: number;
    nEmbd: number;
    vocabSize: number;
    nCtx: number;
  };
  run: {
    prompt: string;
    maxNewTokens: number;
    decoding: string;
    promptTokenIds: number[];
    generatedTokenIds: number[];
    allTokenIds: number[];
    tokens: TokenRecord[];
    outputText: string;
  };
  sourceMap: Record<StageId, SourcePin>;
  coverage: {
    fullBlockSteps: string;
    everyStep: string[];
    attention: string;
    scheduling: string;
    timings: string;
  };
  exact: {
    bpe: { chunks: BpeChunk[] };
    steps: ExactStep[];
    attention: { pairs: number; maxRowSumError: number } | null;
  };
  quantized: {
    attention: {
      layers: number;
      heads: number;
      queryRows: number;
      pairs: number;
      causal: boolean;
      scale: number;
      decimals: number;
      encoding: string;
      order: string[];
      values: number[];
      exactMaxRowSumError: number;
      maxRowSumError: number;
    };
    histograms: {
      bins: number;
      quantization: string;
      items: HistogramItem[];
    };
  };
  scheduling: ScheduleRecord[];
  timings: {
    clock: string;
    prefillMs: number | null;
    decodeMs: number | null;
    totalMs: number | null;
    stages: StageSpan[];
  };
}

export function isStageId(id: string): id is StageId {
  return (STAGE_IDS as readonly string[]).includes(id);
}

export function isBlockStage(id: string): id is BlockStageId {
  return (BLOCK_STAGE_IDS as readonly string[]).includes(id);
}
