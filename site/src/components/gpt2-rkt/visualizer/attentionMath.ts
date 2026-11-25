import type { ExecutionTrace, HistogramItem, KvWrite, LogitTop, StageId, TensorSummary } from './traceTypes';
import { isBlockStage } from './traceTypes';

const HEADS = 12;
const QUERY_ROWS = 20;
const TRI = (QUERY_ROWS * (QUERY_ROWS + 1)) / 2;

export const HEAD_DIM = 64;
export const ATTENTION_SCALE = 1 / Math.sqrt(HEAD_DIM);

/** Causal query row for this recorded step. Prefill uses row 3; decode step s uses 3 + s; the last forward uses row 19. */
export function activeQueryRow(stepIndex: number): number {
  if (stepIndex <= 0) return 3;
  if (stepIndex >= 16) return QUERY_ROWS - 1;
  return 3 + stepIndex;
}

export function attentionOffset(layer: number, head: number, query: number, key: number): number {
  return ((layer * HEADS + head) * TRI) + ((query * (query + 1)) / 2) + key;
}

export function attentionWeight(trace: ExecutionTrace, layer: number, head: number, query: number, key: number): number {
  const pack = trace.quantized.attention;
  const index = attentionOffset(layer, head, query, key);
  return (pack.values[index] ?? 0) / pack.scale;
}

export function causalRow(trace: ExecutionTrace, layer: number, head: number, query: number): number[] {
  const weights: number[] = [];
  for (let key = 0; key <= query; key += 1) weights.push(attentionWeight(trace, layer, head, query, key));
  return weights;
}

export function strongestKey(weights: number[]): number {
  let best = 0;
  for (let i = 1; i < weights.length; i += 1) {
    if ((weights[i] ?? 0) > (weights[best] ?? 0)) best = i;
  }
  return best;
}

export interface ExactView {
  tensor: TensorSummary | null;
  kv: KvWrite | null;
  top: LogitTop[] | null;
  sample: { id: number; text: string } | null;
  missingActivation: boolean;
}

export function exactView(trace: ExecutionTrace, stepIndex: number, stage: StageId, layer: number): ExactView {
  const step = trace.exact.steps[stepIndex];
  const missingActivation = isBlockStage(stage)
    && stage !== 'block-ln1'
    && stage !== 'kv-write'
    && stage !== 'block-mlp-res'
    && stepIndex > 0
    && stepIndex < 16;
  if (!step) return { tensor: null, kv: null, top: null, sample: null, missingActivation };
  if (stage === 'kv-write') {
    return { tensor: null, kv: step.blocks[layer]?.['kv-write'] ?? null, top: null, sample: null, missingActivation: false };
  }
  if (stage === 'tied-logits') {
    const logits = step['tied-logits'] ?? null;
    return { tensor: logits, kv: null, top: logits?.top ?? null, sample: null, missingActivation: false };
  }
  if (stage === 'sample') {
    const sample = step.sample ?? null;
    return {
      tensor: null,
      kv: null,
      top: step['tied-logits']?.top ?? null,
      sample: sample ? { id: sample.id, text: sample.text } : null,
      missingActivation: false,
    };
  }
  if (stage === 'input-embed') {
    return { tensor: step['input-embed'] ?? null, kv: null, top: null, sample: null, missingActivation: false };
  }
  if (stage === 'final-ln') {
    return { tensor: step['final-ln'] ?? null, kv: null, top: null, sample: null, missingActivation: false };
  }
  if (stage === 'bpe-merge' || !isBlockStage(stage)) {
    return { tensor: null, kv: null, top: null, sample: null, missingActivation: false };
  }
  const record = step.blocks[layer]?.[stage];
  const summary = record && 'shape' in record ? record : null;
  return { tensor: summary, kv: null, top: null, sample: null, missingActivation: missingActivation && !summary };
}

export function histogramFor(
  trace: ExecutionTrace,
  stage: string,
  layer: number | null,
  phase: string,
  stepIndex: number,
): HistogramItem | null {
  return trace.quantized.histograms.items.find((item) => (
    item.stage === stage
    && item.phase === phase
    && item.step === stepIndex
    && item.layer === layer
  )) ?? null;
}

export function tokenText(trace: ExecutionTrace, index: number): string {
  return trace.run.tokens[index]?.text ?? '';
}
