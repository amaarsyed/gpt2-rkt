import {
  BLOCK_STAGE_IDS,
  STAGE_IDS,
  type BlockStageId,
  type BpeMerge,
  type ExecutionTrace,
  type KvWrite,
  type LogitTop,
  type ScheduleRecord,
  type SourcePin,
  type StageId,
  isBlockStage,
  isStageId,
} from './traceTypes';

export type VizMode = 'explore' | 'execute';
export type Speed = 1 | 2 | 4;

export interface AttentionPair {
  q: number;
  k: number;
}

export interface ExecutionState {
  mode: VizMode;
  /** 0 is prefill. 1..15 sample. 16 is the last forward, with no new sample. */
  stepIndex: number;
  activeStageId: StageId;
  activeLayer: number;
  /** 0..11. Used by the Attend inspection slice. */
  head: number;
  /** 0 dots, 1 mask, 2 softmax, 3 weighted values. */
  mathPhase: 0 | 1 | 2 | 3;
  inspect: boolean;
  pair: AttentionPair | null;
  hoverPair: AttentionPair | null;
  selectedNodeId: string | null;
  hoveredNodeId: string | null;
  playing: boolean;
  speed: Speed;
}

export const LAST_STEP = 16;
export const LAST_LAYER = 11;

export function initialState(): ExecutionState {
  return {
    mode: 'explore',
    stepIndex: 0,
    activeStageId: 'bpe-merge',
    activeLayer: 0,
    head: 0,
    mathPhase: 2,
    inspect: false,
    pair: null,
    hoverPair: null,
    selectedNodeId: null,
    hoveredNodeId: null,
    playing: false,
    speed: 1,
  };
}

export function stageOrder(state: ExecutionState, trace: ExecutionTrace): StageId[] {
  const step = trace.exact.steps[state.stepIndex];
  const forward: StageId[] = [
    'input-embed',
    ...BLOCK_STAGE_IDS,
    'final-ln',
    'tied-logits',
  ];
  const head: StageId[] = step?.phase === 'prefill' ? ['bpe-merge', ...forward] : forward;
  if (step?.sample) return [...head, 'sample'];
  return head;
}

export function clampStep(index: number): number {
  return Math.max(0, Math.min(LAST_STEP, index));
}

export function clampLayer(layer: number): number {
  return Math.max(0, Math.min(LAST_LAYER, layer));
}

/** Advance one stage. Crossing the end of a step lands on the next step's first stage. */
export function advance(state: ExecutionState, trace: ExecutionTrace): ExecutionState {
  const order = stageOrder(state, trace);
  const at = Math.max(0, order.indexOf(state.activeStageId));
  if (at < order.length - 1) {
    const next = order[at + 1] ?? state.activeStageId;
    return { ...state, activeStageId: next, activeLayer: isBlockStage(next) ? state.activeLayer : state.activeLayer };
  }
  if (state.stepIndex >= LAST_STEP) return { ...state, playing: false };
  const stepIndex = state.stepIndex + 1;
  const nextOrder = stageOrder({ ...state, stepIndex }, trace);
  return { ...state, stepIndex, activeStageId: nextOrder[0] ?? 'input-embed' };
}

export function retreat(state: ExecutionState, trace: ExecutionTrace): ExecutionState {
  const order = stageOrder(state, trace);
  const at = Math.max(0, order.indexOf(state.activeStageId));
  if (at > 0) return { ...state, activeStageId: order[at - 1] ?? state.activeStageId, playing: false };
  if (state.stepIndex <= 0) return { ...state, playing: false };
  const stepIndex = state.stepIndex - 1;
  const prev = stageOrder({ ...state, stepIndex }, trace);
  return { ...state, stepIndex, activeStageId: prev[prev.length - 1] ?? 'sample', playing: false };
}

export function jumpStep(state: ExecutionState, trace: ExecutionTrace, stepIndex: number): ExecutionState {
  const next = clampStep(stepIndex);
  const order = stageOrder({ ...state, stepIndex: next }, trace);
  const stage = order.includes(state.activeStageId) ? state.activeStageId : (order[0] ?? 'input-embed');
  return { ...state, stepIndex: next, activeStageId: stage, playing: false };
}

export interface TokenRibbon {
  prompt: string;
  generated: { text: string; id: number; lit: boolean }[];
}

export function tokenRibbon(trace: ExecutionTrace, stepIndex: number): TokenRibbon {
  const promptCount = trace.run.promptTokenIds.length;
  const generated = trace.run.tokens.slice(promptCount);
  let lit = 0;
  for (let i = 0; i <= stepIndex && i < trace.exact.steps.length; i += 1) {
    if (trace.exact.steps[i]?.sample) lit += 1;
  }
  return {
    prompt: trace.run.prompt,
    generated: generated.map((token, index) => ({ text: token.text, id: token.id, lit: index < lit })),
  };
}

export interface StepFacts {
  phase: 'prefill' | 'decode';
  stepIndex: number;
  tokenId: number | null;
  tokenText: string | null;
  top: LogitTop | null;
  kv: KvWrite | null;
  schedule: ScheduleRecord | null;
  stageMs: number | null;
  stageId: StageId;
  layer: number;
  source: SourcePin | null;
  sentence: string;
}

const scheduleBuckets = new WeakMap<ExecutionTrace, Map<string, ScheduleRecord[][]>>();

function scheduleKey(phase: string, stage: string, layer: number | null): string {
  return `${phase}|${stage}|${layer ?? 'root'}`;
}

function phaseOrdinal(trace: ExecutionTrace, phase: string, stepIndex: number): number {
  let seen = 0;
  for (let i = 0; i < stepIndex; i += 1) {
    if (trace.exact.steps[i]?.phase === phase) seen += 1;
  }
  return seen;
}

function schedulesFor(trace: ExecutionTrace, phase: string, stage: string, layer: number | null, stepIndex: number): ScheduleRecord[] {
  let buckets = scheduleBuckets.get(trace);
  if (!buckets) {
    buckets = new Map();
    const counts = new Map<string, number>();
    for (const row of trace.scheduling) {
      const key = scheduleKey(row.phase, row.stage, row.layer);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    const cursor = new Map<string, number>();
    for (const row of trace.scheduling) {
      const key = scheduleKey(row.phase, row.stage, row.layer);
      const total = counts.get(key) ?? 1;
      const forwards = phaseForwards(trace, row.phase);
      const perStep = forwards > 0 && total % forwards === 0 ? total / forwards : 1;
      const seen = cursor.get(key) ?? 0;
      cursor.set(key, seen + 1);
      const bucket = Math.min(forwards - 1, Math.floor(seen / perStep));
      const list = buckets.get(key) ?? Array.from({ length: forwards }, () => [] as ScheduleRecord[]);
      list[bucket]?.push(row);
      buckets.set(key, list);
    }
    scheduleBuckets.set(trace, buckets);
  }
  const list = buckets.get(scheduleKey(phase, stage, layer)) ?? [];
  const ordinal = phaseOrdinal(trace, phase, stepIndex);
  return list[Math.min(ordinal, list.length - 1)] ?? [];
}

function phaseForwards(trace: ExecutionTrace, phase: string): number {
  return trace.exact.steps.filter((step) => step.phase === phase).length || 1;
}

export function stepFacts(trace: ExecutionTrace, state: ExecutionState): StepFacts {
  const step = trace.exact.steps[state.stepIndex];
  const phase = step?.phase ?? 'decode';
  const layer = clampLayer(state.activeLayer);
  const stageId = state.activeStageId;
  const block = step?.blocks[layer];
  const kv = block?.['kv-write'] ?? null;
  const top = step?.['tied-logits']?.top?.[0] ?? null;
  const tokenId = tokenIdForStep(trace, state.stepIndex);
  const tokenText = tokenId === null ? null : (trace.run.tokens.find((token) => token.id === tokenId)?.text ?? null);
  const scheduleLayer = isBlockStage(stageId) ? layer : null;
  const scheduleStage = stageId === 'kv-write' || stageId === 'attend' || stageId === 'sample' || stageId === 'bpe-merge' || stageId === 'final-ln'
    ? nearestScheduleStage(stageId)
    : stageId;
  const matches = schedulesFor(trace, phase, scheduleStage, scheduleLayer, state.stepIndex);
  const schedule = matches.find((row) => row.axis !== 'serial') ?? matches[0] ?? null;
  const stageMs = spanMs(trace, phase, state.stepIndex, stageId, isBlockStage(stageId) ? layer : null);
  const source = isStageId(stageId) ? trace.sourceMap[stageId] : null;
  return {
    phase,
    stepIndex: state.stepIndex,
    tokenId,
    tokenText,
    top,
    kv,
    schedule,
    stageMs,
    stageId,
    layer,
    source,
    sentence: describe({ phase, stepIndex: state.stepIndex, tokenId, tokenText, top, kv, schedule, stageMs, stageId, layer }),
  };
}

function nearestScheduleStage(stage: StageId): string {
  if (stage === 'kv-write' || stage === 'attend') return 'block-qkv';
  if (stage === 'sample') return 'tied-logits';
  if (stage === 'bpe-merge') return 'input-embed';
  return stage;
}

function tokenIdForStep(trace: ExecutionTrace, stepIndex: number): number | null {
  const step = trace.exact.steps[stepIndex];
  if (!step) return null;
  if (step.sample) return step.sample.id;
  const last = step.inputIds[step.inputIds.length - 1];
  return last ?? null;
}

function spanMs(trace: ExecutionTrace, phase: string, stepIndex: number, stage: string, layer: number | null): number | null {
  const sameLayer = (row: { layer: number | null }) => (layer === null ? row.layer === null : row.layer === layer);
  const span = trace.timings.stages.find((row) => (
    row.phase === phase && row.stage === stage && row.step === stepIndex && sameLayer(row)
  ));
  if (span) return span.ms;
  if (stepIndex !== 0) return null;
  const root = trace.timings.stages.find((row) => (
    row.phase === phase && row.stage === stage && row.step === null && sameLayer(row)
  ));
  return root ? root.ms : null;
}

function describe(facts: Omit<StepFacts, 'source' | 'sentence'>): string {
  const where = facts.phase === 'prefill' ? 'Prefill' : `Decode ${facts.stepIndex}`;
  const token = facts.tokenId === null
    ? ''
    : ` Token ${facts.tokenId}${facts.tokenText !== null ? ` “${facts.tokenText}”` : ''}.`;
  const guess = facts.top
    ? ` Top guess “${facts.top.text}” at ${(facts.top.prob * 100).toFixed(1)}%.`
    : '';
  const memory = facts.kv
    ? ` KV cache ${facts.kv.filledAfter}/${facts.kv.capacity} cache rows, ${facts.kv.bytesWritten.toLocaleString('en-US')} bytes written.`
    : '';
  const split = facts.schedule
    ? ` Future policy: ${facts.schedule.axis === 'serial' ? 'stays serial' : `splits on ${facts.schedule.axis}`} across ${facts.schedule.workerCount} workers (${facts.schedule.splitCount} chunks).`
    : '';
  const clock = facts.stageMs === null
    ? ''
    : ` Measured: ${stageLabel(facts.stageId)}${isBlockStage(facts.stageId) ? ` L${facts.layer}` : ''} ${facts.stageMs.toFixed(2)} ms.`;
  const tail = facts.stepIndex === LAST_STEP && !facts.top ? ' Last forward, so the twentieth attention row is on record. No new sample.' : '';
  return `${where}.${token}${guess}${memory}${split}${clock}${tail}`.replace(/\s+/g, ' ').trim();
}

export function stageLabel(id: string): string {
  switch (id) {
    case 'bpe-merge': return 'BPE';
    case 'input-embed': return 'Embed';
    case 'block-ln1': return 'LN1';
    case 'block-qkv': return 'QKV';
    case 'kv-write': return 'KV Write';
    case 'attend': return 'Attend';
    case 'block-proj-res': return 'Proj+Res';
    case 'block-ln2': return 'LN2';
    case 'mlp-gelu': return 'MLP/GELU';
    case 'block-mlp-res': return 'MLP+Res';
    case 'final-ln': return 'Final LN';
    case 'tied-logits': return 'Logits';
    case 'sample': return 'Sample';
    default: return id;
  }
}

export function mergesFor(trace: ExecutionTrace): BpeMerge[] {
  return trace.exact.bpe.chunks.flatMap((chunk) => chunk.merges ?? []);
}

export function promptTokenCount(trace: ExecutionTrace): number {
  return trace.run.promptTokenIds.length;
}

export const FORWARD_STAGES: readonly StageId[] = STAGE_IDS;

export function blockStageIndex(id: StageId): number {
  return (BLOCK_STAGE_IDS as readonly string[]).indexOf(id);
}

export type { BlockStageId };
