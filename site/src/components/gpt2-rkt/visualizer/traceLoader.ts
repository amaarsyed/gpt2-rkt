import {
  STAGE_IDS,
  TRACE_COMMIT,
  TRACE_OUTPUT,
  TRACE_PAIRS,
  TRACE_SCHEMA,
  type BpeChunk,
  type BlockRecord,
  type ExactStep,
  type ExecutionTrace,
  type HistogramItem,
  type LogitTop,
  type ScheduleRecord,
  type SourcePin,
  type StageId,
  type StageSpan,
  type TensorSummary,
  isStageId,
} from './traceTypes';

export const TRACE_URL = '/traces/once-upon-a-time-16.trace.json';

let pending: Promise<ExecutionTrace> | null = null;

/** One network read for the lifetime of the page. The visualizer mounts before this runs. */
export function loadTrace(url = TRACE_URL): Promise<ExecutionTrace> {
  if (!pending) pending = fetchTrace(url);
  return pending;
}

async function fetchTrace(url: string): Promise<ExecutionTrace> {
  const response = await fetch(url);
  const type = response.headers.get('content-type') ?? '';
  if (!response.ok || !type.includes('json')) {
    throw new Error('The recorded trace is not on this page yet.');
  }
  const json: unknown = await response.json();
  return validateTrace(json);
}

function fail(message: string): never {
  throw new Error(`Trace ${TRACE_SCHEMA}: ${message}`);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function num(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) fail(`${label} is not a finite number`);
  return value;
}

function str(value: unknown, label: string): string {
  if (typeof value !== 'string') fail(`${label} is not a string`);
  return value;
}

function intList(value: unknown, label: string): number[] {
  if (!Array.isArray(value) || value.some((item) => typeof item !== 'number')) fail(`${label} is not a number list`);
  return value as number[];
}

function optionalSummary(value: unknown): TensorSummary | undefined {
  if (value === undefined || value === null) return undefined;
  if (!isRecord(value)) fail('tensor summary is not an object');
  return {
    shape: intList(value.shape, 'summary.shape'),
    min: num(value.min, 'summary.min'),
    max: num(value.max, 'summary.max'),
    mean: num(value.mean, 'summary.mean'),
    std: num(value.std, 'summary.std'),
    l2: num(value.l2, 'summary.l2'),
  };
}

function nullableNumber(value: unknown, label: string): number | null {
  if (value === null) return null;
  return num(value, label);
}

export function validateTrace(json: unknown): ExecutionTrace {
  if (!isRecord(json)) fail('document is not an object');
  if (json.schemaVersion !== TRACE_SCHEMA) fail(`schemaVersion is ${String(json.schemaVersion)}`);
  if (json.commitSha !== TRACE_COMMIT) fail(`commitSha is ${String(json.commitSha)}`);

  const model = isRecord(json.model) ? json.model : fail('model missing');
  const run = isRecord(json.run) ? json.run : fail('run missing');
  if (run.outputText !== TRACE_OUTPUT) fail('outputText does not match the verified run');
  if (run.maxNewTokens !== 16 || run.decoding !== 'greedy') fail('run is not the 16-token greedy pass');

  const sourceRaw = isRecord(json.sourceMap) ? json.sourceMap : fail('sourceMap missing');
  const sourceMap = {} as Record<StageId, SourcePin>;
  for (const id of STAGE_IDS) {
    const pin = sourceRaw[id];
    if (!isRecord(pin)) fail(`sourceMap.${id} missing`);
    sourceMap[id] = {
      commitSha: str(pin.commitSha, `sourceMap.${id}.commitSha`),
      file: str(pin.file, `sourceMap.${id}.file`),
      line: num(pin.line, `sourceMap.${id}.line`),
      endLine: num(pin.endLine, `sourceMap.${id}.endLine`),
      fn: str(pin.fn, `sourceMap.${id}.fn`),
      snippet: str(pin.snippet, `sourceMap.${id}.snippet`),
    };
  }

  const exact = isRecord(json.exact) ? json.exact : fail('exact missing');
  const bpe = isRecord(exact.bpe) ? exact.bpe : fail('exact.bpe missing');
  const chunks = parseChunks(bpe.chunks);
  if (chunks.length !== 4) fail(`expected 4 BPE chunks, found ${chunks.length}`);
  const steps = parseSteps(exact.steps);
  if (steps.length !== 17) fail(`expected 17 steps, found ${steps.length}`);
  if (steps[0]?.phase !== 'prefill') fail('step 0 is not prefill');
  if (!steps.slice(1).every((step) => step.phase === 'decode')) fail('steps 1..16 are not all decode');

  const quantized = isRecord(json.quantized) ? json.quantized : fail('quantized missing');
  const attention = isRecord(quantized.attention) ? quantized.attention : fail('attention missing');
  if (attention.pairs !== TRACE_PAIRS) fail(`attention pairs are ${String(attention.pairs)}`);
  const values = intList(attention.values, 'attention.values');
  if (values.length !== TRACE_PAIRS) fail(`attention values length ${values.length}`);

  const histograms = isRecord(quantized.histograms) ? quantized.histograms : fail('histograms missing');
  const items = parseHistograms(histograms.items);
  if (items.length !== 603) fail(`expected 603 histogram items, found ${items.length}`);

  const scheduling = parseSchedules(json.scheduling);
  if (scheduling.length !== 833) fail(`expected 833 schedules, found ${scheduling.length}`);

  const timingsRaw = isRecord(json.timings) ? json.timings : fail('timings missing');
  const stages = parseSpans(timingsRaw.stages);
  if (stages.length !== 1700) fail(`expected 1700 stage spans, found ${stages.length}`);

  const coverage = isRecord(json.coverage) ? json.coverage : fail('coverage missing');

  const promptTokenIds = intList(run.promptTokenIds, 'promptTokenIds');
  const generatedTokenIds = intList(run.generatedTokenIds, 'generatedTokenIds');
  const allTokenIds = intList(run.allTokenIds, 'allTokenIds');
  if (!Array.isArray(run.tokens)) fail('tokens missing');

  return {
    schemaVersion: TRACE_SCHEMA,
    commitSha: TRACE_COMMIT,
    model: {
      repo: str(model.repo, 'model.repo'),
      revision: str(model.revision, 'model.revision'),
      nLayer: num(model.nLayer, 'model.nLayer'),
      nHead: num(model.nHead, 'model.nHead'),
      nEmbd: num(model.nEmbd, 'model.nEmbd'),
      vocabSize: num(model.vocabSize, 'model.vocabSize'),
      nCtx: num(model.nCtx, 'model.nCtx'),
    },
    run: {
      prompt: str(run.prompt, 'run.prompt'),
      maxNewTokens: 16,
      decoding: 'greedy',
      promptTokenIds,
      generatedTokenIds,
      allTokenIds,
      tokens: (run.tokens as unknown[]).map((token, index) => {
        if (!isRecord(token)) fail(`token ${index} is not an object`);
        return { id: num(token.id, `token ${index} id`), text: str(token.text, `token ${index} text`) };
      }),
      outputText: TRACE_OUTPUT,
    },
    sourceMap,
    coverage: {
      fullBlockSteps: str(coverage.fullBlockSteps, 'coverage.fullBlockSteps'),
      everyStep: Array.isArray(coverage.everyStep) ? coverage.everyStep.map((item) => str(item, 'coverage.everyStep')) : [],
      attention: str(coverage.attention, 'coverage.attention'),
      scheduling: str(coverage.scheduling, 'coverage.scheduling'),
      timings: str(coverage.timings, 'coverage.timings'),
    },
    exact: {
      bpe: { chunks },
      steps,
      attention: isRecord(exact.attention)
        ? { pairs: num(exact.attention.pairs, 'exact.attention.pairs'), maxRowSumError: num(exact.attention.maxRowSumError, 'exact.attention.maxRowSumError') }
        : null,
    },
    quantized: {
      attention: {
        layers: num(attention.layers, 'attention.layers'),
        heads: num(attention.heads, 'attention.heads'),
        queryRows: num(attention.queryRows, 'attention.queryRows'),
        pairs: TRACE_PAIRS,
        causal: attention.causal === true,
        scale: num(attention.scale, 'attention.scale'),
        decimals: num(attention.decimals, 'attention.decimals'),
        encoding: str(attention.encoding, 'attention.encoding'),
        order: Array.isArray(attention.order) ? attention.order.map((item) => str(item, 'order')) : [],
        values,
        exactMaxRowSumError: num(attention.exactMaxRowSumError, 'exactMaxRowSumError'),
        maxRowSumError: num(attention.maxRowSumError, 'maxRowSumError'),
      },
      histograms: {
        bins: num(histograms.bins, 'histograms.bins'),
        quantization: str(histograms.quantization, 'histograms.quantization'),
        items,
      },
    },
    scheduling,
    timings: {
      clock: str(timingsRaw.clock, 'timings.clock'),
      prefillMs: nullableNumber(timingsRaw.prefillMs, 'prefillMs'),
      decodeMs: nullableNumber(timingsRaw.decodeMs, 'decodeMs'),
      totalMs: nullableNumber(timingsRaw.totalMs, 'totalMs'),
      stages,
    },
  };
}

function parseChunks(value: unknown): BpeChunk[] {
  if (!Array.isArray(value)) fail('bpe chunks missing');
  return value.map((chunk, index) => {
    if (!isRecord(chunk)) fail(`bpe chunk ${index} is not an object`);
    const merges = chunk.merges === null || chunk.merges === undefined
      ? null
      : Array.isArray(chunk.merges)
        ? chunk.merges.map((merge) => {
            if (!isRecord(merge)) fail('merge is not an object');
            const pair = merge.pair;
            if (!Array.isArray(pair) || pair.length !== 2) fail('merge.pair');
            return {
              pair: [str(pair[0], 'pair0'), str(pair[1], 'pair1')] as [string, string],
              rank: num(merge.rank, 'rank'),
              merged: str(merge.merged, 'merged'),
              pieces: Array.isArray(merge.pieces) ? merge.pieces.map((piece) => str(piece, 'piece')) : [],
            };
          })
        : fail('merges is not a list');
    return {
      text: str(chunk.text, 'chunk.text'),
      byteEncoded: str(chunk.byteEncoded, 'byteEncoded'),
      merges,
      pieces: Array.isArray(chunk.pieces) ? chunk.pieces.map((piece) => str(piece, 'pieces')) : [],
      tokenIds: intList(chunk.tokenIds, 'tokenIds'),
      tokenStrings: Array.isArray(chunk.tokenStrings) ? chunk.tokenStrings.map((piece) => str(piece, 'tokenStrings')) : [],
      cached: chunk.cached === true,
    };
  });
}

function parseSteps(value: unknown): ExactStep[] {
  if (!Array.isArray(value)) fail('steps missing');
  return value.map((step, index) => {
    if (!isRecord(step)) fail(`step ${index} is not an object`);
    const phase = step.phase === 'prefill' || step.phase === 'decode' ? step.phase : fail(`step ${index} phase`);
    const blocks = Array.isArray(step.blocks) ? step.blocks.map((block, layer) => parseBlock(block, layer)) : fail('blocks');
    const tied = optionalSummary(step['tied-logits']);
    let top: LogitTop[] | undefined;
    if (isRecord(step['tied-logits']) && Array.isArray(step['tied-logits'].top)) {
      top = step['tied-logits'].top.map((entry) => {
        if (!isRecord(entry)) fail('logit top');
        return {
          id: num(entry.id, 'top.id'),
          text: str(entry.text, 'top.text'),
          logit: num(entry.logit, 'top.logit'),
          prob: num(entry.prob, 'top.prob'),
        };
      });
    }
    const sampleRaw = step.sample;
    const sample = sampleRaw === null || sampleRaw === undefined
      ? null
      : isRecord(sampleRaw)
        ? {
            id: num(sampleRaw.id, 'sample.id'),
            text: str(sampleRaw.text, 'sample.text'),
            decoding: str(sampleRaw.decoding, 'sample.decoding'),
            cumulative: str(sampleRaw.cumulative, 'sample.cumulative'),
          }
        : fail('sample');
    return {
      phase,
      inputIds: intList(step.inputIds, 'inputIds'),
      'input-embed': optionalSummary(step['input-embed']),
      blocks,
      'final-ln': optionalSummary(step['final-ln']),
      'tied-logits': tied ? { ...tied, ...(top ? { top } : {}) } : top ? { shape: [], min: 0, max: 0, mean: 0, std: 0, l2: 0, top } : undefined,
      sample,
    };
  });
}

function parseBlock(value: unknown, index: number): BlockRecord {
  if (!isRecord(value)) fail(`block ${index} is not an object`);
  const gelu = optionalSummary(value['mlp-gelu']);
  const cProj = isRecord(value['mlp-gelu']) ? optionalSummary(value['mlp-gelu'].cProj) : undefined;
  const kv = value['kv-write'];
  return {
    layer: num(value.layer, 'block.layer'),
    'block-ln1': optionalSummary(value['block-ln1']),
    'block-qkv': optionalSummary(value['block-qkv']),
    'kv-write': isRecord(kv)
      ? {
          filledBefore: num(kv.filledBefore, 'filledBefore'),
          filledAfter: num(kv.filledAfter, 'filledAfter'),
          capacity: num(kv.capacity, 'capacity'),
          width: num(kv.width, 'width'),
          rowsWritten: num(kv.rowsWritten, 'rowsWritten'),
          elementBytes: num(kv.elementBytes, 'elementBytes'),
          strideBytes: num(kv.strideBytes, 'strideBytes'),
          bytesWritten: num(kv.bytesWritten, 'bytesWritten'),
          kByteOffset: num(kv.kByteOffset, 'kByteOffset'),
          vByteOffset: num(kv.vByteOffset, 'vByteOffset'),
          srcStrideElements: num(kv.srcStrideElements, 'srcStrideElements'),
          dstStrideElements: num(kv.dstStrideElements, 'dstStrideElements'),
          zeroCopy: kv.zeroCopy === true,
          layout: str(kv.layout, 'layout'),
        }
      : undefined,
    attend: optionalSummary(value.attend),
    'block-proj-res': optionalSummary(value['block-proj-res']),
    'block-ln2': optionalSummary(value['block-ln2']),
    'mlp-gelu': gelu ? { ...gelu, ...(cProj ? { cProj } : {}) } : undefined,
    'block-mlp-res': optionalSummary(value['block-mlp-res']),
  };
}

function parseHistograms(value: unknown): HistogramItem[] {
  if (!Array.isArray(value)) fail('histogram items missing');
  return value.map((item) => {
    if (!isRecord(item)) fail('histogram item');
    return {
      stage: str(item.stage, 'hist.stage'),
      layer: item.layer === null ? null : num(item.layer, 'hist.layer'),
      phase: str(item.phase, 'hist.phase'),
      step: item.step === null ? null : num(item.step, 'hist.step'),
      bins: num(item.bins, 'hist.bins'),
      min: num(item.min, 'hist.min'),
      max: num(item.max, 'hist.max'),
      quantization: str(item.quantization, 'hist.quantization'),
      counts: intList(item.counts, 'hist.counts'),
    };
  });
}

function parseSchedules(value: unknown): ScheduleRecord[] {
  if (!Array.isArray(value)) fail('scheduling missing');
  return value.map((row) => {
    if (!isRecord(row)) fail('schedule row');
    const thresholds = isRecord(row.thresholds) ? row.thresholds : fail('thresholds');
    const chunks = Array.isArray(row.chunks) ? row.chunks : fail('chunks');
    return {
      phase: str(row.phase, 'schedule.phase'),
      layer: row.layer === null ? null : num(row.layer, 'schedule.layer'),
      stage: str(row.stage, 'schedule.stage'),
      ops: num(row.ops, 'ops'),
      axis: str(row.axis, 'axis'),
      processorCount: num(row.processorCount, 'processorCount'),
      splitCount: num(row.splitCount, 'splitCount'),
      workerCount: num(row.workerCount, 'workerCount'),
      thresholds: { work: num(thresholds.work, 'thresholds.work'), axis: num(thresholds.axis, 'thresholds.axis') },
      chunks: chunks.map((chunk) => {
        if (!isRecord(chunk)) fail('chunk');
        return { i0: num(chunk.i0, 'i0'), i1: num(chunk.i1, 'i1'), j0: num(chunk.j0, 'j0'), j1: num(chunk.j1, 'j1') };
      }),
    };
  });
}

function parseSpans(value: unknown): StageSpan[] {
  if (!Array.isArray(value)) fail('timing stages missing');
  return value.map((span) => {
    if (!isRecord(span)) fail('span');
    const stage = str(span.stage, 'span.stage');
    return {
      phase: span.phase === null ? null : str(span.phase, 'span.phase'),
      step: span.step === null ? null : num(span.step, 'span.step'),
      layer: span.layer === null ? null : num(span.layer, 'span.layer'),
      stage: isStageId(stage) ? stage : stage,
      ms: num(span.ms, 'span.ms'),
    };
  });
}
