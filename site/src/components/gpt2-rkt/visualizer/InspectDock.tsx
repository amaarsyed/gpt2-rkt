import { useMemo, useState } from 'react';
import { AttentionHeatmap } from './AttentionHeatmap';
import {
  ATTENTION_SCALE,
  HEAD_DIM,
  activeQueryRow,
  attentionWeight,
  causalRow,
  exactView,
  histogramFor,
  strongestKey,
  tokenText,
} from './attentionMath';
import type { ExecutionState } from './executionState';
import { stageLabel } from './executionState';
import { isBlockStage } from './traceTypes';
import type { ExecutionTrace, HistogramItem, ScheduleRecord, TensorSummary } from './traceTypes';

const PHASES = ['Q · Kᵀ / √64', 'Causal mask', 'Softmax', 'Weighted V mix'] as const;

interface Props {
  trace: ExecutionTrace;
  state: ExecutionState;
  schedule: ScheduleRecord | null;
  stageMs: number | null;
  onHead: (head: number) => void;
  onPhase: (phase: 0 | 1 | 2 | 3) => void;
  onPair: (query: number, keyIndex: number) => void;
  onHoverPair: (query: number | null, keyIndex: number | null) => void;
}

function num(value: number, digits = 3): string {
  return value.toFixed(digits);
}

function TensorStats({ tensor }: { tensor: TensorSummary }) {
  return (
    <dl className="viz-stats">
      <div><dt>shape</dt><dd>{tensor.shape.join('×')}</dd></div>
      <div><dt>L2</dt><dd>{num(tensor.l2)}</dd></div>
      <div><dt>min</dt><dd>{num(tensor.min)}</dd></div>
      <div><dt>max</dt><dd>{num(tensor.max)}</dd></div>
      <div><dt>mean</dt><dd>{num(tensor.mean)}</dd></div>
      <div><dt>std</dt><dd>{num(tensor.std)}</dd></div>
    </dl>
  );
}

function Histogram({ item }: { item: HistogramItem }) {
  const peak = Math.max(1, ...item.counts);
  return (
    <figure className="viz-hist">
      <svg viewBox={`0 0 ${item.counts.length} 36`} role="img" aria-label={`${item.bins}-bin quantized histogram`}>
        {item.counts.map((count, index) => {
          const height = (count / peak) * 32;
          return <rect key={index} x={index + 0.12} y={34 - height} width={0.76} height={height} />;
        })}
      </svg>
      <figcaption>{item.bins} uniform bins · {num(item.min, 2)} to {num(item.max, 2)} · {item.quantization}</figcaption>
    </figure>
  );
}

export function InspectDock({ trace, state, schedule, stageMs, onHead, onPhase, onPair, onHoverPair }: Props) {
  const [hover, setHover] = useState<{ q: number; k: number } | null>(null);
  const queryRow = activeQueryRow(state.stepIndex);
  const weights = useMemo(
    () => causalRow(trace, state.activeLayer, state.head, queryRow),
    [trace, state.activeLayer, state.head, queryRow],
  );
  const picked = state.pair && state.pair.q <= queryRow && state.pair.k <= state.pair.q
    ? state.pair
    : { q: queryRow, k: strongestKey(weights) };
  const shown = hover ?? picked;
  const weight = attentionWeight(trace, state.activeLayer, state.head, shown.q, shown.k);
  const rowSum = weights.reduce((sum, value) => sum + value, 0);
  const exact = exactView(trace, state.stepIndex, state.activeStageId, state.activeLayer);
  const layerForHist = state.activeStageId === 'tied-logits' || state.activeStageId === 'final-ln' || state.activeStageId === 'input-embed' || state.activeStageId === 'sample' || state.activeStageId === 'bpe-merge'
    ? null
    : state.activeLayer;
  const phase = trace.exact.steps[state.stepIndex]?.phase ?? 'decode';
  const histogram = histogramFor(trace, state.activeStageId, layerForHist, phase, state.stepIndex);
  const attend = state.activeStageId === 'attend';
  const layered = isBlockStage(state.activeStageId);

  return (
    <aside className="viz-dock" data-no-pop="" data-lenis-prevent="">
      <header className="viz-dock-title">
        <strong>{stageLabel(state.activeStageId)}</strong>
        {layered ? <span>L{state.activeLayer}{attend ? ` · H${state.head}` : ''}</span> : null}
      </header>

      {attend ? (
        <>
          <div className="viz-heads" role="group" aria-label="Attention head" data-no-pop="">
            {Array.from({ length: 12 }, (_, head) => (
              <button key={head} type="button" data-no-pop="" aria-pressed={state.head === head} onClick={() => onHead(head)}>
                H{head}
              </button>
            ))}
          </div>
          <div className="viz-phases" role="group" aria-label="Attention math" data-no-pop="">
            {PHASES.map((label, index) => (
              <button
                key={label}
                type="button"
                data-no-pop=""
                aria-pressed={state.mathPhase === index}
                onClick={() => onPhase(index as 0 | 1 | 2 | 3)}
              >
                {index + 1} {label}
              </button>
            ))}
          </div>
          <p className="viz-eq">
            {state.mathPhase === 0 ? `1/√${HEAD_DIM} = ${ATTENTION_SCALE.toFixed(3)} over d_head = ${HEAD_DIM}. Recorded weights are post-softmax, quantized to 1e-4.` : null}
            {state.mathPhase === 1 ? `Causal mask · k > q is −∞. The upper triangle never mixes.` : null}
            {state.mathPhase === 2 ? `Softmax over 0..${queryRow} · sum ${rowSum.toFixed(4)}` : null}
            {state.mathPhase === 3 ? `Σ wₖ v[k] into d_head = ${HEAD_DIM}` : null}
          </p>
          <p className="viz-badge">Quantized (1e-4, scale: {trace.quantized.attention.scale})</p>
          <div className="viz-math">
          <div className="viz-math-fig">
          <AttentionHeatmap
            trace={trace}
            layer={state.activeLayer}
            head={state.head}
            queryRow={queryRow}
            query={shown.q}
            keyIndex={shown.k}
            phase={state.mathPhase}
            onPick={onPair}
            onHover={(q, k) => {
              setHover(q === null || k === null ? null : { q, k });
              onHoverPair(q, k);
            }}
          />
          {state.mathPhase === 3 ? (
            <ol className="viz-contrib">
              {weights
                .map((value, key) => ({ key, value, text: tokenText(trace, key).trim() || '·' }))
                .sort((a, b) => b.value - a.value)
                .slice(0, 5)
                .map((row) => {
                  const peak = Math.max(...weights, 1e-6);
                  return (
                    <li key={row.key} data-on={row.key === shown.k ? '' : undefined}>
                      <span>{row.text}</span>
                      <span className="viz-top-bar"><i style={{ width: `${Math.max(4, (row.value / peak) * 100)}%` }} /></span>
                      <span>{(row.value * 100).toFixed(1)}%</span>
                    </li>
                  );
                })}
            </ol>
          ) : null}
          <p className="viz-pair">
            ({shown.q}, {shown.k}) {weight.toFixed(4)} · “{tokenText(trace, shown.q)}” ← “{tokenText(trace, shown.k)}”
          </p>
          </div>
      <section className="viz-inspector is-aside" aria-label="Tensor and runtime">
        <div>
          <p className="viz-badge">Exact recorded</p>
          {exact.kv ? (
            <dl className="viz-stats">
              <div><dt>rows</dt><dd>{exact.kv.filledBefore} → {exact.kv.filledAfter} / {exact.kv.capacity} cache rows</dd></div>
              <div><dt>stride</dt><dd>{exact.kv.strideBytes} B</dd></div>
              <div><dt>K offset</dt><dd>{exact.kv.kByteOffset}</dd></div>
              <div><dt>V offset</dt><dd>{exact.kv.vByteOffset}</dd></div>
              <div><dt>written</dt><dd>{exact.kv.bytesWritten.toLocaleString('en-US')} B</dd></div>
              <div><dt>zero copy</dt><dd>{exact.kv.zeroCopy ? 'yes' : 'no'}</dd></div>
            </dl>
          ) : null}
          {exact.tensor ? <TensorStats tensor={exact.tensor} /> : null}
          {exact.sample ? <p className="viz-note">Sample {exact.sample.id} “{exact.sample.text}”.</p> : null}
          {state.activeStageId === 'sample' && !exact.sample ? <p className="viz-note">This forward has no sample.</p> : null}
          {exact.top ? (
            <ol className="viz-top">
              {exact.top.map((row) => {
                const peak = exact.top?.[0]?.prob || 1;
                return (
                  <li key={row.id}>
                    <span className="viz-top-token">{row.text.trim() || '·'}</span>
                    <span className="viz-top-bar" aria-hidden="true">
                      <i style={{ width: `${Math.max(4, (row.prob / peak) * 100)}%` }} />
                    </span>
                    <span className="viz-top-logit">{num(row.logit, 2)}</span>
                    <span className="viz-top-prob">{(row.prob * 100).toFixed(1)}%</span>
                  </li>
                );
              })}
            </ol>
          ) : null}
          {exact.missingActivation ? (
            <p className="viz-note">Full-block stats on steps 0 and 16.</p>
          ) : null}
          {state.activeStageId === 'bpe-merge' ? <p className="viz-note">{trace.exact.bpe.chunks.length} BPE chunks on the prompt.</p> : null}
          <p className="viz-badge viz-break" data-kind="future">future scheduling</p>
          {schedule ? (
            <p className="viz-note">
              QKV matmul · {schedule.axis} · {schedule.splitCount} splits · {schedule.workerCount} workers · work {schedule.thresholds.work} / axis {schedule.thresholds.axis}
            </p>
          ) : <p className="viz-note">No matmul policy on this stage.</p>}
        </div>
        <div>
          <p className="viz-badge viz-break">Quantized display</p>
          {histogram ? <Histogram item={histogram} /> : <p className="viz-note">No histogram. Pairs are the 1e-4 weights.</p>}
          <p className="viz-badge viz-break" data-kind="time">measured runtime</p>
          <p className="viz-note">{stageMs === null ? 'No span for this step.' : `${stageMs.toFixed(3)} ms`}</p>
        </div>
      </section>
          </div>
        </>
      ) : (
      <section className={`viz-inspector${exact.top ? ' is-split' : ''}`} aria-label="Tensor and runtime">
        <div>
          <p className="viz-badge">Exact recorded</p>
          {exact.kv ? (
            <dl className="viz-stats">
              <div><dt>rows</dt><dd>{exact.kv.filledBefore} → {exact.kv.filledAfter} / {exact.kv.capacity} cache rows</dd></div>
              <div><dt>stride</dt><dd>{exact.kv.strideBytes} B</dd></div>
              <div><dt>K offset</dt><dd>{exact.kv.kByteOffset}</dd></div>
              <div><dt>V offset</dt><dd>{exact.kv.vByteOffset}</dd></div>
              <div><dt>written</dt><dd>{exact.kv.bytesWritten.toLocaleString('en-US')} B</dd></div>
              <div><dt>zero copy</dt><dd>{exact.kv.zeroCopy ? 'yes' : 'no'}</dd></div>
            </dl>
          ) : null}
          {exact.tensor ? <TensorStats tensor={exact.tensor} /> : null}
          {exact.sample ? <p className="viz-note">Sample {exact.sample.id} “{exact.sample.text}”.</p> : null}
          {state.activeStageId === 'sample' && !exact.sample ? <p className="viz-note">This forward has no sample.</p> : null}
          {exact.top ? (
            <ol className="viz-top">
              {exact.top.map((row) => {
                const peak = exact.top?.[0]?.prob || 1;
                return (
                  <li key={row.id}>
                    <span className="viz-top-token">{row.text.trim() || '·'}</span>
                    <span className="viz-top-bar" aria-hidden="true">
                      <i style={{ width: `${Math.max(4, (row.prob / peak) * 100)}%` }} />
                    </span>
                    <span className="viz-top-logit">{num(row.logit, 2)}</span>
                    <span className="viz-top-prob">{(row.prob * 100).toFixed(1)}%</span>
                  </li>
                );
              })}
            </ol>
          ) : null}
          {exact.missingActivation ? (
            <p className="viz-note">Full-block activation stats are captured on prefill (step 0) and the final forward (step 16).</p>
          ) : null}
          {state.activeStageId === 'bpe-merge' ? <p className="viz-note">{trace.exact.bpe.chunks.length} BPE chunks on the prompt.</p> : null}
          <p className="viz-badge viz-break" data-kind="future">future scheduling</p>
          {schedule ? (
            <p className="viz-note">
              axis {schedule.axis} · split {schedule.splitCount} · workers {schedule.workerCount} · chunks {schedule.chunks.length} · thresholds work {schedule.thresholds.work} / axis {schedule.thresholds.axis}
            </p>
          ) : <p className="viz-note">No matmul policy on this stage.</p>}
        </div>
        <div>
          <p className="viz-badge viz-break">Quantized display</p>
          {histogram ? <Histogram item={histogram} /> : <p className="viz-note">No histogram was recorded for this step.</p>}
          <p className="viz-badge viz-break" data-kind="time">measured runtime</p>
          <p className="viz-note">{stageMs === null ? 'No span for this step.' : `${stageMs.toFixed(3)} ms`}</p>
        </div>
      </section>
      )}
    </aside>
  );
}
