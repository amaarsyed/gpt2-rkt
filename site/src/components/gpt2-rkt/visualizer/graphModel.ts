import type { StageId } from './traceTypes';

export type NodeRole = 'ink' | 'residual' | 'attention' | 'memory' | 'future' | 'module';
export type EdgeKind = 'forward' | 'residual' | 'memory' | 'module';

/** Semantic roles. The renderer resolves these to CSS variables and never reassigns them. */
export const ROLE_VAR: Record<NodeRole, string> = {
  residual: '--accent',
  attention: '--anatomy-blue',
  future: '--anatomy-green',
  memory: '--anatomy-red',
  ink: '--fg',
  module: '--anatomy-muted',
};

export interface GraphNode {
  id: string;
  label: string;
  sub: string;
  role: NodeRole;
  stageId: StageId | null;
  explore: { x: number; y: number };
  execute: { x: number; y: number };
  radius: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  kind: EdgeKind;
  /** Perpendicular offset, in world pixels, so parallel arcs stay apart. */
  bend: number;
}

export interface ExecutionGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  width: number;
  height: number;
}

const stageRole: Record<StageId, NodeRole> = {
  'bpe-merge': 'ink',
  'input-embed': 'residual',
  'block-ln1': 'residual',
  'block-qkv': 'attention',
  'kv-write': 'memory',
  attend: 'attention',
  'block-proj-res': 'residual',
  'block-ln2': 'residual',
  'mlp-gelu': 'future',
  'block-mlp-res': 'residual',
  'final-ln': 'residual',
  'tied-logits': 'ink',
  sample: 'ink',
};

function node(
  id: string,
  label: string,
  sub: string,
  role: NodeRole,
  stageId: StageId | null,
  explore: { x: number; y: number },
  execute: { x: number; y: number },
  radius = 16,
): GraphNode {
  return { id, label, sub, role, stageId, explore, execute, radius };
}

export function buildGraph(): ExecutionGraph {
  const nodes: GraphNode[] = [
    node('prompt', 'Prompt', 'Once upon a time', 'ink', null, { x: 120, y: 96 }, { x: 250, y: 78 }, 17),
    node('mod-tokenizer', 'tokenizer.rkt', 'byte pair', 'module', null, { x: 210, y: 210 }, { x: 118, y: 78 }, 13),
    node('bpe-merge', 'BPE', 'bpe-merge', stageRole['bpe-merge'], 'bpe-merge', { x: 300, y: 78 }, { x: 418, y: 78 }),
    node('mod-embed', 'embed.rkt', 'wte + wpe', 'module', null, { x: 390, y: 210 }, { x: 118, y: 148 }, 13),
    node('input-embed', 'Embed', 'input-embed', stageRole['input-embed'], 'input-embed', { x: 500, y: 86 }, { x: 586, y: 78 }),
    node('mod-attention', 'attention.rkt', 'causal', 'module', null, { x: 108, y: 360 }, { x: 118, y: 288 }, 13),
    node('block-qkv', 'QKV', 'block-qkv', stageRole['block-qkv'], 'block-qkv', { x: 290, y: 320 }, { x: 418, y: 214 }),
    node('kv-write', 'KV Write', 'kv-write', stageRole['kv-write'], 'kv-write', { x: 168, y: 490 }, { x: 586, y: 214 }),
    node('attend', 'Attend', 'attend', stageRole.attend, 'attend', { x: 360, y: 470 }, { x: 754, y: 214 }),
    node('mod-block', 'block.rkt', 'residual', 'module', null, { x: 560, y: 300 }, { x: 118, y: 218 }, 13),
    node('block-ln1', 'LN1', 'block-ln1', stageRole['block-ln1'], 'block-ln1', { x: 620, y: 128 }, { x: 250, y: 214 }),
    node('block-proj-res', 'Proj+Res', 'block-proj-res', stageRole['block-proj-res'], 'block-proj-res', { x: 760, y: 210 }, { x: 250, y: 350 }),
    node('block-ln2', 'LN2', 'block-ln2', stageRole['block-ln2'], 'block-ln2', { x: 540, y: 420 }, { x: 418, y: 350 }),
    node('block-mlp-res', 'MLP+Res', 'block-mlp-res', stageRole['block-mlp-res'], 'block-mlp-res', { x: 760, y: 390 }, { x: 754, y: 350 }),
    node('mod-nn', 'nn.rkt', 'future split', 'module', null, { x: 500, y: 530 }, { x: 118, y: 358 }, 13),
    node('mlp-gelu', 'MLP/GELU', 'mlp-gelu', stageRole['mlp-gelu'], 'mlp-gelu', { x: 640, y: 450 }, { x: 586, y: 350 }),
    node('mod-model', 'model.rkt', 'stack', 'module', null, { x: 1280, y: 220 }, { x: 118, y: 428 }, 13),
    node('final-ln', 'Final LN', 'final-ln', stageRole['final-ln'], 'final-ln', { x: 1100, y: 56 }, { x: 418, y: 486 }),
    node('tied-logits', 'Logits', 'tied-logits', stageRole['tied-logits'], 'tied-logits', { x: 720, y: 300 }, { x: 586, y: 486 }),
    node('mod-generate', 'generate.rkt', 'greedy', 'module', null, { x: 760, y: 500 }, { x: 118, y: 486 }, 13),
    node('sample', 'Sample', 'sample', stageRole.sample, 'sample', { x: 900, y: 490 }, { x: 754, y: 486 }),
  ];

  const edges: GraphEdge[] = [
    edge('f-prompt', 'prompt', 'bpe-merge', 'forward', 16),
    edge('f-bpe', 'bpe-merge', 'input-embed', 'forward', -16),
    edge('f-embed', 'input-embed', 'block-ln1', 'forward', 18),
    edge('f-ln1', 'block-ln1', 'block-qkv', 'forward', -14),
    edge('f-qkv', 'block-qkv', 'attend', 'forward', 14),
    edge('f-attend', 'attend', 'block-proj-res', 'forward', -22),
    edge('f-proj', 'block-proj-res', 'block-ln2', 'forward', 16),
    edge('f-ln2', 'block-ln2', 'mlp-gelu', 'forward', -14),
    edge('f-gelu', 'mlp-gelu', 'block-mlp-res', 'forward', 14),
    edge('f-mlp', 'block-mlp-res', 'final-ln', 'forward', -20),
    edge('f-fln', 'final-ln', 'tied-logits', 'forward', 14),
    edge('f-logits', 'tied-logits', 'sample', 'forward', -14),
    edge('r-attn', 'block-ln1', 'block-proj-res', 'residual', 72),
    edge('r-mlp', 'block-ln2', 'block-mlp-res', 'residual', 56),
    edge('m-write', 'block-qkv', 'kv-write', 'memory', 22),
    edge('m-read', 'kv-write', 'attend', 'memory', -22),
    edge('u-tok', 'mod-tokenizer', 'bpe-merge', 'module', 10),
    edge('u-prompt', 'prompt', 'mod-tokenizer', 'module', -8),
    edge('u-emb', 'mod-embed', 'input-embed', 'module', 10),
    edge('u-qkv', 'mod-attention', 'block-qkv', 'module', 12),
    edge('u-att', 'mod-attention', 'attend', 'module', -12),
    edge('u-kv', 'mod-attention', 'kv-write', 'module', 8),
    edge('u-ln1', 'mod-block', 'block-ln1', 'module', -10),
    edge('u-ln2', 'mod-block', 'block-ln2', 'module', 10),
    edge('u-proj', 'mod-block', 'block-proj-res', 'module', 8),
    edge('u-mlp', 'mod-block', 'block-mlp-res', 'module', -8),
    edge('u-nn', 'mod-nn', 'mlp-gelu', 'module', 8),
    edge('u-model', 'mod-model', 'final-ln', 'module', 10),
    edge('u-tie', 'mod-model', 'tied-logits', 'module', -10),
    edge('u-gen', 'mod-generate', 'sample', 'module', 8),
  ];

  return { nodes, edges, width: 980, height: 540 };
}

function edge(id: string, source: string, target: string, kind: EdgeKind, bend: number): GraphEdge {
  return { id, source, target, kind, bend };
}

const MODULE_FOR_STAGE: Record<string, string> = {
  'bpe-merge': 'mod-tokenizer',
  'input-embed': 'mod-embed',
  'block-ln1': 'mod-block',
  'block-ln2': 'mod-block',
  'block-proj-res': 'mod-block',
  'block-mlp-res': 'mod-block',
  'block-qkv': 'mod-attention',
  'kv-write': 'mod-attention',
  attend: 'mod-attention',
  'mlp-gelu': 'mod-nn',
  'final-ln': 'mod-model',
  'tied-logits': 'mod-model',
  sample: 'mod-generate',
};

/** The Racket file that owns a pipeline stage, for the execute-mode module rail. */
export function moduleIdForStage(stageId: string | null): string | null {
  if (!stageId) return null;
  return MODULE_FOR_STAGE[stageId] ?? null;
}

export function nodeById(graph: ExecutionGraph, id: string | null): GraphNode | null {
  if (!id) return null;
  return graph.nodes.find((item) => item.id === id) ?? null;
}

export function neighbors(graph: ExecutionGraph, id: string): Set<string> {
  const set = new Set<string>([id]);
  for (const item of graph.edges) {
    if (item.source === id) set.add(item.target);
    if (item.target === id) set.add(item.source);
  }
  return set;
}

let cached: ExecutionGraph | null = null;

export function executionGraph(): ExecutionGraph {
  if (!cached) cached = buildGraph();
  return cached;
}
