import type { StageId } from './traceTypes';

/** One plain sentence per stage, shown on hover and focus. */
export const STAGE_TIP: Record<string, string> = {
  prompt: 'The words you typed, already split into the tokens the model will read.',
  'bpe-merge': 'Merges byte pairs with GPT-2’s ranked list until the text is tokens.',
  'input-embed': 'Adds each token’s 768-number vector to a vector for its position.',
  'block-ln1': 'Rescales the token vectors before attention.',
  'block-qkv': 'Projects each vector into a query, a key, and a value.',
  'kv-write': 'Writes the new key and value rows into this block’s cache.',
  attend: 'Each token looks back at earlier tokens to decide what matters.',
  'block-proj-res': 'Projects the attention result and adds it back onto the original vector.',
  'block-ln2': 'Rescales the vector before the feed-forward network.',
  'mlp-gelu': 'Expands each vector from 768 to 3,072 numbers and applies GELU.',
  'block-mlp-res': 'Projects that expansion back to 768 and adds it to the residual.',
  'final-ln': 'Normalizes the last block’s vector before scoring the vocabulary.',
  'tied-logits': 'Dots that vector with every token embedding to score the next token.',
  sample: 'Picks the highest score. This recorded run is greedy.',
};

export const MODULE_TIP: Record<string, string> = {
  'mod-tokenizer': 'tokenizer.rkt runs the byte-pair merges.',
  'mod-embed': 'embed.rkt holds the token and position tables.',
  'mod-attention': 'attention.rkt is the causal attention kernel.',
  'mod-block': 'block.rkt is one transformer block.',
  'mod-nn': 'nn.rkt is the GELU feed-forward.',
  'mod-model': 'model.rkt stacks the blocks, the final norm, and the tied logits.',
  'mod-generate': 'generate.rkt is the greedy decode loop.',
};

/** A slightly longer explanation for the inspector’s first tab. */
export const STAGE_WHAT: Record<string, string> = {
  ...STAGE_TIP,
  attend: 'Each token looks back at earlier tokens to decide what matters. Future positions are masked, the scores are softmaxed, and the values are mixed by those weights.',
  'kv-write': 'Writes the new key and value rows into this block’s cache, straight from the packed QKV projection, so later steps can read them in place.',
  'tied-logits': 'Dots the final vector with every row of the token embedding table. That table is tied: the same weights that embedded the input score the next token.',
};

export const CODE_NOTES: Partial<Record<StageId, { line: number; note: string }[]>> = {
  'bpe-merge': [
    { line: 118, note: 'picks the merge pair with the best rank' },
    { line: 122, note: 'replaces that pair with one token and continues' },
  ],
  'input-embed': [
    { line: 81, note: 'token vector plus position vector' },
  ],
  'block-ln1': [
    { line: 149, note: 'rescales x with this block’s gain and bias' },
  ],
  'block-qkv': [
    { line: 156, note: 'one matmul produces packed query, key, and value' },
  ],
  'kv-write': [
    { line: 122, note: 'key starts after the query slice of the packed row' },
    { line: 125, note: 'copies that key into the cache' },
    { line: 126, note: 'copies the value, the next slice, into the cache' },
  ],
  attend: [
    { line: 36, note: 'masks future tokens' },
    { line: 48, note: 'scales the query-key dot by 1/√d' },
    { line: 58, note: 'exponentiates the score into an unnormalized weight' },
    { line: 64, note: 'divides by the sum, which is the softmax' },
    { line: 70, note: 'adds this value, weighted, into the output' },
  ],
  'block-proj-res': [
    { line: 196, note: 'projects the heads and adds the residual' },
  ],
  'block-ln2': [
    { line: 204, note: 'rescales the attention residual' },
  ],
  'mlp-gelu': [
    { line: 211, note: 'expands to the MLP width and applies GELU' },
  ],
  'block-mlp-res': [
    { line: 218, note: 'projects back to 768 and adds the residual' },
  ],
  'final-ln': [
    { line: 187, note: 'normalizes the stack output' },
  ],
  'tied-logits': [
    { line: 196, note: 'scores the vocabulary with the token embedding table' },
  ],
  sample: [
    { line: 243, note: 'walks the logits and keeps the largest' },
  ],
};
