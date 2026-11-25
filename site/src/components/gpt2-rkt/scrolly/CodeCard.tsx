import type { ReactNode } from 'react';
import {
  CodeBlock,
  CodeBlockContent,
  CodeBlockHeader,
  CodeBlockLine,
  CodeBlockTitle,
} from '@/components/ui/code-block';
import type { Snippet } from './snippets';

const KEYS = new Set([
  'define', 'define-values', 'let', 'let*', 'let-values', 'lambda', 'if', 'cond', 'else', 'unless', 'when',
  'for', 'for/list', 'for/fold', 'in-range', 'in-list', 'and', 'or', 'begin', 'stage', 'if-trace',
]);

/** Racket, colored the way the card's editor theme expects. */
export function paint(line: string): ReactNode[] {
  const out: ReactNode[] = [];
  const pattern = /;.*$|"(?:\\.|[^"\\])*"|#:[\w-]+|'[\w-]+|-?\d+(?:\.\d+)?(?:e-?\d+)?\b|[()[\]]|[^\s()[\]";]+|\s+/g;
  let previous = '';
  let match = pattern.exec(line);
  let index = 0;
  while (match) {
    const text = match[0];
    let kind = '';
    if (text.startsWith(';')) kind = 'cmt';
    else if (text.startsWith('"') || text.startsWith("'") || text.startsWith('#:')) kind = 'str';
    else if (/^-?\d/.test(text)) kind = 'num';
    else if (KEYS.has(text)) kind = 'kw';
    else if (/^[()[\]]$/.test(text)) kind = 'pn';
    else if (previous === '(' && /\S/.test(text)) kind = 'fn';
    out.push(kind ? <span key={index} className={`cc-${kind}`}>{text}</span> : text);
    if (/\S/.test(text)) previous = text;
    index += 1;
    match = pattern.exec(line);
  }
  return out;
}

/** A Racket excerpt in shadcn's code block: the file name and numbered lines. */
export function CodeCard({ snippet, peek = false, className }: { snippet: Snippet; caption?: ReactNode; peek?: boolean; className?: string }) {
  // One indent for the whole excerpt, so lines keep their nesting relative to each other.
  const all = snippet.chunks.flatMap((chunk) => chunk.code.split('\n')).filter((line) => line.trim());
  const indent = Math.min(...all.map((line) => line.length - line.trimStart().length));
  const rows = snippet.chunks.flatMap((chunk, c) => [
    ...(c > 0 ? [{ gap: true as const, key: `gap-${c}`, number: 0, text: '' }] : []),
    ...chunk.code.split('\n').map((line, i) => ({
      gap: false as const,
      key: `${chunk.start + i}`,
      number: chunk.start + i,
      text: line.slice(Number.isFinite(indent) ? indent : 0).trimEnd(),
    })),
  ]);
  const name = snippet.file.replace(/^src\//, '');

  return (
    <CodeBlock className={className ? `cc ${className}` : 'cc'} size={peek ? 'sm' : 'default'}>
      {peek ? null : (
        <CodeBlockHeader>
          <CodeBlockTitle>{name}</CodeBlockTitle>
        </CodeBlockHeader>
      )}
      <CodeBlockContent lineNumbers className="cc-code">
        {rows.map((row) => (row.gap ? (
          <CodeBlockLine key={row.key} className="cc-gap" aria-hidden="true" />
        ) : (
          // Excerpts skip lines, so each line sets its own number.
          <CodeBlockLine key={row.key} style={{ counterSet: `line ${row.number}` }}>
            {paint(row.text)}
          </CodeBlockLine>
        )))}
      </CodeBlockContent>
    </CodeBlock>
  );
}
