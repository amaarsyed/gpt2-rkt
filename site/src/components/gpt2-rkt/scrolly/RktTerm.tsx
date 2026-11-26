import { useRef, useState, type ReactNode } from 'react';
import { HoverCard } from 'radix-ui';
import { RACKET_COMMIT } from '../visualizer/racketSource';
import { RKT } from './snippets';
import { useTermFocus } from './TermFocus';

/**
 * A word in the write-up that names a part of the model. Hover (or focus) it to see
 * which .rkt file does that and a sentence on what it does.
 */
export function RktTerm({ file, children }: { file: keyof typeof RKT; children: ReactNode }) {
  const entry = RKT[file];
  const [open, setOpenState] = useState(false);
  const point = useTermFocus();
  // While its card is open, the graph frames and lights this file's module node.
  const setOpen = (next: boolean) => {
    setOpenState(next);
    point(next ? `mod-${file}` : null);
  };
  // Touch has no hover: the first tap previews the file and a second tap closes it (a tap
  // anywhere else closes it too, via the card's own dismiss). The card carries the GitHub link.
  // Radix closes the card on pointerdown before the click lands, so the click goes by
  // whether it was open when the finger came down.
  const touched = useRef(false);
  const wasOpen = useRef(false);
  if (!entry) return <>{children}</>;
  const { chunks } = entry.snippet;
  const first = chunks[0]?.start ?? 1;
  const tail = chunks[chunks.length - 1];
  const last = tail ? tail.start + tail.code.split('\n').length - 1 : first;
  const href = `https://github.com/amaarsyed/gpt2-rkt/blob/${RACKET_COMMIT}/${entry.snippet.file}#L${first}-L${last}`;
  return (
    <HoverCard.Root open={open} onOpenChange={setOpen} openDelay={120} closeDelay={80}>
      <HoverCard.Trigger asChild>
        <a
          className="rkt-term"
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          data-no-pop=""
          onPointerDown={(event) => {
            touched.current = event.pointerType === 'touch';
            wasOpen.current = open;
          }}
          onClick={(event) => {
            if (!touched.current) return;
            event.preventDefault();
            setOpen(!wasOpen.current);
          }}
          onFocus={() => { if (!touched.current) setOpen(true); }}
          onBlur={() => setOpen(false)}
          onKeyDown={(event) => { if (event.key === 'Escape') setOpen(false); }}
        >
          <code>{children}</code>
        </a>
      </HoverCard.Trigger>
      <HoverCard.Portal>
        <HoverCard.Content className="rkt-card" side="top" align="start" sideOffset={10} collisionPadding={16}>
          <p className="rkt-card-file">{entry.file}</p>
          <p className="rkt-card-what">{entry.what}</p>
          <a className="rkt-card-open" href={href} target="_blank" rel="noopener noreferrer">
            Open on GitHub ↗
          </a>
          <HoverCard.Arrow className="rkt-card-arrow" width={14} height={7} />
        </HoverCard.Content>
      </HoverCard.Portal>
    </HoverCard.Root>
  );
}
