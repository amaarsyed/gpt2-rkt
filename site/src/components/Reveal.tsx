import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';

export type RevealVariant =
  /** Text and blocks: fade in, lift up and sharpen. */
  | 'up'
  /** Same, dropping in from above (for the header). */
  | 'down'
  /** Opacity only, for areas with their own transforms (the sticker desk). */
  | 'fade'
  /** Curtain wipe from the top, for photos and video. */
  | 'wipe-down'
  /** Wipe from the left, like the signature being written. */
  | 'wipe-right';

type RevealProps = {
  children: ReactNode;
  variant?: RevealVariant;
  /** Milliseconds before it starts, so neighbours can cascade. */
  delay?: number;
  /** Wait until it scrolls into view instead of playing on load. */
  whenVisible?: boolean;
  className?: string;
  id?: string;
  style?: CSSProperties;
};

/**
 * Reveals its children once, on load or when scrolled into view. The motion is
 * pure CSS (see `.reveal` in index.css), and its keyframes only describe the
 * hidden start, so afterwards the element is back to its normal style with no
 * leftover transform, filter or clip.
 */
export function Reveal({
  children,
  variant = 'up',
  delay = 0,
  whenVisible = false,
  className,
  id,
  style,
}: RevealProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(!whenVisible);

  useEffect(() => {
    if (shown) return;
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === 'undefined') {
      setShown(true);
      return;
    }
    const io = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        setShown(true);
        io.disconnect();
      },
      { rootMargin: '0px 0px -8% 0px' },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [shown]);

  return (
    <div
      ref={ref}
      id={id}
      className={`reveal reveal-${variant}${shown ? ' is-in' : ''}${className ? ` ${className}` : ''}`}
      style={{ ...style, '--reveal-delay': `${delay}ms` } as CSSProperties}
    >
      {children}
    </div>
  );
}
