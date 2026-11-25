import { useEffect, useRef, useState } from 'react';
import { RushTypeCard } from './rush-type/RushTypeCard';
import { resolveDark } from '../lib/theme';

const SEEN_KEY = 'gpt2-rkt-seen-intro';
/** Never trap a reader: the intro always ends within this many ms. */
const MAX_MS = 6000;
const HOLD_MS = 450;
const FADE_MS = 650;

/**
 * Once per browser session, unless the reader asked for reduced motion, so a
 * refresh while reading doesn't replay it. Adding ?intro to the address plays it again.
 */
export function shouldPlayStartup(): boolean {
  try {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return false;
    if (new URLSearchParams(window.location.search).has('intro')) return true;
    if (window.sessionStorage.getItem(SEEN_KEY)) return false;
    window.sessionStorage.setItem(SEEN_KEY, '1');
    return true;
  } catch {
    return false;
  }
}

/**
 * The same opening as Amaar's site: the title rushes in full-screen, holds, then fades
 * while the write-up builds in underneath (its entrance animations wait on html.is-intro).
 */
export function StartupIntro() {
  const [out, setOut] = useState(false);
  const [gone, setGone] = useState(false);
  const started = useRef(false);
  const dark = resolveDark();

  useEffect(() => {
    document.documentElement.classList.add('is-intro');
    return () => document.documentElement.classList.remove('is-intro');
  }, []);

  const finish = () => {
    if (started.current) return;
    started.current = true;
    window.setTimeout(() => {
      setOut(true);
      // Release the page while the overlay is still fading, so it builds in underneath.
      document.documentElement.classList.remove('is-intro');
      window.setTimeout(() => setGone(true), FADE_MS);
    }, HOLD_MS);
  };

  useEffect(() => {
    const cap = window.setTimeout(finish, MAX_MS);
    return () => window.clearTimeout(cap);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (gone) return null;

  return (
    <div
      className={`startup-intro${out ? ' is-out' : ''}`}
      style={{ background: dark ? '#121210' : '#f5f5f5' }}
      aria-hidden="true"
    >
      <div className="startup-intro-card">
        <RushTypeCard
          dark={dark}
          style={{ width: '100%', height: '100%' }}
          onSettled={finish}
          replayOnTouch={false}
        />
      </div>
    </div>
  );
}
