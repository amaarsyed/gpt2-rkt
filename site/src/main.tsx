import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Analytics } from '@vercel/analytics/react';
import { Writeup } from './components/gpt2-rkt/Writeup';
import { shouldPlayStartup, StartupIntro } from './components/StartupIntro';
import { playPop } from './lib/audio';
import './index.css';
import './components/iso/iso.css';
import './components/gpt2-rkt/writeup.css';

// Same bubble click as the rest of the site: any link or button.
document.addEventListener('pointerdown', (e) => {
  const target = e.target as Element | null;
  if (!target?.closest) return;
  if (target.closest('[data-no-pop]')) return;
  if (target.closest('a, button')) playPop();
});

// The title intro, once per session. Before the first paint, so the page's entrance
// animations wait underneath from the start.
const playStartup = shouldPlayStartup();
if (playStartup) document.documentElement.classList.add('is-intro');

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {/* Native scrolling: smooth on every device, with nothing in JavaScript between the
        wheel or finger and the page. */}
    <Writeup />
    <Analytics />
    {playStartup && <StartupIntro />}
  </StrictMode>,
);
