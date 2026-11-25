import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Writeup } from './components/gpt2-rkt/Writeup';
import { SmoothScroll } from './components/SmoothScroll';
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

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <SmoothScroll>
      <Writeup />
    </SmoothScroll>
  </StrictMode>,
);
