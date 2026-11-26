import { useEffect, useState, type ReactNode } from 'react';
import { Reveal } from '../Reveal';
import { palette, paletteDark } from '../../lib/content';
import { readStoredTheme, resolveDark, saveTheme, systemDarkQuery } from '../../lib/theme';
import { CacheTable } from './Blocks';
import { ScrollyLayout } from './scrolly/ScrollyLayout';
import { Step } from './scrolly/Step';
import { CodeStack } from './scrolly/CodeStack';
import { Tex } from './scrolly/Tex';
import { RktTerm } from './scrolly/RktTerm';

const REPO = 'https://github.com/amaarsyed/gpt2-rkt';

/** The authors' sites. This page lives in the gpt2-rkt repository; both sites link to it. */
/**
 * Which author's site the reader came from, so "← back" returns there. The link from
 * each site adds ?from=amaar or ?from=abdullah; otherwise the referrer decides. It is
 * kept for the session, so a refresh or an in-page jump doesn't lose it. Unknown: Amaar.
 */
function cameFrom(): keyof typeof AUTHORS {
  const KEY = 'gpt2-rkt-from';
  try {
    const asked = new URLSearchParams(window.location.search).get('from');
    if (asked === 'amaar' || asked === 'abdullah') {
      window.sessionStorage.setItem(KEY, asked);
      return asked;
    }
    const host = document.referrer ? new URL(document.referrer).hostname : '';
    if (host.includes('abdullah')) { window.sessionStorage.setItem(KEY, 'abdullah'); return 'abdullah'; }
    if (host.startsWith('portfolio-v01')) { window.sessionStorage.setItem(KEY, 'amaar'); return 'amaar'; }
    const kept = window.sessionStorage.getItem(KEY);
    if (kept === 'amaar' || kept === 'abdullah') return kept;
  } catch {
    // Storage or a malformed referrer: fall through to the default.
  }
  return 'amaar';
}

export const AUTHORS = {
  abdullah: { name: 'abdullah', href: 'https://abdullahehsan.vercel.app/projects' },
  amaar: { name: 'amaar', href: 'https://portfolio-v01-amber.vercel.app/#builds' },
} as const;

const ACCENT = '#f5c400';

/** A section opens with a small uppercase kicker over a big bold heading. Its heading and opening lines are one step of the scroll. */
function StepSection({ id, kicker, title, lead, children }: { id: Parameters<typeof Step>[0]['id']; kicker: string; title: string; lead?: ReactNode; children?: ReactNode }) {
  return (
    <Reveal whenVisible>
      <section className="wu-section">
        <Step id={id}>
          <p className="wu-kicker">{kicker}</p>
          <h2>{title}</h2>
          {lead}
        </Step>
        {children}
      </section>
    </Reveal>
  );
}

/** Every workload the repo's bench/run.sh measures, run on Amaar's M5 Pro on 2026-11-25. */
const M5_RESULTS: [string, string][] = [
  ['Load the weights', '1.75 s'],
  ['Prefill, 16 tokens', '113 ms'],
  ['Prefill, 128 tokens', '407 ms'],
  ['Prefill, 512 tokens', '1.50 s'],
  ['Prefill, 1,000 tokens', '3.52 s'],
  ['Greedy, 128 new tokens', '34.3 tokens/s'],
  ['Matrix product 768 × 768 × 768', '71.9 GFLOP/s'],
  ['Tied head 1 × 768 × 50,257', '34.6 GFLOP/s'],
];

function BenchTable() {
  return (
    <figure className="wu-bench" id="benchmarks">
      <figcaption className="wu-bench-title">
        Benchmarks <span>Apple M5 Pro · Racket 9.3 CS</span>
      </figcaption>
      <table>
        <tbody>
          {M5_RESULTS.map(([what, value]) => (
            <tr key={what}>
              <th scope="row">{what}</th>
              <td>{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}

function GitHubMark() {
  return (
    <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
      <path
        fill="currentColor"
        d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"
      />
    </svg>
  );
}

function SourcePill() {
  return (
    <a className="wu-gh" href={REPO} target="_blank" rel="noopener noreferrer" aria-label="Source code on GitHub" title="Source code on GitHub">
      <GitHubMark />
    </a>
  );
}

export function Writeup() {
  const [dark, setDark] = useState(resolveDark);
  const colors = dark ? paletteDark : palette;
  const theme = dark ? 'dark' : 'light';

  useEffect(() => {
    const root = document.documentElement;
    for (const [key, value] of Object.entries(colors)) root.style.setProperty(`--${key}`, value);
    root.style.setProperty('--accent', ACCENT);
    root.style.background = colors.bg;
    root.style.colorScheme = theme;
    root.dataset.theme = theme;
    document.body.style.background = colors.bg;
  }, [colors, theme]);

  // Follow the device's light/dark setting live, until they pick one themselves.
  useEffect(() => {
    const query = systemDarkQuery();
    const onChange = (e: MediaQueryListEvent) => {
      if (!readStoredTheme()) setDark(e.matches);
    };
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }, []);

  const toggleTheme = () => {
    const next = !dark;
    saveTheme(next);
    setDark(next);
  };

  const [from] = useState(cameFrom);
  const back = { href: AUTHORS[from].href, label: from === 'abdullah' ? 'Abdullah' : 'Amaar' };

  return (
    <div className="wu-root iso is-scrolly" data-theme={theme}>
      <nav className="wu-nav" aria-label="Site">
        <a href={back.href} className="wu-back">
          <span aria-hidden="true">←</span> {back.label}
        </a>
        <button
          type="button"
          className="theme-toggle wu-toggle"
          aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
          title={dark ? 'Light mode' : 'Dark mode'}
          onClick={toggleTheme}
        >
          {dark ? (
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2.5v2.5M12 19v2.5M2.5 12H5M19 12h2.5M5.3 5.3l1.8 1.8M16.9 16.9l1.8 1.8M5.3 18.7l1.8-1.8M16.9 7.1l1.8-1.8" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M20.5 14.2A8.5 8.5 0 0 1 9.8 3.5a8.5 8.5 0 1 0 10.7 10.7z" />
            </svg>
          )}
        </button>
      </nav>

      <main className="wu-main">
        <ScrollyLayout theme={theme}>
        <article>
          <Step id="intro">
          <Reveal>
            <header className="wu-header">
              <h1>GPT-2 in Racket λ</h1>
              <p className="wu-deck">A 124M-parameter language model, running in pure Racket.</p>
              <p className="wu-byline">
                by{' '}
                <a href={AUTHORS.abdullah.href} className="hover-fill-host">
                  <span className="fill-highlight">abdullah</span>
                </a>{' '}
                &amp;{' '}
                <a href={AUTHORS.amaar.href} className="hover-fill-host">
                  <span className="fill-highlight">amaar</span>
                </a>
              </p>
              <SourcePill />
            </header>
          </Reveal>

          <Reveal delay={120}>
            <p className="wu-kicker">In one sentence</p>
            <div className="wu-tldr">
              We load GPT-2 small's 124 million pretrained weights into an engine written in plain Racket, check its
              output against PyTorch, and find where its memory and time go: copies in the attention cache, and
              matrix products split across CPU cores. On an Apple M5 Pro it generates <strong>about 34 tokens per second</strong>.
            </div>
            <p className="wu-lede">
              This began as a daydream in a{' '}
              <a href="https://student.cs.uwaterloo.ca/~cs145/" target="_blank" rel="noopener noreferrer">
                CS145
              </a>{' '}
              lecture: what if LLM inference were written in Racket?
            </p>
            <CodeStack cardId="tree" />
          </Reveal>
          </Step>

          <StepSection id="forward-pass" kicker="The forward pass" title="From a prompt to a token">
            <Step id="tokenize">
            <p>
              The <RktTerm file="tokenizer">tokenizer</RktTerm> merges byte pairs using GPT-2's ranked merge list, so “Once upon a time” becomes four
              tokens. Each token's <RktTerm file="embed">embedding</RktTerm> is added to an embedding for its position, a vector of 768 numbers:
            </p>
            <Tex block>{String.raw`\sym{xi}{x_i} = \sym{Wte}{W_{te}}[\sym{ti}{t_i}] + \sym{Wpe}{W_{pe}}[\sym{i}{i}], \qquad \sym{xi}{x_i} \in \sym{R768}{\mathbb{R}^{768}}`}</Tex>
            <p>That vector runs through twelve blocks.</p>
            <CodeStack cardId="tokenizer" />
            </Step>
            <Step id="block">
            <p>
              Each <RktTerm file="block">block</RktTerm> normalizes its input, runs attention, and adds the result back to the vector; then it
              does the same with a feed-forward network:{' '}
              <Tex>{String.raw`\sym{x}{x} \leftarrow \sym{x}{x} + \sym{Attn}{\mathrm{Attn}}(\sym{LN1}{\mathrm{LN}_1}\,\sym{x}{x})`}</Tex>, then{' '}
              <Tex>{String.raw`\sym{x}{x} \leftarrow \sym{x}{x} + \sym{MLP}{\mathrm{MLP}}(\sym{LN2}{\mathrm{LN}_2}\,\sym{x}{x})`}</Tex>.
            </p>
            <p>
              For attention, the block projects each vector into a query, a key, and a value. <RktTerm file="attention">Attention</RktTerm> scores the
              query against keys at its own and earlier positions, and a softmax turns those scores into weights for
              mixing the values:
            </p>
            <Tex block>{String.raw`\sym{Attn}{\mathrm{Attn}}(\sym{Q}{Q}, \sym{K}{K}, \sym{V}{V}) = \sym{softmax}{\mathrm{softmax}}\left( \frac{\sym{QKT}{Q K^{\top}}}{\sym{sqrtdk}{\sqrt{d_k}}} + \sym{M}{M} \right) \sym{V}{V}`}</Tex>
            <p>
              Here <Tex>{String.raw`\sym{M}{M}`}</Tex> hides future positions and <Tex>{String.raw`\sym{dk}{d_k} = 64`}</Tex>. Each of the 12 heads
              computes this over 64 of the 768 numbers, and their outputs are joined and projected back to 768. The{' '}
              <RktTerm file="nn">feed-forward network</RktTerm> takes each vector from 768 to 3,072 numbers and back, with GELU in its tanh
              approximation:{' '}
              <Tex>{String.raw`\sym{MLP}{\mathrm{MLP}}(\sym{x}{x}) = \sym{W2}{W_2} \, \sym{GELU}{\mathrm{GELU}}(\sym{W1}{W_1} \sym{x}{x} + \sym{b1}{b_1}) + \sym{b2}{b_2}`}</Tex>.
            </p>
            <CodeStack cardId="attention" />
            </Step>
            <Step id="logits">
            <p>
              After the last block, the model normalizes the final vector and <RktTerm file="model">scores it</RktTerm> against every row of the
              token embedding table, the same table used for the input embedding (<strong>weight tying</strong>).{' '}
              <RktTerm file="generate">Greedy decoding</RktTerm> takes the highest of the 50,257 scores:
            </p>
            <Tex block>{String.raw`\sym{z}{z} = \sym{Wte}{W_{te}} \, \sym{LN}{\mathrm{LN}}(\sym{hL}{h_L}) \in \sym{R50257}{\mathbb{R}^{50257}}, \qquad \sym{tnext}{t_{\mathrm{next}}} = \sym{argmax}{\arg\max_j} \sym{z}{z_j}`}</Tex>
            <CodeStack cardId="model" />
            </Step>
          </StepSection>

          <StepSection id="kv-cache" kicker="The KV cache" title="Where the copies were" lead={<>
            <p>
              The first pass over the prompt, called <strong>prefill</strong>, saves every block's keys and values in a cache. After
              the model picks the comma, only the comma runs through the blocks. Each block adds one key row and one
              value row, and attention reads all five.
            </p>
            <CacheTable />
          </>}>
            <Step id="kv-copies">
            <p>
              Our earlier code cached those rows, but every attention head copied its slice into fresh arrays on each
              step, so the copies grew with the text. Three revisions removed them. First, each head reads its
              columns of the cache in place. Next, the cache is sized to the prompt plus the output; that revision,
              which also changed the futures scheduler and top-k, took a 1,000-token prefill <strong>from 5.6 s to 4.0 s</strong> and
              greedy decoding <strong>from 18 to 31 tokens per second</strong>. Last, new keys and values are written straight from the
              packed projection into the cache. That change cut the bytes allocated during the same prefill <strong>from
              1,555 MB to 1,337 MB</strong>, while the time barely moved, 4.0 s to 3.8 s. Allocated bytes count everything the
              call allocates, not memory in use.
            </p>
            <p className="wu-aside-line">
              Timings are medians of three runs, each in a fresh process after one warmup, on an Apple M5 Pro (6
              performance and 12 efficiency cores, Racket 9.3 CS). Three runs show a direction, not a statistical
              result.
            </p>
            <CodeStack cardId="block" />
            </Step>
          </StepSection>

          <StepSection id="parallelism" kicker="Parallelism" title="Splitting the math across cores" lead={<>
            <p>
              The engine stores tensors in flvectors, Racket's flat arrays of floats, and splits large matrix products
              across CPU cores with <RktTerm file="nn">Racket futures</RktTerm>. A long prompt has enough rows to split by row. A single
              decoding token has one row, so the engine splits by output column:{' '}
              <Tex>{String.raw`\sym{rows}{\text{rows}} \ge \sym{n32}{32} \Rightarrow \text{split rows, else columns}`}</Tex>. Products under
              100,000 multiply-adds stay on one thread. With one worker, a 128-token prefill took 3.9 s; with all
              18 cores it took <strong>0.44 s</strong>. Decoding peaked earlier: 128 greedy tokens took 10.0 s on one worker, <strong>3.0 s on
              12</strong>, and 4.1 s on all 18, probably because the single-row products then also wait on the slower efficiency
              cores.
            </p>
            <CodeStack cardId="nn" />
          </>}>
            <Step id="checks">
            <p>
              We compare hidden states, logits, and the greedy tokens against <strong>an eager PyTorch run of the same
              weights</strong>: within <Tex>{String.raw`|a - e| \le 10^{-3} + 10^{-4}|e|`}</Tex> in fp32, and for the last row
              against a float64 copy at <Tex>{String.raw`10^{-6}`}</Tex>. The timed runs produced <strong>the same token ids</strong>.
            </p>
            <BenchTable />
            </Step>
          </StepSection>

          <Step id="read-the-code">
          <Reveal whenVisible>
            <footer className="wu-endcard">
              <h2>Read the code</h2>
              <p className="wu-links">
                <SourcePill />
              </p>
              <p className="wu-pair">
                This write-up is shared between our sites. Find it under{' '}
                <a href={AUTHORS.abdullah.href}>Abdullah's projects</a> and <a href={AUTHORS.amaar.href}>Amaar's builds</a>.
              </p>
              <p>We hope you had as much fun reading this as we did making it!</p>
            </footer>
          </Reveal>
          </Step>
        </article>
        </ScrollyLayout>
      </main>
    </div>
  );
}
