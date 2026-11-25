# GPT-2 in Racket: the write-up

A scroll-led write-up of how gpt2-rkt runs one forward pass. The text sits on the left and a pinned execution visualizer on the right. The visualizer replays a trace recorded from this engine: `public/traces/once-upon-a-time-16.trace.json`, exported with `tools/export-trace.rkt`.

```sh
cd site
npm install
npm run dev       # http://localhost:5173
npm run build     # static files in site/dist
```

It is a static Vite + React app. Any static host works. On Vercel, set the project's root directory to `site`.

The code excerpts on the page are pinned to commit `75286c5`. The benchmark figures are from runs on an Apple M5 Pro on 25 November 2026.
