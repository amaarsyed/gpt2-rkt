export const NAMES = [
  'عمار',
  'अम्मार',
  '阿马尔',
  'アマル',
  '아마르',
  'Аммар',
  'Αμμάρ',
  'עמאר',
  'อัมมาร์',
  'ammar',
] as const;

// BCP 47 tags, index-aligned with NAMES, so each script gets the right font and voice.
export const NAME_LANGS = [
  'ar',
  'hi',
  'zh',
  'ja',
  'ko',
  'ru',
  'el',
  'he',
  'th',
  'en',
] as const;

export const NAV = [
  { label: 'builds', href: '#builds' },
  { label: 'shots', href: '/photos.html' },
  { label: 'desk', href: '#desk' },
] as const;

/** Things built, each with its write-up. Shared write-ups are linked from both authors' sites. */
export const BUILDS = [
  {
    title: 'gpt-2 in racket',
    href: '/gpt2-rkt.html',
    partner: { name: 'abdullah', href: 'https://abdullahehsan.vercel.app/' },
  },
] as const;

export const palette = {
  bg: '#faf9f7',
  fg: '#1c1b19',
  muted: '#726f68',
  faint: '#eceae4',
  line: '#e4e0d8',
} as const;

export const paletteDark = {
  bg: '#121210',
  fg: '#ecebe6',
  muted: '#ffffff',
  faint: '#1f1e1b',
  line: '#34322d',
} as const;

export const SOCIALS = {
  x: {
    label: 'X',
    href: 'https://x.com/amaarrsyed',
  },
  github: {
    label: 'GitHub',
    href: 'https://github.com/amaarsyed',
  },
  linkedin: {
    label: 'LinkedIn',
    href: 'https://www.linkedin.com/in/amaarshah/',
  },
} as const;
