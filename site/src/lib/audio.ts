const POP_SRC = '/sounds/bubble.mp3';
const POP_VOLUME = 0.55;
const INTRO_SRC = '/sounds/game-start.mp3';
const INTRO_VOLUME = 0.65;
const SIX_SRC = '/sounds/6-god.mp3';
const SIX_VOLUME = 0.7;
const SHUTTER_SRC = '/sounds/camera-shutter.mp3';
const SHUTTER_VOLUME = 0.75;

let base: HTMLAudioElement | null = null;
let intro: HTMLAudioElement | null = null;
let six: HTMLAudioElement | null = null;
let shutter: HTMLAudioElement | null = null;

function getPop(): HTMLAudioElement {
  if (!base) {
    base = new Audio(POP_SRC);
    base.preload = 'auto';
    base.volume = POP_VOLUME;
  }
  return base;
}

function getIntro(): HTMLAudioElement {
  if (!intro) {
    intro = new Audio(INTRO_SRC);
    intro.preload = 'auto';
    intro.volume = INTRO_VOLUME;
  }
  return intro;
}

/** UI pop for clicks — clones so rapid taps can overlap. */
export function playPop() {
  try {
    const sample = getPop();
    const node = sample.cloneNode(true) as HTMLAudioElement;
    node.volume = sample.volume;
    void node.play();
  } catch {
    // ignore audio failures (autoplay policy, missing file, etc.)
  }
}

function getSix(): HTMLAudioElement {
  if (!six) {
    six = new Audio(SIX_SRC);
    six.preload = 'auto';
    six.volume = SIX_VOLUME;
  }
  return six;
}

/** Toronto 6 God sting — 5s cooldown so repeats don’t stack. */
const SIX_COOLDOWN_MS = 5000;
let sixReadyAt = 0;

export function playSixGod() {
  const now = Date.now();
  if (now < sixReadyAt) return;
  sixReadyAt = now + SIX_COOLDOWN_MS;
  try {
    const sample = getSix();
    const node = sample.cloneNode(true) as HTMLAudioElement;
    node.volume = sample.volume;
    node.currentTime = 0;
    void node.play();
  } catch {
    // ignore audio failures (autoplay policy, missing file, etc.)
  }
}

function getShutter(): HTMLAudioElement {
  if (!shutter) {
    shutter = new Audio(SHUTTER_SRC);
    shutter.preload = 'auto';
    shutter.volume = SHUTTER_VOLUME;
  }
  return shutter;
}

/** Camera shutter for the Shot on iPhone sticker. */
export function playShutter() {
  try {
    const sample = getShutter();
    const node = sample.cloneNode(true) as HTMLAudioElement;
    node.volume = sample.volume;
    node.currentTime = 0;
    void node.play();
  } catch {
    // ignore audio failures (autoplay policy, missing file, etc.)
  }
}

let introPlayed = false;

/** Game-start sting for the hello intro (once per page load). */
export function playIntroStart() {
  if (introPlayed) return;
  introPlayed = true;
  try {
    const sample = getIntro();
    sample.currentTime = 0;
    void sample.play();
  } catch {
    // ignore audio failures (autoplay policy, missing file, etc.)
  }
}
