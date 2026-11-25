export const THEME_KEY = 'portfolio-theme';

/** The pick made with the toggle this visit, if any. New visits start from the device setting. */
export function readStoredTheme(): 'dark' | 'light' | null {
  try {
    const v = window.sessionStorage.getItem(THEME_KEY);
    return v === 'dark' || v === 'light' ? v : null;
  } catch {
    return null; // storage blocked
  }
}

export const systemDarkQuery = () => window.matchMedia('(prefers-color-scheme: dark)');

/** Their saved pick wins; otherwise follow the device's light/dark setting. */
export function resolveDark(): boolean {
  const stored = readStoredTheme();
  return stored ? stored === 'dark' : systemDarkQuery().matches;
}

export function saveTheme(dark: boolean) {
  try {
    window.sessionStorage.setItem(THEME_KEY, dark ? 'dark' : 'light');
  } catch {
    /* storage blocked: the choice lasts this visit only */
  }
}
