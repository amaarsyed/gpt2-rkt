import { createContext, useContext } from 'react';

/** Lets a .rkt term in the text point the pinned graph at its module while its card is open. */
export const TermFocusContext = createContext<(moduleId: string | null) => void>(() => {});
export const useTermFocus = () => useContext(TermFocusContext);
