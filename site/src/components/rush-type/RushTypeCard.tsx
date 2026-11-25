import { useEffect, useRef, type CSSProperties } from "react";
import { RushType } from "./engine";
import { FONT_VAR, FONT_WEIGHT } from "./params";

export function RushTypeCard({
  className,
  style,
  onSettled,
  replayOnTouch = true,
  dark = false,
}: {
  className?: string;
  style?: CSSProperties;
  /** Fires once the name has rushed in and settled (or at once if it can't play). */
  onSettled?: () => void;
  /** Touching the card plays the rush-in again. */
  replayOnTouch?: boolean;
  /** Pale ink on the dark page instead of dark ink on pale paper. */
  dark?: boolean;
} = {}) {
  const hostRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const reduced = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    let engine: RushType | null = null;
    let onScreen = false;
    let hidden = false;

    const sync = () => {
      if (!engine || reduced) return;
      if (onScreen && !hidden) engine.start();
      else engine.stop();
    };

    const raf = requestAnimationFrame(() => {
      if (!hostRef.current) return;
      engine = new RushType(host);
      if (!engine.ok) {
        onSettled?.();
        return;
      }
      engine.setTheme(dark);
      engine.onSettled = () => onSettled?.();

      if (reduced) {
        engine.showRest();
        onSettled?.();
      } else {
        engine.renderStill();
        sync();
      }

      if (document.fonts?.load) {
        document.fonts.load(`${FONT_WEIGHT} 1em ${FONT_VAR}`).then(
          () => engine?.refreshFont(),
          () => {},
        );
      }
    });

    const io = new IntersectionObserver(
      (es) => {
        onScreen = es[0]?.isIntersecting ?? false;
        sync();
      },
      { threshold: 0.2 },
    );
    io.observe(host);

    const onVis = () => {
      hidden = document.hidden;
      sync();
    };
    document.addEventListener("visibilitychange", onVis);

    // Touching the card plays the rush-in again.
    const grab = () => {
      if (!reduced && replayOnTouch) engine?.replay();
    };
    const release = () => {};
    host.addEventListener("pointerenter", grab);
    host.addEventListener("pointerdown", grab);
    host.addEventListener("pointerleave", release);
    host.addEventListener("pointerup", release);
    host.addEventListener("pointercancel", release);

    const ro = new ResizeObserver(() => engine?.onResize());
    ro.observe(host);

    return () => {
      cancelAnimationFrame(raf);
      io.disconnect();
      ro.disconnect();
      document.removeEventListener("visibilitychange", onVis);
      host.removeEventListener("pointerenter", grab);
      host.removeEventListener("pointerdown", grab);
      host.removeEventListener("pointerleave", release);
      host.removeEventListener("pointerup", release);
      host.removeEventListener("pointercancel", release);
      engine?.destroy();
    };
  }, []);

  return (
    <div
      ref={hostRef}
      className={className}
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        background: dark ? "#121210" : "#f5f5f5",
        overflow: "hidden",
        ...style,
      }}
    />
  );
}
