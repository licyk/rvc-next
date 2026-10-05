import {
  AudioLines,
  CheckCircle2,
  ChevronRight,
  GraduationCap,
  ListChecks,
  Mic,
  Pause,
  Play,
  Plus,
  Search,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { useSiteCopy } from "@/lib/site-copy";

const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)";

/** A sketch of the voice library: one voice converting, one live, one in training. */
const PREVIEW_VOICES = [
  { name: "kikiV1", tone: "running", operationIcon: AudioLines },
  { name: "zhanzhanv2-xi", tone: "ready", operationIcon: Mic },
  { name: "my-voice", tone: "maintenance", operationIcon: GraduationCap },
] as const;

export default function ProductWorkspacePreview() {
  const copy = useSiteCopy().home.preview;
  const rootRef = useRef<HTMLElement>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [isAutoPlaying, setIsAutoPlaying] = useState(true);
  const [isInteractionPaused, setIsInteractionPaused] = useState(false);
  const [isPageVisible, setIsPageVisible] = useState(true);
  const [isOnScreen, setIsOnScreen] = useState(false);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    const media = window.matchMedia(REDUCED_MOTION_QUERY);

    function syncMotionPreference() {
      setPrefersReducedMotion(media.matches);
    }

    syncMotionPreference();
    media.addEventListener("change", syncMotionPreference);
    return () => media.removeEventListener("change", syncMotionPreference);
  }, []);

  useEffect(() => {
    function syncPageVisibility() {
      setIsPageVisible(document.visibilityState === "visible");
    }

    syncPageVisibility();
    document.addEventListener("visibilitychange", syncPageVisibility);
    return () => document.removeEventListener("visibilitychange", syncPageVisibility);
  }, []);

  // Scrolled past, the preview has nothing to show; stop advancing rather than
  // re-rendering and running the progress animation for no one.
  useEffect(() => {
    const root = rootRef.current;
    if (!root || typeof window.IntersectionObserver !== "function") {
      setIsOnScreen(true);
      return;
    }

    const observer = new IntersectionObserver(([entry]) => setIsOnScreen(entry.isIntersecting), {
      threshold: 0.2,
    });

    observer.observe(root);
    return () => observer.disconnect();
  }, []);

  /*
   * The progress bar is the clock. Advancing on its `animationend` instead of a
   * parallel `setTimeout` means pausing the animation pauses the cycle exactly,
   * with no second timer that could drift out of step with what is on screen.
   */
  const isCycling = isAutoPlaying && isOnScreen && isPageVisible && !prefersReducedMotion;

  function advance() {
    setActiveIndex((index) => (index + 1) % PREVIEW_VOICES.length);
  }

  const activeInstance = PREVIEW_VOICES[activeIndex];
  const activeCopy = copy.voices[activeIndex];
  const OperationIcon = activeInstance.operationIcon;

  return (
    <figure
      className={`product-preview page-interactive${isCycling ? " is-cycling" : ""}`}
      aria-label={copy.label}
      ref={rootRef}
      onBlurCapture={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
          setIsInteractionPaused(false);
        }
      }}
      onFocusCapture={() => setIsInteractionPaused(true)}
      onMouseEnter={() => setIsInteractionPaused(true)}
      onMouseLeave={() => setIsInteractionPaused(false)}
    >
      <div className="product-preview-window page-interactive-part">
        <div className="product-preview-titlebar">
          <div className="product-preview-window-controls" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div className="product-preview-brand">
            <span className="product-preview-brand-mark" aria-hidden="true">
              <AudioLines />
            </span>
            RVC Next
          </div>
          <div className="product-preview-titlebar-actions">
            <ListChecks aria-hidden="true" />
            <button
              aria-label={isAutoPlaying ? copy.pause : copy.play}
              className="product-preview-action page-interactive"
              onClick={() => setIsAutoPlaying((playing) => !playing)}
              title={isAutoPlaying ? copy.pause : copy.play}
              type="button"
            >
              {isAutoPlaying ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
            </button>
          </div>
        </div>

        <div className="product-preview-body">
          <div className="product-preview-heading">
            <div>
              <span>{copy.workspace}</span>
              <strong>{copy.select}</strong>
            </div>
            <span className="product-preview-new-instance">
              <Plus aria-hidden="true" />
              {copy.create}
            </span>
          </div>

          <div className="product-preview-search">
            <Search aria-hidden="true" />
            <span>{copy.search}</span>
            <kbd>⌘ K</kbd>
          </div>

          <div className="product-preview-instances">
            {PREVIEW_VOICES.map((instance, index) => {
              const instanceCopy = copy.voices[index];
              return (
                <button
                  aria-pressed={activeIndex === index}
                  className={`product-preview-instance page-interactive ${
                    activeIndex === index ? "is-active" : ""
                  }`}
                  key={instance.name}
                  onClick={() => setActiveIndex(index)}
                  type="button"
                >
                  <span className="product-preview-instance-topline">
                    <span className={`product-preview-state is-${instance.tone}`}>
                      <span aria-hidden="true" />
                      {instanceCopy.status}
                    </span>
                    <ChevronRight className="page-interactive-part" aria-hidden="true" />
                  </span>
                  <strong>{instance.name}</strong>
                  <small>{instanceCopy.detail}</small>
                </button>
              );
            })}
          </div>

          <div className="product-preview-operation">
            <span className="product-preview-operation-icon" aria-hidden="true">
              <OperationIcon />
            </span>
            <div className="product-preview-operation-copy" key={activeInstance.name}>
              <strong>{activeCopy.operationTitle}</strong>
              <span>
                <CheckCircle2 aria-hidden="true" />
                {activeCopy.operationDetail}
              </span>
            </div>
            <div
              className={`product-preview-progress${isCycling ? " is-cycling" : ""}${
                isInteractionPaused ? " is-paused" : ""
              }`}
              aria-hidden="true"
            >
              <span key={activeIndex} onAnimationEnd={advance} />
            </div>
          </div>
        </div>
      </div>
    </figure>
  );
}
