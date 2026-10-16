/** Named transitions and helpers for the motion system. See motion.css. */
import { nextTick, ref, watch, type Ref } from 'vue';

export const TRANSITIONS = {
  fadeThrough: 'fade-through',
  sharedAxisX: 'shared-axis-x',
  container: 'container',
  sheet: 'sheet',
  scrim: 'scrim',
  list: 'list',
  collapse: 'collapse',
  snackbar: 'snackbar',
} as const;

const STAGGER_MS = 20;
const STAGGER_CAP = 12;

/** Style for the n-th item of a list transition: a 20 ms stagger, capped. */
export function staggerStyle(index: number): Record<string, string> {
  return { '--stagger': `${Math.min(index, STAGGER_CAP) * STAGGER_MS}ms` };
}

/**
 * The way a ``shared-axis-x`` tab change moves, for ``--axis-dir`` (as Hanaikada's tag tabs): 1 to a
 * tab on the right, -1 to one on the left. Set synchronously, before the panes re-render.
 */
export function useAxisDirection<T>(value: Ref<T>, order: () => readonly T[]): Ref<number> {
  const dir = ref(1);
  watch(value, (next, prev) => (dir.value = order().indexOf(next) >= order().indexOf(prev) ? 1 : -1), { flush: 'sync' });
  return dir;
}

/**
 * Whether a meter's value last went down, for audio-meter ballistics: a rise shows at once, a fall
 * settles over a longer transition. Set synchronously, with the value it describes.
 */
export function useFalling(value: () => number): Ref<boolean> {
  const falling = ref(false);
  watch(value, (next, prev) => (falling.value = next < prev), { flush: 'sync' });
  return falling;
}

export function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined') return false;
  if (document.documentElement.dataset.motion === 'reduced') return true;
  return !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
}

/**
 * CSS variables that make a dialog grow from ``from`` (a card's rectangle) into ``to``.
 * Returns an empty object when the source rectangle is unknown, which falls back to fade and scale.
 */
export function containerFrom(from: DOMRect | null | undefined, to: DOMRect | null | undefined): Record<string, string> {
  if (!from || !to || to.width === 0 || to.height === 0) return {};
  const scaleX = from.width / to.width;
  const scaleY = from.height / to.height;
  const dx = from.left - to.left;
  const dy = from.top - to.top;
  return {
    '--from-origin': 'top left',
    '--from-transform': `translate(${dx}px, ${dy}px) scale(${scaleX}, ${scaleY})`,
  };
}

/** Hooks for <Transition name="collapse"> that animate height from and to the content's size. */
export const collapseHooks = {
  onBeforeEnter(el: Element) {
    (el as HTMLElement).style.height = '0';
  },
  onEnter(el: Element) {
    const e = el as HTMLElement;
    e.style.height = `${e.scrollHeight}px`;
  },
  onAfterEnter(el: Element) {
    (el as HTMLElement).style.height = '';
  },
  onBeforeLeave(el: Element) {
    const e = el as HTMLElement;
    e.style.height = `${e.scrollHeight}px`;
  },
  onLeave(el: Element) {
    const e = el as HTMLElement;
    void e.offsetHeight;
    e.style.height = '0';
  },
};

/**
 * ``onBeforeLeave`` for a ``list`` TransitionGroup: the leaving item is taken out of the flow
 * (``position: absolute``), which would shrink it to its content's width and drop it to wherever
 * its static position lands; pinned to its own box, it fades out exactly where it was. The list
 * (or the element around it) must be the positioned parent.
 */
export function pinLeaving(el: Element) {
  const e = el as HTMLElement;
  Object.assign(e.style, { top: `${e.offsetTop}px`, left: `${e.offsetLeft}px`, width: `${e.offsetWidth}px` });
}

/**
 * A box that eases to its new height when what it holds is added or removed, instead of jumping
 * (and taking the page under it, a sticky footer included, along). Pass the change to the returned
 * function: it holds the current height, makes the change, waits for the DOM to update, then glides
 * to the new height over the list's move time, clipping only meanwhile. Not under reduced motion.
 * The box holds one element, measured for the new height: letting the box itself collapse to
 * measure would clamp the page's scroll position when the page ends below it.
 */
export function useHeightGlide(box: Ref<HTMLElement | null | undefined>): (change: () => void) => Promise<void> {
  const release = (el: HTMLElement) => {
    el.style.height = el.style.overflow = el.style.transition = '';
    el.removeEventListener('transitionend', onEnd);
  };
  function onEnd(e: Event) {
    const el = e.currentTarget as HTMLElement;
    if (e.target === el && (e as TransitionEvent).propertyName === 'height') release(el);
  }
  return async (change) => {
    const el = box.value;
    if (!el || prefersReducedMotion()) return change();
    // Mid-glide, this is the height now shown, so a second change carries on from there.
    const from = el.getBoundingClientRect().height;
    el.style.transition = 'none';
    el.style.height = `${from}px`;
    el.style.overflow = 'clip';
    // The change queues the re-render, so the tick waited for is the one that applies it.
    change();
    await nextTick();
    const to = el.firstElementChild?.getBoundingClientRect().height ?? 0;
    if (Math.abs(to - from) < 0.5) return release(el);
    void el.offsetHeight;
    el.addEventListener('transitionend', onEnd);
    el.style.transition = 'height var(--md-sys-motion-duration-medium2) var(--md-sys-motion-easing-emphasized)';
    el.style.height = `${to}px`;
    if (parseFloat(getComputedStyle(el).transitionDuration) === 0) release(el);
  };
}
