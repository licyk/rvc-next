import { nextTick, onBeforeUnmount, watch, type Ref } from 'vue';
import { prefersReducedMotion } from '@/ui/motion/transitions';

const MARGIN = 8;

function scrollParent(el: HTMLElement): HTMLElement | null {
  for (let p = el.parentElement; p; p = p.parentElement) {
    const overflow = getComputedStyle(p).overflowY;
    if (overflow === 'auto' || overflow === 'scroll') return p;
  }
  return null;
}

/**
 * Scroll a dropdown that lists in place (the voice picker, a device menu) fully into view, clear
 * of the page's sticky footer (``data-page-footer``): a list opened near the bottom moves up
 * instead of hiding under the Start bar or covering it. A list taller than the room shows from
 * its top.
 */
export function revealDropdown(el: HTMLElement | null | undefined): void {
  if (!el) return;
  const box = scrollParent(el);
  if (!box) return;
  const view = box.getBoundingClientRect();
  const footer = box.querySelector<HTMLElement>('[data-page-footer]');
  const bottom = footer ? Math.min(view.bottom, footer.getBoundingClientRect().top) : view.bottom;
  const rect = el.getBoundingClientRect();
  const delta = Math.min(rect.bottom + MARGIN - bottom, rect.top - view.top - MARGIN);
  if (delta > 0) box.scrollBy({ top: delta, behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
}

/**
 * Keep a dropdown revealed while it is open: on opening, and again whenever its size changes (the
 * search field inside is a web component that lays out a moment later; a search shortens the list).
 */
export function useRevealWhileOpen(open: () => boolean, el: Ref<HTMLElement | null>): void {
  let observer: ResizeObserver | null = null;
  const stop = () => {
    observer?.disconnect();
    observer = null;
  };
  watch(open, async (isOpen) => {
    stop();
    if (!isOpen) return;
    await nextTick();
    const target = el.value;
    if (!target) return;
    revealDropdown(target);
    if (typeof ResizeObserver === 'undefined') return;
    observer = new ResizeObserver(() => revealDropdown(target));
    observer.observe(target);
  });
  onBeforeUnmount(stop);
}
