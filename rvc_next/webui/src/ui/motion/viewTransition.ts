import { nextTick, ref, type Ref } from 'vue';
import type { Router } from 'vue-router';

interface ViewTransition {
  finished: Promise<void>;
  ready?: Promise<void>;
  updateCallbackDone?: Promise<void>;
}
type StartViewTransition = (cb: () => Promise<void> | void) => ViewTransition;

/**
 * Wrap router navigation between destinations in ``document.startViewTransition`` where the browser
 * has it. The returned ref is true then, and the app renders its views without its own
 * fade-through ``<Transition>``; without the API that ``<Transition>`` is the fallback.
 *
 * The browser pauses rendering, and with it ``requestAnimationFrame``, while the update callback
 * runs, so the callback must finish on Vue's ``nextTick`` (the new view is in the DOM then), never on
 * a frame. Navigation inside one destination (a query change) is not wrapped. Reduced motion is
 * still wrapped: motion.css turns the fade through into a short crossfade, never none.
 */
export function installViewTransitions(router: Router): Ref<boolean> {
  const doc = typeof document === 'undefined' ? null : (document as Document & { startViewTransition?: StartViewTransition });
  const supported = ref(!!doc?.startViewTransition);
  if (!doc?.startViewTransition) return supported;
  const start = doc.startViewTransition.bind(doc);
  let finish: (() => void) | null = null;

  router.beforeResolve((to, from) => {
    if (!from.name || to.name === from.name) return true;
    // A navigation that interrupts an unfinished one lets the earlier transition end now.
    finish?.();
    return new Promise<boolean>((resolve) => {
      const vt = start(
        () =>
          new Promise<void>((done) => {
            finish = () => {
              finish = null;
              done();
            };
            resolve(true);
          }),
      );
      // A transition skipped by a newer one rejects these; that is expected, not an error.
      for (const p of [vt.finished, vt.ready, vt.updateCallbackDone]) p?.catch(() => undefined);
    });
  });
  router.afterEach(() => {
    if (finish) nextTick().then(() => finish?.());
  });
  router.onError(() => finish?.());
  return supported;
}
