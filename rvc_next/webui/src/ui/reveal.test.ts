import { describe, expect, it, vi } from 'vitest';
import { revealDropdown } from '@/ui/reveal';

function box(top: number, bottom: number): DOMRect {
  return { top, bottom, left: 0, right: 100, width: 100, height: bottom - top, x: 0, y: top, toJSON: () => ({}) } as DOMRect;
}

/** A scroller (0–800) holding a dropdown and, optionally, a sticky footer from 700. */
function page(list: [number, number], footer: boolean) {
  const scroller = document.createElement('main');
  scroller.style.overflowY = 'auto';
  scroller.getBoundingClientRect = () => box(0, 800);
  scroller.scrollBy = vi.fn() as typeof scroller.scrollBy;
  const el = document.createElement('div');
  el.getBoundingClientRect = () => box(...list);
  scroller.append(el);
  if (footer) {
    const f = document.createElement('footer');
    f.dataset.pageFooter = '';
    f.getBoundingClientRect = () => box(700, 800);
    scroller.append(f);
  }
  document.body.append(scroller);
  return { scroller, el };
}

describe('revealDropdown', () => {
  it('scrolls a list that reaches under the page footer up until it clears it', () => {
    const { scroller, el } = page([500, 760], true);
    revealDropdown(el);
    expect(scroller.scrollBy).toHaveBeenCalledWith(expect.objectContaining({ top: 68 }));
  });

  it('leaves a list that already fits alone', () => {
    const { scroller, el } = page([300, 600], true);
    revealDropdown(el);
    expect(scroller.scrollBy).not.toHaveBeenCalled();
  });

  it('never scrolls past the list top', () => {
    const { scroller, el } = page([40, 900], false);
    revealDropdown(el);
    expect(scroller.scrollBy).toHaveBeenCalledWith(expect.objectContaining({ top: 32 }));
  });
});
