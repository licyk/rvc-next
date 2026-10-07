import { mount } from '@vue/test-utils';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { defineComponent, h, KeepAlive, nextTick, ref } from 'vue';

const post = vi.fn((_path: string, _init: { body: { on: boolean } }) => Promise.resolve({}));
vi.mock('@/api/client', () => ({ api: { POST: (path: string, init: { body: { on: boolean } }) => post(path, init) }, unwrap: (x: unknown) => x }));
const { useMeterClaim } = await import('@/api/queries/live');

const asked = () => post.mock.calls.map(([, init]) => init.body.on);
const Panel = defineComponent({
  props: { wanted: { type: Boolean, default: true } },
  setup(props) {
    useMeterClaim(() => props.wanted);
    return () => h('div');
  },
});

describe('useMeterClaim', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    post.mockClear();
  });
  afterEach(() => vi.useRealTimers());

  it('asks once for every panel shown, renews the lease, and lets it go with the last panel', async () => {
    const a = mount(Panel);
    const b = mount(Panel);
    expect(asked()).toEqual([true]);
    vi.advanceTimersByTime(5000);
    expect(asked()).toEqual([true, true]);
    a.unmount();
    expect(asked()).toEqual([true, true]);
    b.unmount();
    expect(asked()).toEqual([true, true, false]);
    vi.advanceTimersByTime(20000);
    expect(asked()).toEqual([true, true, false]);
  });

  it('follows the setting, and a view kept alive in the background holds no claim', async () => {
    const wanted = ref(false);
    const shown = ref(true);
    const Host = defineComponent({ setup: () => () => h(KeepAlive, null, [shown.value ? h(Panel, { wanted: wanted.value }) : h('span')]) });
    const host = mount(Host);
    expect(asked()).toEqual([]);
    wanted.value = true;
    await nextTick();
    expect(asked()).toEqual([true]);
    shown.value = false;
    await nextTick();
    expect(asked()).toEqual([true, false]);
    shown.value = true;
    await nextTick();
    expect(asked()).toEqual([true, false, true]);
    host.unmount();
    expect(asked()).toEqual([true, false, true, false]);
  });
});
