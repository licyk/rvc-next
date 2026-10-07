import { flushPromises, mount } from '@vue/test-utils';
import { QueryClient, VueQueryPlugin } from '@tanstack/vue-query';
import { createPinia } from 'pinia';
import { describe, expect, it, vi } from 'vitest';
import type { EffectModel } from '@/api/types';
import EffectsPanel from '@/components/EffectsPanel.vue';

const catalog = {
  available: true,
  effects: [
    { kind: 'reverb', live: true, params: [{ name: 'room_size', default: 0.4, min: 0, max: 1, unit: '' }] },
    { kind: 'gain', live: true, params: [{ name: 'gain_db', default: 0, min: -24, max: 24, unit: 'dB' }] },
  ],
};

vi.mock('@/api/client', () => ({
  api: { GET: vi.fn(async () => ({ data: catalog })) },
  unwrap: async (p: Promise<{ data: unknown }>) => (await p).data,
}));

function mountPanel(effects: EffectModel[]) {
  const client = new QueryClient();
  return mount(EffectsPanel, { props: { modelValue: effects }, global: { plugins: [createPinia(), [VueQueryPlugin, { queryClient: client }]] } });
}

describe('EffectsPanel', () => {
  it('shows each effect with its parameters, and moves and removes them', async () => {
    const w = mountPanel([
      { kind: 'reverb', params: {} },
      { kind: 'gain', params: { gain_db: -6 } },
    ]);
    await flushPromises();
    expect(w.text()).toContain('1. Reverb');
    expect(w.text()).toContain('2. Gain');
    expect(w.text()).toContain('-6 dB');
    const buttons = w.findAll('[title="Move up"]');
    await buttons[1]!.find('*').trigger('click');
    const moved = w.emitted('change')![0]![0] as EffectModel[];
    expect(moved.map((e) => e.kind)).toEqual(['gain', 'reverb']);
    await w.findAll('[title="Remove"]')[0]!.find('*').trigger('click');
    expect((w.emitted('change')![1]![0] as EffectModel[]).map((e) => e.kind)).toEqual(['reverb']);
  });

  it('says how to install pedalboard when it is missing', async () => {
    catalog.available = false;
    const w = mountPanel([]);
    await flushPromises();
    expect(w.text()).toContain('pip install rvc-next[effects]');
    catalog.available = true;
  });
});
