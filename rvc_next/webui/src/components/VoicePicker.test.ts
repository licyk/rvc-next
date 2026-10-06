import { mount } from '@vue/test-utils';
import { createPinia } from 'pinia';
import { describe, expect, it, vi } from 'vitest';
import { ref } from 'vue';
import VoicePicker from '@/components/VoicePicker.vue';
import PickerMenu from '@/ui/PickerMenu.vue';

const voices = ref<Record<string, unknown>[] | undefined>(undefined);
const loading = ref(true);
vi.mock('@/api/queries/models', () => ({ useModels: () => ({ data: voices, isLoading: loading }) }));
vi.mock('vue-router', () => ({ useRouter: () => ({ push: vi.fn() }) }));

const voice = (id: string, name: string, extra: Record<string, unknown> = {}) => ({
  id, name, sample_rate: 40000, version: 'v2', pitch_guidance: true, speakers: [], tags: [], has_index: true, legacy: false, ...extra,
});
const mountPicker = () => mount(VoicePicker, { props: { modelValue: 'b' }, global: { plugins: [createPinia()] } });

describe('VoicePicker', () => {
  it('is a PickerMenu: loading first, then every voice with its details, tags searchable', async () => {
    voices.value = undefined;
    loading.value = true;
    const w = mountPicker();
    expect(w.findComponent(PickerMenu).props()).toMatchObject({ loading: true, placeholder: 'Loading voices…' });
    voices.value = [voice('a', 'Alto', { tags: ['choir'], has_index: false }), voice('b', 'Bass', { legacy: true, speakers: [{ id: 0, name: 'One' }, { id: 1, name: 'Two' }] })];
    loading.value = false;
    await w.vm.$nextTick();
    const picker = w.findComponent(PickerMenu);
    expect(picker.props('loading')).toBe(false);
    expect(picker.props('options')).toEqual([
      { value: 'a', label: 'Alto', description: '40 kHz · v2 · pitch guidance', keywords: ['choir'], badges: [{ text: 'no index', tone: 'warning' }] },
      { value: 'b', label: 'Bass', description: '40 kHz · v2 · pitch guidance · 2 speakers', keywords: [], badges: [{ text: 'legacy', tone: 'neutral' }] },
    ]);
    expect(picker.props('searchLabel')).toBe('Search voices');
  });
});
