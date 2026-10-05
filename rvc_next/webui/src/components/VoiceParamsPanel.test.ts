import { QueryClient, VueQueryPlugin } from '@tanstack/vue-query';
import { mount } from '@vue/test-utils';
import { createPinia } from 'pinia';
import { describe, expect, it, vi } from 'vitest';
import { nextTick, ref } from 'vue';
import { PARAM_SPECS } from '@/api/paramSpecs';
import type { VoiceModel, VoiceParams } from '@/api/types';
import VoiceParamsPanel from '@/components/VoiceParamsPanel.vue';
import { defaults } from '@/components/paramFields';
import ParamSlider from '@/ui/ParamSlider.vue';

const defaultPreset = { id: 'p1', name: 'Default', voice_id: 'v1', is_default: true, params: { ...defaults<VoiceParams>('voice'), pitch: 7 }, stream: null, updated_at: '' };
vi.mock('@/api/queries/presets', () => ({
  usePresets: () => ({ data: ref([defaultPreset]) }),
  usePresetMutations: () => ({ create: { mutate: vi.fn(), isPending: ref(false) } }),
}));

const voice = (p: Partial<VoiceModel> = {}): VoiceModel => ({
  id: 'v1', name: 'Alto', description: '', tags: [], location: 'library', legacy: false, model_path: '/m.pth', sample_rate: 40000, version: 'v2', pitch_guidance: true,
  speakers: [], speaker_slots: 109, indexes: {}, has_index: true, size: 1, info: '', hidden: false, catalog_id: null, created_at: '', updated_at: '', ...p,
});

function mountPanel(v: VoiceModel) {
  const model = ref(defaults<VoiceParams>('voice'));
  const wrapper = mount(VoiceParamsPanel, {
    props: { voice: v, modelValue: model.value, 'onUpdate:modelValue': (x: VoiceParams) => (model.value = x) },
    global: { plugins: [createPinia(), [VueQueryPlugin, { queryClient: new QueryClient() }]] },
  });
  return { wrapper, model };
}

describe('VoiceParamsPanel', () => {
  it('takes every range from the server schema, in the one order', () => {
    const { wrapper } = mountPanel(voice());
    const sliders = wrapper.findAllComponents(ParamSlider);
    const keys = ['pitch', 'formant', 'index_rate', 'protect', 'rms_mix_rate'];
    expect(sliders).toHaveLength(keys.length);
    sliders.forEach((s, i) => {
      expect(s.props('min')).toBe(PARAM_SPECS.voice[keys[i]].min);
      expect(s.props('max')).toBe(PARAM_SPECS.voice[keys[i]].max);
      expect(s.props('defaultValue')).toBe(PARAM_SPECS.voice[keys[i]].default);
    });
  });

  it('loads the voice default preset, and says why a control is off', async () => {
    const { wrapper, model } = mountPanel(voice({ has_index: false, pitch_guidance: false }));
    await nextTick();
    expect(model.value.pitch).toBe(7);
    const sliders = wrapper.findAllComponents(ParamSlider);
    expect(sliders[2].props('disabled')).toBe(true);
    expect(sliders[2].props('disabledReason')).toMatch(/no index/);
    expect(sliders[3].props('disabled')).toBe(true);
    expect(sliders[0].props('disabled')).toBe(false);
  });
});
