import { QueryClient, VueQueryPlugin } from '@tanstack/vue-query';
import { mount } from '@vue/test-utils';
import { createPinia } from 'pinia';
import { describe, expect, it, vi } from 'vitest';
import { ref } from 'vue';
import type { ImportPlan, StagedFile } from '@/api/queries/imports';
import ImportReviewDialog from '@/components/ImportReviewDialog.vue';
import AppButton from '@/ui/AppButton.vue';
import SelectField from '@/ui/SelectField.vue';

vi.mock('@/api/queries/models', () => ({
  useModels: () => ({ data: ref([{ id: 'duo', name: 'Duo', speakers: [{ id: 0, name: 'a' }, { id: 1, name: 'b' }] }]) }),
}));

const file = (p: Partial<StagedFile>): StagedFile => ({
  id: 'x', name: 'x', group: null, kind: 'voice', size: 1, note: '', version: 'v2', sample_rate: '40k', pitch_guidance: true, speakers: [],
  dim: null, vectors: null, model_type: null, instruments: [], target_instrument: null, ...p,
});

// Two v2 voices and one v2 index the names say nothing about; the plan offers only those two voices.
const plan: ImportPlan = {
  session_id: 's1',
  files: [
    file({ id: 'a', name: 'alto.pth' }),
    file({ id: 'b', name: 'bass.pth' }),
    file({ id: 'i', name: 'feature.index', kind: 'index', dim: 768, vectors: 500 }),
    file({ id: 'n', name: 'notes.txt', kind: 'unsupported', note: 'not a model file' }),
  ],
  voices: [
    { file_id: 'a', name: 'alto', action: 'import' },
    { file_id: 'b', name: 'bass', action: 'import' },
  ],
  indexes: [
    {
      file_id: 'i', target: null, key: 'default', status: 'choose', confident: false, reasons: ['ambiguous'],
      candidates: [{ target: 'file:a', name: 'alto', score: 0.2 }, { target: 'file:b', name: 'bass', score: 0.2 }, { target: 'voice:duo', name: 'Duo', score: 0 }],
    },
  ],
  generators: [],
  separations: [],
  confident: false,
  created_at: '',
};

function mountDialog() {
  return mount(ImportReviewDialog, {
    props: { plan },
    global: { plugins: [createPinia(), [VueQueryPlugin, { queryClient: new QueryClient() }]] },
    attachTo: document.body,
  });
}

describe('ImportReviewDialog', () => {
  it('offers only the compatible voices for an index, plus keeping it unassigned', () => {
    const wrapper = mountDialog();
    const select = wrapper.findAllComponents(SelectField).find((s) => s.props('label') === 'Voice')!;
    expect(select.props('options').map((o: { value: string }) => o.value)).toEqual(['', 'file:a', 'file:b', 'voice:duo']);
    expect(document.body.textContent).toContain('notes.txt');
    wrapper.unmount();
  });

  it('asks for the speaker of a multi-speaker voice and sends every choice back', async () => {
    const wrapper = mountDialog();
    const voiceSelect = () => wrapper.findAllComponents(SelectField).find((s) => s.props('label') === 'Voice')!;
    voiceSelect().vm.$emit('update:modelValue', 'voice:duo');
    await wrapper.vm.$nextTick();
    const speaker = wrapper.findAllComponents(SelectField).find((s) => s.props('label') === 'Speaker')!;
    expect(speaker.props('options').map((o: { value: string }) => o.value)).toEqual(['default', 'spk0', 'spk1']);
    speaker.vm.$emit('update:modelValue', 'spk1');
    await wrapper.vm.$nextTick();
    const buttons = wrapper.findAllComponents(AppButton);
    buttons[buttons.length - 1].vm.$emit('click', new MouseEvent('click'));
    const decisions = wrapper.emitted('commit')?.[0]?.[0] as { indexes: { target: string; key: string }[]; voices: unknown[] };
    expect(decisions.indexes[0]).toMatchObject({ target: 'voice:duo', key: 'spk1' });
    expect(decisions.voices).toHaveLength(2);
    wrapper.unmount();
  });
});
