<script setup lang="ts">
import { computed } from 'vue';
import type { StreamParams } from '@/api/types';
import { STREAM_FIELDS, latencyShare, spec } from '@/components/paramFields';
import { useI18n } from '@/i18n';
import { ParamSlider, Switch } from '@/ui';

/** Live buffering and clean-up; each setting shows its share of the estimated latency. */
const props = defineProps<{ disabled?: boolean }>();
const model = defineModel<StreamParams>({ required: true });
const emit = defineEmits<{ change: [StreamParams] }>();
const { t } = useI18n();

const fields = computed(() =>
  STREAM_FIELDS.map((f) => {
    const s = spec('stream', f.key);
    const share = latencyShare(f.key, model.value[f.key] as number, model.value.crossfade_ms);
    const help = t(f.help) + (share !== null ? ` (+${share} ms)` : '');
    return { ...f, min: s.min, max: s.max, default: s.default as number, help };
  }),
);

function set(key: keyof StreamParams, value: number | boolean, commit: boolean) {
  const next = { ...model.value, [key]: value } as StreamParams;
  model.value = next;
  if (commit) emit('change', next);
}
const denoiseIn = computed({ get: () => model.value.input_denoise, set: (v) => set('input_denoise', v, true) });
const denoiseOut = computed({ get: () => model.value.output_denoise, set: (v) => set('output_denoise', v, true) });
const denoiseShare = computed(() => latencyShare('input_denoise', model.value.input_denoise ? 1 : 0, model.value.crossfade_ms));
const phaseVocoder = computed({ get: () => model.value.phase_vocoder, set: (v) => set('phase_vocoder', v, true) });
const strength = computed(() => {
  const s = spec('stream', 'denoise_strength');
  return { min: s.min, max: s.max, default: s.default as number };
});
</script>

<template>
  <section class="stream" :aria-label="t('stream.title')">
    <h2 class="type-title-medium title">{{ t('stream.title') }}</h2>
    <div class="fields">
      <ParamSlider
        v-for="f in fields"
        :key="f.key"
        :model-value="model[f.key] as number"
        :label="t(f.label)"
        :help="f.help"
        :min="f.min"
        :max="f.max"
        :step="f.step"
        :unit="f.unit"
        :default-value="f.default"
        :disabled="props.disabled"
        :reset-label="t('common.reset')"
        @update:model-value="set(f.key, $event, false)"
        @commit="set(f.key, $event, true)"
      />
    </div>
    <div class="fields">
      <Switch v-model="denoiseIn" :label="t('stream.inputDenoise')" :supporting-text="denoiseShare ? `+${denoiseShare} ms` : ''" :disabled="props.disabled" />
      <Switch v-model="denoiseOut" :label="t('stream.outputDenoise')" :disabled="props.disabled" />
      <ParamSlider
        v-if="model.input_denoise || model.output_denoise"
        :model-value="model.denoise_strength"
        :label="t('stream.denoiseStrength')"
        :help="t('stream.denoiseStrengthHelp')"
        :min="strength.min"
        :max="strength.max"
        :step="0.05"
        unit=""
        :default-value="strength.default"
        :disabled="props.disabled"
        :reset-label="t('common.reset')"
        @update:model-value="set('denoise_strength', $event, false)"
        @commit="set('denoise_strength', $event, true)"
      />
      <Switch v-model="phaseVocoder" :label="t('stream.phaseVocoder')" :supporting-text="t('stream.phaseVocoderHelp')" :disabled="props.disabled" />
    </div>
  </section>
</template>

<style scoped>
.stream { display: flex; flex-direction: column; gap: var(--app-space-2); min-width: 0; }
.title { margin: 0; }
/* The sliders wrap: a wide page shows more of them per row, never longer ones. */
.fields { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(320px, 100%), 1fr)); gap: var(--app-space-2) var(--app-space-6); align-items: start; }
</style>
