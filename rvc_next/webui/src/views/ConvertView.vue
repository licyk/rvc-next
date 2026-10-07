<script setup lang="ts">
import { computed, onActivated, ref } from 'vue';

import { useSettings } from '@/api/queries/app';
import { useConvert } from '@/api/queries/convert';
import { useSeparationPresets } from '@/api/queries/separate';
import type { EffectModel, Output, VoiceModel, VoiceParams } from '@/api/types';
import AssetGate from '@/components/AssetGate.vue';
import AudioSourceInput from '@/components/AudioSourceInput.vue';
import EffectsPanel from '@/components/EffectsPanel.vue';
import ErrorNotice from '@/components/ErrorNotice.vue';
import JobCard from '@/components/JobCard.vue';
import ResultsList from '@/components/ResultsList.vue';
import VoiceParamsPanel from '@/components/VoiceParamsPanel.vue';
import VoicePicker from '@/components/VoicePicker.vue';
import { inputKey, readyRefs, type InputItem } from '@/components/inputs';
import { defaults, embedderAsset, pitchAssets } from '@/components/paramFields';
import { useSessionOutputs } from '@/components/sessionOutputs';
import { useI18n } from '@/i18n';
import { useHandoffStore } from '@/stores/handoff';
import { isFinished, useJobsStore } from '@/stores/jobs';
import { usePreferencesStore } from '@/stores/preferences';
import { AppButton, ExpansionPanel, PageFooter, ParamSlider, SelectField, Surface, Switch, TRANSITIONS, icons, useSnackbar } from '@/ui';

const { t } = useI18n();
const prefs = usePreferencesStore();
const jobs = useJobsStore();
const handoff = useHandoffStore();
const settings = useSettings();
const sepPresets = useSeparationPresets();
const convert = useConvert();
const snackbar = useSnackbar();

const inputs = ref<InputItem[]>([]);
const voiceId = ref<string | null>(prefs.prefs.convertVoice);
const voice = ref<VoiceModel | null>(null);
const params = ref<VoiceParams>(defaults<VoiceParams>('voice'));
const separate = ref(false);
const sepPreset = ref<string | null>('vocals');
const remix = ref(false);
const remixGain = ref(0);
const format = ref<string | null>(null);
const resample = ref<string | null>('');
const denoise = ref(0);
const effects = ref<EffectModel[]>([]);
const stepsOpen = ref(false);
const error = ref<unknown>(null);

const formatOptions = ['wav', 'flac', 'mp3', 'm4a', 'ogg'].map((v) => ({ value: v, label: v.toUpperCase() }));
const resampleOptions = computed(() => [{ value: '', label: t('convert.resampleOff') }, ...[22050, 32000, 40000, 44100, 48000].map((r) => ({ value: String(r), label: `${r / 1000} kHz` }))]);
const presetOptions = computed(() => (sepPresets.data.value ?? []).map((p) => ({ value: p.id, label: p.title })));
const chosenFormat = computed(() => format.value ?? settings.data.value?.convert.output_format ?? 'wav');
const speaker = computed({ get: () => params.value.speaker_id, set: (v: number) => (params.value = { ...params.value, speaker_id: v }) });

function onVoice(v: VoiceModel | null) {
  voice.value = v;
  if (v) prefs.prefs.convertVoice = v.id;
}

const required = computed(() => {
  const ids = [embedderAsset(voice.value?.embedder), ...pitchAssets(voice.value?.pitch_guidance, params.value.f0_method)];
  if (separate.value) ids.push(...((sepPresets.data.value ?? []).find((p) => p.id === sepPreset.value)?.assets ?? []));
  return ids;
});
const refs = computed(() => readyRefs(inputs.value));
const canRun = computed(() => !!voice.value && refs.value.length > 0 && !inputs.value.some((i) => i.status === 'uploading'));
const sessionJobs = computed(() => jobs.fromScreen('convert').filter((j) => !isFinished(j.state)));
const outputs = useSessionOutputs('convert', ['converted', 'remix', 'preview']);

function run(preview: boolean) {
  if (!voice.value || !canRun.value) return;
  error.value = null;
  convert.mutate(
    {
      inputs: refs.value,
      voice_id: voice.value.id,
      params: params.value,
      separate: separate.value && sepPreset.value ? { preset: sepPreset.value } : null,
      remix: separate.value && remix.value ? { gain_db: remixGain.value, vocal_gain_db: 0 } : null,
      output_format: chosenFormat.value as 'wav',
      resample_to: resample.value ? Number(resample.value) : null,
      output_denoise: denoise.value,
      effects: effects.value,
      preview_seconds: preview ? 15 : null,
    },
    { onError: (e) => (error.value = e) },
  );
}

function useAsInput(o: Output) {
  inputs.value = [...inputs.value, { key: inputKey(), name: o.name, ref: { kind: 'output', id: o.id, path: null }, status: 'ready', duration: o.duration, outputId: o.id }];
}
function rerun(o: Output) {
  if (o.model_id) voiceId.value = o.model_id;
  if (o.voice) params.value = { ...o.voice };
  // The input it was made from, unless it is already queued.
  if (o.source_name && !inputs.value.some((i) => i.ref?.kind === 'output-source' && i.ref?.id === o.id)) {
    inputs.value = [...inputs.value, { key: inputKey(), name: o.source_name, ref: { kind: 'output-source', id: o.id, path: null }, status: 'ready', duration: o.duration }];
  }
  snackbar.show(t('results.rerun'));
}
function convertThis(o: Output) {
  useAsInput(o);
}

// "Use as input" from another screen, and "Try" from Train, arrive here.
onActivated(() => {
  const items = handoff.takeInputs('convert');
  if (items.length) inputs.value = [...inputs.value, ...items.map((i) => ({ key: inputKey(), name: i.name, ref: i.ref, duration: i.duration, status: 'ready' as const, outputId: i.ref.kind === 'output' ? i.ref.id : null }))];
  const v = handoff.takeVoice();
  if (v) {
    voiceId.value = v.id;
    if (v.params) params.value = { ...v.params };
  }
});
</script>

<template>
  <div class="convert">
    <div class="sections">
      <Surface :level="0" shape="large" class="section">
        <h2 class="type-title-medium heading">{{ t('convert.inputs') }}</h2>
        <AudioSourceInput v-model="inputs" folders tts />
      </Surface>
      <Surface :level="0" shape="large" class="section">
        <VoicePicker v-model="voiceId" v-model:speaker="speaker" @voice="onVoice" />
        <VoiceParamsPanel v-model="params" :voice="voice" />
        <ExpansionPanel v-model:open="stepsOpen" :label="t('convert.steps')" :icon="icons.SlidersHorizontal">
          <div class="steps">
            <div class="fields">
              <Switch v-model="separate" :label="t('convert.separateFirst')" :supporting-text="t('convert.separateHint')" />
              <SelectField v-if="separate" v-model="sepPreset" :label="t('separate.presets')" :options="presetOptions" />
              <Switch v-if="separate" v-model="remix" :label="t('convert.addBack')" />
              <ParamSlider v-if="separate && remix" v-model="remixGain" :label="t('convert.accompanimentGain')" :min="-24" :max="12" unit="dB" :default-value="0" :reset-label="t('common.reset')" />
            </div>
            <h3 class="type-title-small sub">{{ t('convert.output') }}</h3>
            <div class="fields">
              <SelectField :model-value="chosenFormat" :label="t('convert.format')" :options="formatOptions" @update:model-value="format = $event" />
              <SelectField v-model="resample" :label="t('convert.resample')" :options="resampleOptions" />
              <ParamSlider v-model="denoise" :label="t('convert.denoise')" :help="t('convert.denoiseHelp')" :min="0" :max="1" :step="0.05" unit="" :default-value="0" :reset-label="t('common.reset')" />
            </div>
            <EffectsPanel v-model="effects" />
          </div>
        </ExpansionPanel>
      </Surface>
      <Surface :level="0" shape="large" class="section">
        <ResultsList :outputs="outputs" @use-as-input="useAsInput" @rerun="rerun" @convert-this="convertThis" />
      </Surface>
    </div>
    <PageFooter>
      <ErrorNotice v-if="error" :error="error" />
      <TransitionGroup :name="TRANSITIONS.list" tag="div" class="jobs">
        <JobCard v-for="j in sessionJobs" :key="j.id" :job="j" />
      </TransitionGroup>
      <AssetGate :assets="required">
        <div class="actions">
          <span v-if="!voice" class="type-body-small muted">{{ t('convert.noVoice') }}</span>
          <span v-else-if="!refs.length" class="type-body-small muted">{{ t('convert.noInputs') }}</span>
          <AppButton variant="tonal" :icon="icons.Play" :disabled="!canRun" @click="run(true)">{{ t('convert.preview') }}</AppButton>
          <AppButton :icon="icons.AudioLines" :disabled="!canRun" :loading="convert.isPending.value" @click="run(false)">{{ t('convert.convert') }}</AppButton>
        </div>
      </AssetGate>
    </PageFooter>
  </div>
</template>

<style scoped>
/* One column of sections, each as wide as the page (as Hanaikada's Settings); the footer stays in view. */
.convert { display: flex; flex-direction: column; min-height: 100%; }
.sections { display: flex; flex-direction: column; gap: var(--app-space-4); flex: 1; padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); }
.section { display: flex; flex-direction: column; gap: var(--app-space-4); padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); min-width: 0; }
.heading { margin: 0; }
.steps { display: flex; flex-direction: column; gap: var(--app-space-3); padding: var(--app-space-2) 0; }
.fields { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(280px, 100%), 1fr)); gap: var(--app-space-3) var(--app-space-6); align-items: center; }
.sub { margin: var(--app-space-2) 0 0; }
.actions { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: var(--app-space-2); }
.actions > span { margin-right: auto; }
/* Several jobs scroll inside the footer rather than push the page out of view. */
.jobs { position: relative; display: flex; flex-direction: column; gap: var(--app-space-2); max-height: 40vh; overflow-y: auto; }
.jobs:empty { display: none; }
@container app-content (max-width: 599px) {
  .sections { padding: var(--app-space-3); }
  .section { padding: var(--app-space-4); }
}
</style>
