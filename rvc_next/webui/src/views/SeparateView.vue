<script setup lang="ts">
import { computed, onActivated, ref, watch } from 'vue';
import { useRouter } from 'vue-router';
import { useSettings } from '@/api/queries/app';
import { useSeparate, useSeparationPresets } from '@/api/queries/separate';
import type { Output } from '@/api/types';
import AssetGate from '@/components/AssetGate.vue';
import AudioSourceInput from '@/components/AudioSourceInput.vue';
import ErrorNotice from '@/components/ErrorNotice.vue';
import JobCard from '@/components/JobCard.vue';
import ResultsList from '@/components/ResultsList.vue';
import { inputKey, readyRefs, type InputItem } from '@/components/inputs';
import { useSessionOutputs } from '@/components/sessionOutputs';
import { useI18n } from '@/i18n';
import { useHandoffStore } from '@/stores/handoff';
import { isFinished, useJobsStore } from '@/stores/jobs';
import { usePreferencesStore } from '@/stores/preferences';
import { AppButton, AppCard, Badge, PageFooter, SelectField, Surface, TRANSITIONS, icons } from '@/ui';

const { t } = useI18n();
const router = useRouter();
const prefs = usePreferencesStore();
const settings = useSettings();
const presets = useSeparationPresets();
const separate = useSeparate();
const jobs = useJobsStore();
const handoff = useHandoffStore();

const inputs = ref<InputItem[]>([]);
const preset = ref<string | null>(prefs.prefs.separatePreset);
const format = ref<string | null>(null);
const error = ref<unknown>(null);
watch(
  () => [settings.data.value, presets.data.value] as const,
  ([s]) => {
    if (!preset.value && s) preset.value = s.separation.default_preset;
  },
  { immediate: true },
);
watch(preset, (p) => (prefs.prefs.separatePreset = p));

const chosen = computed(() => (presets.data.value ?? []).find((p) => p.id === preset.value) ?? null);
const formatOptions = ['wav', 'flac', 'mp3', 'm4a'].map((v) => ({ value: v, label: v.toUpperCase() }));
const chosenFormat = computed(() => format.value ?? settings.data.value?.separation.output_format ?? 'flac');
const refs = computed(() => readyRefs(inputs.value));
const sessionJobs = computed(() => jobs.fromScreen('separate').filter((j) => !isFinished(j.state)));
const outputs = useSessionOutputs('separate', ['stem']);

function run() {
  if (!preset.value || !refs.value.length) return;
  error.value = null;
  separate.mutate({ inputs: refs.value, preset: preset.value, output_format: chosenFormat.value as 'flac' }, { onError: (e) => (error.value = e) });
}
function toConvert(o: Output) {
  handoff.sendInputs('convert', [{ ref: { kind: 'output', id: o.id, path: null }, name: o.name, duration: o.duration }]);
  router.push('/convert');
}
function useAsInput(o: Output) {
  inputs.value = [...inputs.value, { key: inputKey(), name: o.name, ref: { kind: 'output', id: o.id, path: null }, status: 'ready', duration: o.duration, outputId: o.id }];
}
onActivated(() => {
  const items = handoff.takeInputs('separate');
  if (items.length) inputs.value = [...inputs.value, ...items.map((i) => ({ key: inputKey(), name: i.name, ref: i.ref, duration: i.duration, status: 'ready' as const }))];
});
</script>

<template>
  <div class="separate">
    <div class="sections">
      <Surface :level="0" shape="large" class="section">
        <h2 class="type-title-medium heading">{{ t('convert.inputs') }}</h2>
        <AudioSourceInput v-model="inputs" folders />
      </Surface>
      <Surface :level="0" shape="large" class="section">
        <h2 class="type-title-medium heading">{{ t('separate.presets') }}</h2>
        <div class="grid" role="radiogroup" :aria-label="t('separate.presets')">
          <AppCard v-for="p in presets.data.value ?? []" :key="p.id" interactive :selected="p.id === preset" variant="outlined" class="preset" role="radio" :aria-checked="p.id === preset" @activate="preset = p.id">
            <div class="preset-head">
              <span class="type-title-small">{{ p.title }}</span>
              <Badge v-if="p.kind === 'chain'" :value="t('separate.chain')" tone="neutral" />
              <Badge v-if="p.source === 'imported'" :value="t('separate.imported')" tone="primary" />
            </div>
            <span class="type-body-small">{{ p.description }}</span>
            <span class="type-body-small muted">{{ t('separate.stems', { list: p.stems.join(', ') }) }}</span>
          </AppCard>
        </div>
        <div class="fields">
          <SelectField :model-value="chosenFormat" :label="t('separate.format')" :options="formatOptions" @update:model-value="format = $event" />
        </div>
      </Surface>
      <Surface :level="0" shape="large" class="section">
        <ResultsList :outputs="outputs" group-by-source @convert-this="toConvert" @use-as-input="useAsInput" />
      </Surface>
    </div>
    <PageFooter>
      <ErrorNotice v-if="error" :error="error" />
      <TransitionGroup :name="TRANSITIONS.list" tag="div" class="jobs">
        <JobCard v-for="j in sessionJobs" :key="j.id" :job="j" />
      </TransitionGroup>
      <AssetGate :assets="chosen?.assets ?? []">
        <div class="actions">
          <span v-if="!refs.length" class="type-body-small muted">{{ t('separate.noInputs') }}</span>
          <AppButton :icon="icons.Split" :disabled="!refs.length || !preset" :loading="separate.isPending.value" @click="run">{{ t('separate.separate') }}</AppButton>
        </div>
      </AssetGate>
    </PageFooter>
  </div>
</template>

<style scoped>
/* One column of sections, each as wide as the page (as Hanaikada's Settings); the footer stays in view. */
.separate { display: flex; flex-direction: column; min-height: 100%; }
.sections { display: flex; flex-direction: column; gap: var(--app-space-4); flex: 1; padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); }
.section { display: flex; flex-direction: column; gap: var(--app-space-4); padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); min-width: 0; }
.heading { margin: 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(240px, 100%), 1fr)); gap: var(--app-space-2); }
.fields { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(280px, 100%), 1fr)); gap: var(--app-space-3) var(--app-space-6); }
.preset { display: flex; flex-direction: column; gap: var(--app-space-1); padding: var(--app-space-3); min-width: 0; }
.preset-head { display: flex; align-items: center; gap: var(--app-space-2); }
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
