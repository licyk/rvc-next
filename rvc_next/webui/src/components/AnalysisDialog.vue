<script setup lang="ts">
import { computed, ref } from 'vue';
import { useOutputAnalysis } from '@/api/queries/outputs';
import type { Output } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import { useI18n } from '@/i18n';
import { formatDuration } from '@/format';
import { AppDialog, DataList, LineChart, ProgressBar, SegmentedControl, Spectrogram, type Series } from '@/ui';

/**
 * A result's spectrogram and pitch, beside its source's: the high-frequency cut-off of a voice,
 * noise and artefacts, octave errors, and how far the result's pitch sits from the input's.
 */
const props = defineProps<{ output: Output | null }>();
const open = defineModel<boolean>('open', { default: false });
const { t } = useI18n();
const id = computed(() => (open.value ? (props.output?.id ?? null) : null));
const hasSource = computed(() => !!props.output?.source_path);
const result = useOutputAnalysis(id);
const source = useOutputAnalysis(() => (hasSource.value ? id.value : null), true);
const which = ref<'result' | 'source'>('result');
const shown = computed(() => (which.value === 'source' ? source.data.value : result.data.value));
const options = computed(() => [
  { value: 'result' as const, label: t('analysis.result') },
  { value: 'source' as const, label: t('analysis.source') },
]);

const series = computed<Series[]>(() => {
  const out: Series[] = [];
  const line = (label: string, pitch: number[], hop: number) => ({ label, points: pitch.map((hz, i) => [i * hop, hz > 0 ? hz : Number.NaN] as [number, number]) });
  if (result.data.value) out.push(line(t('analysis.result'), result.data.value.pitch, result.data.value.pitch_hop));
  if (source.data.value) out.push(line(t('analysis.source'), source.data.value.pitch, source.data.value.pitch_hop));
  return out;
});
const shift = computed(() => {
  const a = result.data.value?.pitch_median;
  const b = source.data.value?.pitch_median;
  return a && b ? 12 * Math.log2(a / b) : null;
});
const rows = computed(() => {
  const a = shown.value;
  if (!a) return [];
  return [
    { label: t('analysis.duration'), value: formatDuration(a.duration) },
    { label: t('analysis.format'), value: `${a.sample_rate / 1000} kHz · ${a.channels === 1 ? t('analysis.mono') : t('analysis.channels', { n: a.channels })}${a.codec ? ` · ${a.codec}` : ''}` },
    { label: t('analysis.medianPitch'), value: a.pitch_median ? `${Math.round(a.pitch_median)} Hz` : t('analysis.noPitch') },
    ...(shift.value !== null ? [{ label: t('analysis.shift'), value: t('analysis.semitones', { n: (shift.value >= 0 ? '+' : '') + shift.value.toFixed(1) }) }] : []),
  ];
});
const loading = computed(() => result.isLoading.value || (hasSource.value && source.isLoading.value));
const error = computed(() => result.error.value ?? source.error.value);
</script>

<template>
  <AppDialog v-model:open="open" :title="t('analysis.title', { name: output?.name ?? '' })" width="large" :close-label="t('common.close')">
    <ProgressBar v-if="loading" :label="t('analysis.working')" />
    <ErrorNotice v-else-if="error" :error="error" />
    <div v-else class="analysis">
      <SegmentedControl v-if="hasSource" v-model="which" :options="options" />
      <template v-if="shown">
        <Spectrogram :rows="shown.spectrogram.rows" :cols="shown.spectrogram.cols" :data="shown.spectrogram.data" :label="t('analysis.spectrogram')" :height="180" />
        <p class="type-body-small muted">{{ t('analysis.spectrogramHint', { khz: (shown.spectrogram.fmax / 1000).toFixed(1) }) }}</p>
        <DataList :rows="rows" />
      </template>
      <h3 class="type-title-small sub">{{ t('analysis.pitch') }}</h3>
      <LineChart :series="series" :x-label="t('analysis.seconds')" :empty-text="t('analysis.noPitch')" :height="200" />
    </div>
  </AppDialog>
</template>

<style scoped>
.analysis { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.sub { margin: var(--app-space-2) 0 0; }
</style>
