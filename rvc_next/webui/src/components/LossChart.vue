<script setup lang="ts">
import { computed } from 'vue';
import type { TrainMetric } from '@/api/types';
import { useI18n } from '@/i18n';
import { LineChart, type Series } from '@/ui';

/** Loss curves of a training run, by step. The total losses lead; the components follow. */
const props = defineProps<{ metrics: TrainMetric[] }>();
const { t } = useI18n();
const ORDER = ['g_total', 'd_total', 'mel', 'kl', 'fm', 'gen'];
const series = computed<Series[]>(() => {
  const keys = new Set<string>();
  for (const m of props.metrics) for (const k of Object.keys(m.losses)) keys.add(k);
  const sorted = [...keys].sort((a, b) => (ORDER.indexOf(a) + 100 * +(ORDER.indexOf(a) < 0)) - (ORDER.indexOf(b) + 100 * +(ORDER.indexOf(b) < 0)));
  return sorted.map((k) => ({ label: k, points: props.metrics.filter((m) => k in m.losses).map((m) => [m.step, m.losses[k]] as [number, number]) }));
});
</script>

<template>
  <LineChart :series="series" :x-label="'step'" :empty-text="t('train.noCheckpoints')" />
</template>
