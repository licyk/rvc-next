<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from '@/i18n';
import { useLiveStore } from '@/stores/live';
import { AppButton, Badge, LevelMeter, ProgressBar } from '@/ui';

/**
 * Input and output meters, the load bar (inference time over block length, warning above 0.8),
 * latency (estimated, and measured when a loopback measurement fits the session) and under/overruns.
 * The bars glide between the stats (10 per second).
 */
const emit = defineEmits<{ increaseBlock: [number] }>();
const live = useLiveStore();
const { t } = useI18n();
const s = computed(() => live.stats);
const measured = computed(() => (live.running && live.state.latency_test?.ok ? live.state.latency_test.latency_ms : null));
const loadHigh = computed(() => live.load > 0.8);
const suggested = computed(() => (s.value ? Math.min(1500, Math.ceil((s.value.infer_ms_p95 / 0.7) / 10) * 10) : 0));
</script>

<template>
  <div class="stats">
    <div class="meters">
      <LevelMeter :label="t('live.input')" :rms-db="s?.input_rms_db ?? null" :peak-db="s?.input_peak_db ?? null" />
      <LevelMeter :label="t('live.output')" :rms-db="s?.output_rms_db ?? null" :peak-db="s?.output_peak_db ?? null" />
    </div>
    <div class="load">
      <span class="type-label-medium muted">{{ t('live.load') }}</span>
      <ProgressBar :value="Math.min(1, live.load)" :label="t('live.load')" :tone="loadHigh ? 'error' : 'primary'" class="bar" />
      <span class="type-label-large num">{{ live.load > 1 ? '>100' : Math.round(live.load * 100) }}%</span>
    </div>
    <span class="type-label-large num">{{ t('live.latency') }} {{ s ? Math.round(s.est_latency_ms) : '—' }} ms</span>
    <span v-if="measured !== null" class="type-label-large num">{{ t('live.measured', { ms: Math.round(measured) }) }}</span>
    <Badge v-if="s && s.underruns" :value="t('live.underruns', { n: s.underruns })" tone="warning" />
    <Badge v-if="s && s.overruns" :value="t('live.overruns', { n: s.overruns })" tone="warning" />
    <span v-if="s?.vram_mb" class="type-label-medium muted">{{ t('live.vram', { mb: Math.round(s.vram_mb) }) }}</span>
    <AppButton v-if="loadHigh && s" variant="text" @click="emit('increaseBlock', suggested)">{{ t('live.increaseBlock', { ms: suggested }) }}</AppButton>
  </div>
</template>

<style scoped>
.stats { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-3) var(--app-space-4); min-width: 0; }
.meters { display: flex; flex-direction: column; gap: var(--app-space-1); flex: 1 1 220px; min-width: 0; }
.load { display: flex; align-items: center; gap: var(--app-space-2); flex: 1 1 160px; min-width: 0; }
.bar { flex: 1; min-width: 0; }
.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
</style>
