<script setup lang="ts">
import { computed } from 'vue';
import type { DatasetReport } from '@/api/types';
import { useI18n } from '@/i18n';
import { formatDuration } from '@/format';
import { Badge, DataList } from '@/ui';

/** What "Scan dataset" found: clip count, duration, sample rates, and problem clips. */
const props = defineProps<{ report: DatasetReport }>();
const { t, tOr } = useI18n();
const rows = computed(() => [
  { label: t('train.report.clips', { n: props.report.clips }), value: t('train.report.duration', { d: formatDuration(props.report.total_seconds) }) },
  { label: t('train.report.rates'), value: Object.entries(props.report.sample_rates).map(([r, n]) => `${Number(r) / 1000} kHz × ${n}`).join(', ') || '—' },
  ...Object.entries(props.report.speakers ?? {}).map(([name, n]) => ({ label: name, value: t('train.report.clips', { n }) })),
]);
</script>

<template>
  <section class="report">
    <DataList :rows="rows" />
    <h4 class="type-title-small title">{{ t('train.report.issues') }}</h4>
    <p v-if="!report.issues.length" class="type-body-medium muted">{{ t('train.report.noIssues') }}</p>
    <ul v-else class="issues">
      <li v-for="i in report.issues.slice(0, 200)" :key="i.path + i.problem" class="issue">
        <Badge :value="tOr(`train.report.problem.${i.problem}`, i.problem)" tone="warning" />
        <span class="type-body-small path">{{ i.path }}</span>
        <span v-if="i.detail" class="type-body-small muted">{{ i.detail }}</span>
      </li>
    </ul>
  </section>
</template>

<style scoped>
.report { display: flex; flex-direction: column; gap: var(--app-space-2); min-width: 0; }
.title { margin: var(--app-space-2) 0 0; }
.issues { list-style: none; margin: 0; padding: 0; max-height: 240px; overflow: auto; display: flex; flex-direction: column; gap: var(--app-space-1); }
.issue { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.path { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
