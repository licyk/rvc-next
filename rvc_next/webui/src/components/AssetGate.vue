<script setup lang="ts">
import { computed, watch } from 'vue';
import { useAssets, useDownloadAssets } from '@/api/queries/assets';
import { useI18n } from '@/i18n';
import { formatBytes } from '@/format';
import { useJobsStore } from '@/stores/jobs';
import { AppButton, ProgressBar, icons } from '@/ui';

/**
 * Missing assets are offered, not reported: while any of ``assets`` is not
 * installed this replaces its slot (the action needing them) with "Needs RMVPE — Download (172 MB)";
 * once installed it shows the slot again and emits ``ready``, so the caller can run the action.
 */
const props = defineProps<{ assets: string[] }>();
const emit = defineEmits<{ ready: [] }>();
const { t } = useI18n();
const all = useAssets();
const download = useDownloadAssets();
const jobs = useJobsStore();

const wanted = computed(() => (all.data.value ?? []).filter((a) => props.assets.includes(a.id)));
const missing = computed(() => wanted.value.filter((a) => a.state !== 'installed' && a.state !== 'corrupt'));
const known = computed(() => !!all.data.value);
const size = computed(() => missing.value.reduce((n, a) => n + (a.size - a.installed_bytes), 0));
const job = computed(() => {
  const id = missing.value.find((a) => a.job_id)?.job_id;
  return id ? (jobs.byId[id] ?? null) : null;
});
const downloading = computed(() => missing.value.some((a) => a.state === 'downloading') || download.isPending.value);

watch(
  () => missing.value.length,
  (n, before) => {
    if (known.value && n === 0 && (before ?? 0) > 0) emit('ready');
  },
);
</script>

<template>
  <slot v-if="!known || !missing.length" />
  <div v-else class="gate">
    <span class="type-body-medium">{{ t('assets.needs', { names: missing.map((a) => a.title).join(', ') }) }}</span>
    <ProgressBar v-if="downloading" :value="job?.progress ?? null" :label="t('assets.downloading')" />
    <AppButton v-else variant="tonal" :icon="icons.Download" @click="download.mutate({ ids: missing.map((a) => a.id) })">{{ t('assets.download', { size: formatBytes(size) }) }}</AppButton>
  </div>
</template>

<style scoped>
.gate { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-3); padding: var(--app-space-3) var(--app-space-4); border-radius: var(--md-sys-shape-corner-medium); background: var(--md-sys-color-tertiary-container); color: var(--md-sys-color-on-tertiary-container); }
.gate > :deep(.progress), .gate > * { min-width: 0; }
</style>
