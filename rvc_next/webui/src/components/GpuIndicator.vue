<script setup lang="ts">
import { computed } from 'vue';
import { useComputeUsage, useReleaseCompute } from '@/api/queries/compute';
import { errorView } from '@/components/errors';
import { useI18n } from '@/i18n';
import { formatMiB } from '@/format';
import { AppMenu, IconButton, icons, useSnackbar, type MenuItem } from '@/ui';

/** GPU memory in the top bar; its menu holds Free GPU memory (the original's 卸载音色省显存). */
const { t, tOr } = useI18n();
const usage = useComputeUsage();
const release = useReleaseCompute();
const snackbar = useSnackbar();
const label = computed(() => {
  const u = usage.data.value;
  if (!u) return t('gpu.title');
  const mem = u.vram_total_mb ? t('gpu.used', { used: formatMiB(u.vram_used_mb), total: formatMiB(u.vram_total_mb) }) : u.device;
  return `${t('gpu.title')}: ${mem} · ${u.cached.length ? t('gpu.cached', { list: u.cached.join(', ') }) : t('gpu.none')}`;
});
// Available whenever nothing is working: models may sit in the live worker or the allocator's
// cache even when the server's own list is empty. The server refuses while a job or Live runs.
const busy = computed(() => usage.data.value?.busy ?? []);
const items = computed<MenuItem[]>(() => [
  {
    id: 'free',
    label: busy.value.length ? t('gpu.freeBusy', { reason: busy.value.map((b) => tOr(`gpu.busy.${b}`, b)).join(', ') }) : t('gpu.free'),
    icon: icons.MemoryStick,
    disabled: !(usage.data.value?.can_release ?? false) || release.isPending.value,
  },
]);
function select(id: string) {
  if (id === 'free') release.mutate(undefined, { onSuccess: () => snackbar.show(t('gpu.freed')), onError: (e) => snackbar.show(errorView(e, t, tOr).message) });
}
</script>

<template>
  <AppMenu :items="items" @select="select">
    <template #default="{ toggle }">
      <IconButton :icon="icons.MemoryStick" :label="label" :badge="usage.data.value?.cached.length || null" @click="toggle" />
    </template>
  </AppMenu>
</template>
