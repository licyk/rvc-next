<script setup lang="ts">
import { useMeta } from '@/api/queries/app';
import { useI18n } from '@/i18n';
import { Skeleton } from '@/ui';

/** DevicePanel's shape while the saved selection (the settings) is still on its way. */
const { t } = useI18n();
const meta = useMeta();
</script>

<template>
  <section class="devices" role="status" aria-busy="true">
    <h2 class="type-title-medium title">{{ meta.data.value?.host ? t('devices.title', { host: meta.data.value.host }) : t('devices.titleUnknown') }}</h2>
    <p class="type-body-small muted">{{ t('devices.loading') }}</p>
    <div class="roles">
      <Skeleton v-for="role in 3" :key="role" height="56px" shape="medium" />
    </div>
  </section>
</template>

<style scoped>
.devices { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.title { margin: 0; min-width: 0; overflow-wrap: anywhere; }
.roles { display: flex; flex-direction: column; gap: var(--app-space-5); }
.muted { margin: 0; }
</style>
