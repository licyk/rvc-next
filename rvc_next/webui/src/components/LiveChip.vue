<script setup lang="ts">
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useLiveControl } from '@/api/queries/live';
import { useI18n } from '@/i18n';
import { useLiveStore } from '@/stores/live';
import { AppIcon, IconButton, icons } from '@/ui';

/** Shown on every screen while a live session runs: state, latency and a stop button. */
const live = useLiveStore();
const control = useLiveControl();
const router = useRouter();
const { t, tOr } = useI18n();
const text = computed(() => (live.running && live.stats ? t('live.chip', { ms: Math.round(live.state.latency_test?.ok && live.state.latency_test.latency_ms !== null ? live.state.latency_test.latency_ms : live.stats.est_latency_ms) }) : tOr(live.statusKey, live.state.state)));
</script>

<template>
  <Transition name="scrim">
    <div v-if="live.active" class="live-chip" :class="live.state.state">
      <button type="button" class="open state-layer" @click="router.push('/live')">
        <AppIcon :icon="icons.Mic" :size="18" :spin="false" />
        <span class="type-label-large">{{ text }}</span>
      </button>
      <IconButton :icon="icons.Square" :label="t('live.stop')" @click="control.stop.mutate()" />
    </div>
  </Transition>
</template>

<style scoped>
.live-chip { display: flex; align-items: center; gap: 0; padding-left: var(--app-space-1); border-radius: var(--md-sys-shape-corner-full); background: var(--md-sys-color-primary-container); color: var(--md-sys-color-on-primary-container); }
.live-chip.reconnecting, .live-chip.error { background: var(--md-sys-color-error-container); color: var(--md-sys-color-on-error-container); }
.open { display: flex; align-items: center; gap: var(--app-space-2); height: 40px; padding: 0 var(--app-space-2); border: 0; background: transparent; color: inherit; border-radius: var(--md-sys-shape-corner-full); cursor: pointer; white-space: nowrap; }
</style>
