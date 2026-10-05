<script setup lang="ts">
import { computed, ref } from 'vue';
import { errorView } from '@/components/errors';
import { useI18n } from '@/i18n';
import { AppIcon, icons } from '@/ui';

/** A domain error in place: its translated title, the server's message, and the detail on request. */
const props = defineProps<{ error: unknown }>();
const { t, tOr } = useI18n();
const view = computed(() => errorView(props.error, t, tOr));
const open = ref(false);
const hasDetail = computed(() => Object.keys(view.value.detail).length > 0);
</script>

<template>
  <div class="error-notice" role="alert">
    <AppIcon :icon="icons.AlertTriangle" :size="20" />
    <div class="body">
      <span class="type-title-small title">{{ view.title }}</span>
      <span v-if="view.message && view.message !== view.title" class="type-body-medium message">{{ view.message }}</span>
      <button v-if="hasDetail" type="button" class="toggle type-label-large" @click="open = !open">{{ t('common.details') }}</button>
      <pre v-if="open" class="type-body-small detail">{{ JSON.stringify(view.detail, null, 2) }}</pre>
    </div>
  </div>
</template>

<style scoped>
.error-notice { display: flex; gap: var(--app-space-3); padding: var(--app-space-3) var(--app-space-4); border-radius: var(--md-sys-shape-corner-medium); background: var(--md-sys-color-error-container); color: var(--md-sys-color-on-error-container); min-width: 0; }
.body { display: flex; flex-direction: column; gap: var(--app-space-1); min-width: 0; flex: 1; }
.message { overflow-wrap: anywhere; }
.toggle { align-self: flex-start; padding: 0; border: 0; background: none; color: inherit; text-decoration: underline; cursor: pointer; }
.detail { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; max-height: 200px; overflow: auto; }
</style>
