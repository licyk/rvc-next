<script setup lang="ts">
import { computed, ref } from 'vue';
import JobCard from '@/components/JobCard.vue';
import { useI18n } from '@/i18n';
import { useJobsStore } from '@/stores/jobs';
import { AppButton, TextField, icons } from '@/ui';

/**
 * Add models from links beside a drop zone: Hugging Face files, repositories or folders, Google
 * Drive files, or any http(s) file. The server downloads them; the import then goes on as for files.
 */
const props = defineProps<{ fetching: string | null; disabled?: boolean }>();
const emit = defineEmits<{ links: [string] }>();
const { t } = useI18n();
const jobs = useJobsStore();
const text = ref('');
const job = computed(() => (props.fetching ? (jobs.byId[props.fetching] ?? null) : null));

function add() {
  if (!text.value.trim()) return;
  emit('links', text.value);
  text.value = '';
}
</script>

<template>
  <div class="links">
    <div class="row">
      <TextField v-model="text" :label="t('models.links')" :supporting-text="t('models.linksHint')" :disabled="disabled || !!fetching" @enter="add" />
      <AppButton variant="tonal" :icon="icons.Download" :disabled="disabled || !!fetching || !text.trim()" @click="add">{{ t('common.add') }}</AppButton>
    </div>
    <JobCard v-if="job" :job="job" />
  </div>
</template>

<style scoped>
.links { display: flex; flex-direction: column; gap: var(--app-space-2); }
/* The field and its button: the button centred on the field. */
.row { display: flex; align-items: center; gap: var(--app-space-2); }
.row > :first-child { flex: 1; min-width: 0; }
</style>
