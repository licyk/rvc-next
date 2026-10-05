<script setup lang="ts">
import { computed, ref } from 'vue';
import { useRouter } from 'vue-router';
import { useExperimentMutations, useExperiments } from '@/api/queries/train';
import type { ExperimentSummary } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import ServerPathDialog from '@/components/ServerPathDialog.vue';
import { useI18n } from '@/i18n';
import { formatAgo } from '@/format';
import { AppButton, AppCard, AppDialog, Badge, EmptyState, PathField, SegmentedControl, TRANSITIONS, TextField, icons, staggerStyle } from '@/ui';

const { t, tOr, locale } = useI18n();
const router = useRouter();
const list = useExperiments();
const m = useExperimentMutations();

const STAGES = ['slice', 'f0', 'features', 'fit', 'index'];
const tone = (s: string) => ({ done: 'primary', running: 'primary', failed: 'error', stale: 'warning' })[s] ?? 'neutral';

const newOpen = ref(false);
const name = ref('');
const mode = ref<'single' | 'multi'>('single');
const folder = ref('');
const browseOpen = ref(false);
const importOpen = ref(false);
const importPath = ref('');
const importName = ref('');
const error = ref<unknown>(null);
const modes = computed(() => [
  { value: 'single' as const, label: t('train.mode.single') },
  { value: 'multi' as const, label: t('train.mode.multi') },
]);

function create() {
  error.value = null;
  m.create.mutate(
    { name: name.value.trim(), dataset: { mode: mode.value, folder: mode.value === 'single' && folder.value ? folder.value : null, speakers: [], clean_preset: null } },
    {
      onSuccess: (exp) => {
        newOpen.value = false;
        router.push(`/train/${encodeURIComponent(exp.name)}`);
      },
      onError: (e) => (error.value = e),
    },
  );
}
function doImport() {
  error.value = null;
  m.importLegacy.mutate(
    { path: importPath.value.trim(), name: importName.value.trim() || null },
    { onSuccess: (exp) => ((importOpen.value = false), router.push(`/train/${encodeURIComponent(exp.name)}`)), onError: (e) => (error.value = e) },
  );
}
const open = (e: ExperimentSummary) => router.push(`/train/${encodeURIComponent(e.name)}`);
</script>

<template>
  <div class="train">
    <header class="bar">
      <AppButton :icon="icons.Plus" @click="(newOpen = true), (name = ''), (folder = ''), (error = null)">{{ t('train.new') }}</AppButton>
      <AppButton variant="tonal" :icon="icons.FolderInput" @click="(importOpen = true), (error = null)">{{ t('train.import') }}</AppButton>
    </header>
    <ErrorNotice v-if="list.error.value" :error="list.error.value" />
    <EmptyState v-else-if="list.isSuccess.value && !list.data.value?.length" :icon="icons.GraduationCap" :title="t('train.empty')" />
    <TransitionGroup v-else :name="TRANSITIONS.list" tag="div" class="grid">
      <AppCard v-for="(e, i) in list.data.value ?? []" :key="e.name" interactive variant="outlined" class="card" :style="staggerStyle(i)" @activate="open(e)">
        <div class="card-head">
          <span class="type-title-medium name">{{ e.name }}</span>
          <Badge v-if="e.running_job" :value="tOr('train.stageStates.running', 'running')" tone="primary" />
        </div>
        <span class="type-body-small muted">{{ e.sample_rate }} · {{ e.version }} · {{ t(`train.mode.${e.mode}`) }}</span>
        <div class="stages">
          <Badge v-for="s in STAGES" :key="s" :value="tOr(`train.stages.${s}`, s)" :tone="tone(e.stages[s]?.status ?? 'pending') as 'primary'" />
        </div>
        <span class="type-body-small muted">{{ t('train.lastActivity', { when: formatAgo(e.last_activity, locale) }) }}</span>
      </AppCard>
    </TransitionGroup>

    <AppDialog v-model:open="newOpen" :title="t('train.new')" width="small" :close-label="t('common.close')">
      <TextField v-model="name" :label="t('train.name')" />
      <SegmentedControl v-model="mode" :options="modes" />
      <div v-if="mode === 'single'" class="folder">
        <PathField v-model="folder" :label="t('train.folder')" />
        <AppButton variant="text" @click="browseOpen = true">{{ t('common.browse') }}</AppButton>
      </div>
      <ErrorNotice v-if="error" :error="error" />
      <template #actions>
        <AppButton variant="text" @click="newOpen = false">{{ t('common.cancel') }}</AppButton>
        <AppButton :disabled="!name.trim()" :loading="m.create.isPending.value" @click="create">{{ t('common.add') }}</AppButton>
      </template>
    </AppDialog>
    <AppDialog v-model:open="importOpen" :title="t('train.import')" width="small" :close-label="t('common.close')">
      <PathField v-model="importPath" :label="t('train.importPath')" />
      <TextField v-model="importName" :label="t('train.name')" />
      <ErrorNotice v-if="error" :error="error" />
      <template #actions>
        <AppButton variant="text" @click="importOpen = false">{{ t('common.cancel') }}</AppButton>
        <AppButton :disabled="!importPath.trim()" :loading="m.importLegacy.isPending.value" @click="doImport">{{ t('train.import') }}</AppButton>
      </template>
    </AppDialog>
    <ServerPathDialog v-model:open="browseOpen" folders :files="false" @select="(s) => s[0] && (folder = s[0].path)" />
  </div>
</template>

<style scoped>
.train { display: flex; flex-direction: column; gap: var(--app-space-4); padding: var(--app-space-4); }
.bar { display: flex; flex-wrap: wrap; gap: var(--app-space-2); }
.grid { position: relative; display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: var(--app-space-3); }
.card { display: flex; flex-direction: column; gap: var(--app-space-2); padding: var(--app-space-4); min-width: 0; }
.card-head { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.stages { display: flex; flex-wrap: wrap; gap: var(--app-space-1); }
/* The folder field and Browse: the button centred on the field. */
.folder { display: flex; align-items: center; gap: var(--app-space-2); }
.folder > :first-child { flex: 1; min-width: 0; }
</style>
