<script setup lang="ts">
import { computed, reactive } from 'vue';
import { type InboxIndex, useInboxIndexes, useInboxMutations } from '@/api/queries/imports';
import { useModels } from '@/api/queries/models';
import { useI18n } from '@/i18n';
import { formatBytes } from '@/format';
import { AppButton, IconButton, SelectField, Surface, icons, useSnackbar } from '@/ui';

/** Indexes imported without a voice: assign each to a compatible voice in the library, or delete it. */
const emit = defineEmits<{ error: [unknown] }>();
const { t } = useI18n();
const snackbar = useSnackbar();
const inbox = useInboxIndexes();
const library = useModels();
const m = useInboxMutations();
const chosen = reactive<Record<string, { voice: string; key: string }>>({});

const items = computed(() => inbox.data.value ?? []);
function choice(i: InboxIndex) {
  return (chosen[i.id] ??= { voice: i.proposed ?? '', key: 'default' });
}
const voiceOptions = (i: InboxIndex) => i.candidates.map((c) => ({ value: c.target, label: c.name }));
function speakers(target: string) {
  return (library.data.value ?? []).find((v) => `voice:${v.id}` === target)?.speakers ?? [];
}
const keyOptions = (target: string) => [{ value: 'default', label: t('models.review.allSpeakers') }, ...speakers(target).map((s) => ({ value: `spk${s.id}`, label: `${s.id} · ${s.name}` }))];
const describe = (i: InboxIndex) => [i.version ?? t('models.inbox.dim', { n: i.dim }), t('models.review.vectors', { n: i.vectors.toLocaleString() }), formatBytes(i.size)].join(' · ');

function assign(i: InboxIndex) {
  const c = choice(i);
  if (!c.voice) return;
  m.assign.mutate(
    { id: i.id, voiceId: c.voice.replace(/^voice:/, ''), key: c.key },
    { onSuccess: (v) => snackbar.show(t('models.inbox.assigned', { name: i.name, voice: v.name })), onError: (e) => emit('error', e) },
  );
}
</script>

<template>
  <Surface v-if="items.length" :level="1" class="inbox">
    <div class="head">
      <h2 class="type-title-small title">{{ t('models.inbox.title') }}</h2>
      <span class="type-body-small muted">{{ t('models.inbox.hint') }}</span>
    </div>
    <div v-for="i in items" :key="i.id" class="row">
      <div class="text">
        <span class="type-body-medium name">{{ i.name }}</span>
        <span class="type-body-small muted">{{ describe(i) }}</span>
      </div>
      <template v-if="i.candidates.length">
        <SelectField v-model="choice(i).voice" :label="t('models.review.voice')" :options="voiceOptions(i)" />
        <SelectField v-if="speakers(choice(i).voice).length" v-model="choice(i).key" :label="t('models.review.speaker')" :options="keyOptions(choice(i).voice)" />
        <AppButton variant="tonal" :disabled="!choice(i).voice" :loading="m.assign.isPending.value" @click="assign(i)">{{ t('models.inbox.assign') }}</AppButton>
      </template>
      <span v-else class="type-body-small muted">{{ t('models.inbox.noVoice', { version: i.version ?? '?' }) }}</span>
      <IconButton :icon="icons.Trash2" :label="t('common.delete')" @click="m.remove.mutate(i.id, { onError: (e) => emit('error', e) })" />
    </div>
  </Surface>
</template>

<style scoped>
.inbox { display: flex; flex-direction: column; gap: var(--app-space-2); padding: var(--app-space-3) var(--app-space-4); }
.head { display: flex; flex-direction: column; }
.title { margin: 0; }
.row { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-3); min-width: 0; }
.text { display: flex; flex-direction: column; flex: 1 1 220px; min-width: 0; }
.name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.muted { color: var(--md-sys-color-on-surface-variant); }
</style>
