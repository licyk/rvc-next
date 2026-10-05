<script setup lang="ts">
import { computed, ref } from 'vue';
import type { SpeakerEntry } from '@/api/types';
import ServerPathDialog from '@/components/ServerPathDialog.vue';
import { useI18n } from '@/i18n';
import { AppButton, IconButton, TextField, icons } from '@/ui';

/**
 * The speaker table of a multi-speaker dataset (replacing the paged 110-slot helper): name, ID
 * 0–109, folder and repeat per row. ``fromFolders`` asks to fill it from Name_ID_Repeat folders.
 */
const rows = defineModel<SpeakerEntry[]>({ default: () => [] });
const emit = defineEmits<{ fromFolders: [string] }>();
const { t } = useI18n();
const pickFor = ref<number | null>(null);
const pickOpen = ref(false);
const rootOpen = ref(false);

const duplicates = computed(() => {
  const seen = new Map<number, number>();
  for (const r of rows.value) seen.set(r.id, (seen.get(r.id) ?? 0) + 1);
  return new Set([...seen.entries()].filter(([, n]) => n > 1).map(([id]) => id));
});
function patch(i: number, p: Partial<SpeakerEntry>) {
  rows.value = rows.value.map((r, n) => (n === i ? { ...r, ...p } : r));
}
function add() {
  const used = new Set(rows.value.map((r) => r.id));
  let id = 0;
  while (used.has(id) && id < 109) id++;
  rows.value = [...rows.value, { name: `speaker${id}`, id, folder: '', repeat: 1 }];
}
const removeRow = (i: number) => (rows.value = rows.value.filter((_, n) => n !== i));
function browse(i: number) {
  pickFor.value = i;
  pickOpen.value = true;
}
function picked(sel: { path: string }[]) {
  if (pickFor.value !== null && sel[0]) patch(pickFor.value, { folder: sel[0].path });
}
</script>

<template>
  <section class="speakers">
    <h3 class="type-title-small title">{{ t('train.speakers.title') }}</h3>
    <div class="table" role="table">
      <div class="row head type-label-medium muted" role="row">
        <span role="columnheader">{{ t('train.speakers.name') }}</span><span role="columnheader">{{ t('train.speakers.id') }}</span><span role="columnheader">{{ t('train.speakers.folder') }}</span><span role="columnheader">{{ t('train.speakers.repeat') }}</span><span />
      </div>
      <div v-for="(r, i) in rows" :key="i" class="row" role="row">
        <TextField :model-value="r.name" :label="t('train.speakers.name')" @update:model-value="patch(i, { name: String($event ?? '') })" />
        <TextField type="number" :min="0" :max="109" :model-value="r.id" :label="t('train.speakers.id')" :error-text="duplicates.has(r.id) ? t('train.speakers.duplicate', { id: r.id }) : ''" @update:model-value="patch(i, { id: Number($event) })" />
        <div class="folder">
          <span class="type-body-medium path">{{ r.folder || '—' }}</span>
          <IconButton :icon="icons.FolderOpen" :label="t('common.browse')" @click="browse(i)" />
        </div>
        <TextField type="number" :min="1" :max="100" :model-value="r.repeat" :label="t('train.speakers.repeat')" @update:model-value="patch(i, { repeat: Number($event) })" />
        <IconButton :icon="icons.Trash2" :label="t('common.remove')" @click="removeRow(i)" />
      </div>
    </div>
    <div class="actions">
      <AppButton variant="tonal" :icon="icons.Plus" @click="add">{{ t('train.speakers.add') }}</AppButton>
      <AppButton variant="text" :icon="icons.FolderInput" @click="rootOpen = true">{{ t('train.speakers.fromFolders') }}</AppButton>
    </div>
    <ServerPathDialog v-model:open="pickOpen" folders :files="false" @select="picked" />
    <ServerPathDialog v-model:open="rootOpen" folders :files="false" @select="(s) => s[0] && emit('fromFolders', s[0].path)" />
  </section>
</template>

<style scoped>
.speakers { display: flex; flex-direction: column; gap: var(--app-space-2); min-width: 0; }
.title { margin: 0; }
.table { display: flex; flex-direction: column; gap: var(--app-space-2); overflow-x: auto; }
.row { display: grid; grid-template-columns: minmax(120px, 1.2fr) 88px minmax(160px, 2fr) 88px 48px; gap: var(--app-space-2); align-items: center; min-width: 560px; }
.folder { display: flex; align-items: center; gap: var(--app-space-1); min-width: 0; }
.path { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; direction: rtl; text-align: left; }
.actions { display: flex; flex-wrap: wrap; gap: var(--app-space-2); }
</style>
