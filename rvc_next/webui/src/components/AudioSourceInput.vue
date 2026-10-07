<script setup lang="ts">
import { computed, ref } from 'vue';
import { ApiError } from '@/api/client';
import { useMeta } from '@/api/queries/app';
import { resolveServerPath, synthesizeSpeech, uploadAudio } from '@/api/queries/audio';
import { urls } from '@/api/client';
import type { Output, S } from '@/api/types';
import OutputsPickerDialog from '@/components/OutputsPickerDialog.vue';
import ServerPathDialog from '@/components/ServerPathDialog.vue';
import TtsDialog from '@/components/TtsDialog.vue';
import { inputKey, type InputItem } from '@/components/inputs';
import { useI18n } from '@/i18n';
import { formatDuration } from '@/format';
import { usePlayerStore } from '@/stores/player';
import { AppButton, AppIcon, DropZone, IconButton, ProgressBar, TRANSITIONS, icons, staggerStyle } from '@/ui';

/**
 * The one way to give audio: drop files, pick server files (on the server's own
 * machine, or with an access token), choose earlier results, or, with ``tts``, speak text. ``folders``
 * also allows a server folder as one input (Convert expands it; a dataset is a folder).
 */
const props = withDefaults(defineProps<{ multiple?: boolean; folders?: boolean; uploads?: boolean; outputs?: boolean; tts?: boolean; label?: string }>(), {
  multiple: true,
  folders: false,
  uploads: true,
  outputs: true,
  tts: false,
  label: '',
});
const items = defineModel<InputItem[]>({ default: () => [] });
const { t } = useI18n();
const meta = useMeta();
const player = usePlayerStore();
const serverOpen = ref(false);
const outputsOpen = ref(false);
const ttsOpen = ref(false);
const canBrowse = computed(() => meta.data.value?.trusted ?? false);

function add(item: InputItem) {
  items.value = props.multiple ? [...items.value, item] : [item];
}
function patch(key: string, p: Partial<InputItem>) {
  items.value = items.value.map((i) => (i.key === key ? { ...i, ...p } : i));
}

async function onFiles(files: File[]) {
  for (const file of props.multiple ? files : files.slice(0, 1)) {
    const key = inputKey();
    add({ key, name: file.name, ref: null, status: 'uploading', progress: 0 });
    try {
      const f = await uploadAudio(file, (loaded, total) => patch(key, { progress: total ? loaded / total : 0 }));
      patch(key, { status: 'ready', ref: { kind: 'upload', id: f.id, path: null }, duration: f.duration, fileId: f.id });
    } catch (e) {
      patch(key, { status: 'failed', error: e instanceof ApiError ? e.message : String(e) });
    }
  }
}

async function onServer(selected: { path: string; name: string; dir: boolean }[]) {
  for (const s of selected) {
    const key = inputKey();
    if (s.dir) {
      add({ key, name: s.name, ref: { kind: 'path', path: s.path, id: null }, status: 'ready', folder: true });
      continue;
    }
    add({ key, name: s.name, ref: null, status: 'uploading' });
    try {
      // Registering the file probes it (duration) and gives it an id for playing and peaks.
      const f = await resolveServerPath(s.path);
      patch(key, { status: 'ready', ref: { kind: 'upload', id: f.id, path: null }, duration: f.duration, fileId: f.id });
    } catch (e) {
      patch(key, { status: 'failed', error: e instanceof ApiError ? e.message : String(e) });
    }
  }
}

async function onSpeech(request: S['TtsRequest']) {
  const key = inputKey();
  add({ key, name: request.text.length > 40 ? `${request.text.slice(0, 40)}…` : request.text, ref: null, status: 'uploading' });
  try {
    const f = await synthesizeSpeech(request);
    patch(key, { status: 'ready', name: f.name, ref: { kind: 'upload', id: f.id, path: null }, duration: f.duration, fileId: f.id });
  } catch (e) {
    patch(key, { status: 'failed', error: e instanceof ApiError ? e.message : String(e) });
  }
}

function onOutputs(outs: Output[]) {
  for (const o of outs) add({ key: inputKey(), name: o.name, ref: { kind: 'output', id: o.id, path: null }, status: 'ready', duration: o.duration, outputId: o.id });
}

const remove = (key: string) => (items.value = items.value.filter((i) => i.key !== key));
const playUrl = (i: InputItem) => (i.fileId ? urls.audioFile(i.fileId) : i.outputId ? urls.output(i.outputId) : null);
function play(i: InputItem) {
  const url = playUrl(i);
  if (url) player.play({ key: i.key, url, label: i.name });
}
</script>

<template>
  <section class="source" :aria-label="label || t('source.title')">
    <DropZone v-if="uploads" :label="t('source.drop')" :hint="t('source.dropHint')" accept="audio/*,video/*,.wav,.flac,.mp3,.m4a,.ogg,.opus,.aac,.wma" :multiple="multiple" @files="onFiles" />
    <div class="buttons">
      <AppButton v-if="canBrowse" variant="tonal" :icon="icons.HardDrive" @click="serverOpen = true">{{ t('source.serverFiles') }}</AppButton>
      <AppButton v-if="outputs" variant="tonal" :icon="icons.History" @click="outputsOpen = true">{{ t('source.outputs') }}</AppButton>
      <AppButton v-if="tts" variant="tonal" :icon="icons.Speech" @click="ttsOpen = true">{{ t('source.tts') }}</AppButton>
      <AppButton v-if="items.length > 1" variant="text" @click="items = []">{{ t('source.clear') }}</AppButton>
    </div>
    <TransitionGroup :name="TRANSITIONS.list" tag="ul" class="items">
      <li v-for="(i, n) in items" :key="i.key" class="item" :class="i.status" :style="staggerStyle(n)">
        <IconButton v-if="playUrl(i)" :icon="player.isPlaying(i.key) ? icons.Pause : icons.Play" :label="player.isPlaying(i.key) ? t('results.pause') : t('results.play')" @click="play(i)" />
        <AppIcon v-else :icon="i.folder ? icons.Folder : icons.Music" :size="20" class="lead" />
        <span class="text">
          <span class="type-body-large name">{{ i.name }}</span>
          <span v-if="i.status === 'failed'" class="type-body-small err">{{ t('source.uploadFailed', { name: i.name }) }}: {{ i.error }}</span>
          <ProgressBar v-else-if="i.status === 'uploading'" :value="i.progress ?? null" :label="t('source.uploading', { name: i.name })" />
        </span>
        <span v-if="i.duration" class="type-body-small muted">{{ formatDuration(i.duration) }}</span>
        <IconButton :icon="icons.X" :label="t('common.remove')" @click="remove(i.key)" />
      </li>
    </TransitionGroup>
    <ServerPathDialog v-model:open="serverOpen" :folders="folders" @select="onServer" />
    <OutputsPickerDialog v-model:open="outputsOpen" @select="onOutputs" />
    <TtsDialog v-if="tts" v-model:open="ttsOpen" @add="onSpeech" />
  </section>
</template>

<style scoped>
.source { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.buttons { display: flex; flex-wrap: wrap; gap: var(--app-space-2); }
.items { position: relative; list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--app-space-1); }
.item { display: flex; align-items: center; gap: var(--app-space-2); padding: var(--app-space-1) var(--app-space-1) var(--app-space-1) var(--app-space-2); border-radius: var(--md-sys-shape-corner-small); background: var(--md-sys-color-surface-container); min-width: 0; }
.lead { margin: 0 var(--app-space-3); }
.text { display: flex; flex-direction: column; gap: var(--app-space-1); flex: 1; min-width: 0; }
.name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.err { color: var(--md-sys-color-error); overflow-wrap: anywhere; }
</style>
