<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useRouter } from 'vue-router';
import { downloadUrl, urls } from '@/api/client';
import { useMeta } from '@/api/queries/app';
import { useConvert } from '@/api/queries/convert';
import { revealModel, useAttachIndexFile, useDeleteModel, useModel, useModelJobs, useSetIndexPath, useUpdateModel } from '@/api/queries/models';
import { usePresetMutations, usePresets } from '@/api/queries/presets';
import { useExperiments } from '@/api/queries/train';
import type { VoiceModel, VoiceParams } from '@/api/types';
import AssetGate from '@/components/AssetGate.vue';
import AudioSourceInput from '@/components/AudioSourceInput.vue';
import ErrorNotice from '@/components/ErrorNotice.vue';
import ResultsList from '@/components/ResultsList.vue';
import ServerPathDialog from '@/components/ServerPathDialog.vue';
import VoiceParamsPanel from '@/components/VoiceParamsPanel.vue';
import { readyRefs, type InputItem } from '@/components/inputs';
import { pitchAssets } from '@/components/paramFields';
import { useSessionOutputs } from '@/components/sessionOutputs';
import { useI18n } from '@/i18n';
import { useJobsStore } from '@/stores/jobs';
import { formatBytes, formatRate } from '@/format';
import { AppButton, ConfirmDialog, DataList, DropZone, IconButton, PathField, SelectField, Surface, TextField, containerFrom, icons, useSnackbar } from '@/ui';

const props = defineProps<{ id: string }>();
const { t } = useI18n();
const router = useRouter();
const snackbar = useSnackbar();
const meta = useMeta();
const voiceQ = useModel(() => props.id);
const voice = computed(() => voiceQ.data.value ?? null);
const update = useUpdateModel();
const remove = useDeleteModel();
const attachFile = useAttachIndexFile();
const setIndexPath = useSetIndexPath();
const { buildIndex } = useModelJobs();
const experiments = useExperiments();
const presets = usePresets(() => props.id);
const presetMut = usePresetMutations();
const convert = useConvert();
const error = ref<unknown>(null);
const onErr = (e: unknown) => (error.value = e);

// Container transform from the voice card: the detail grows from the card's rectangle
// into the content area; without one it fades and scales.
function fromCard(): Record<string, string> {
  try {
    const from = JSON.parse(sessionStorage.getItem('rvc-next:fromRect') ?? 'null');
    sessionStorage.removeItem('rvc-next:fromRect');
    const to = document.querySelector('main.content')?.getBoundingClientRect();
    return from && to ? containerFrom(from as DOMRect, to) : {};
  } catch {
    return {};
  }
}
const motion = fromCard();

const name = ref('');
const description = ref('');
const tags = ref('');
const speakerNames = ref<Record<number, string>>({});
watch(
  voice,
  (v) => {
    if (!v) return;
    name.value = v.name;
    description.value = v.description;
    tags.value = v.tags.join(', ');
    speakerNames.value = Object.fromEntries(v.speakers.map((s) => [s.id, s.name]));
  },
  { immediate: true },
);
function save() {
  update.mutate(
    {
      id: props.id,
      patch: {
        name: name.value.trim(),
        description: description.value,
        tags: tags.value.split(',').map((x) => x.trim()).filter(Boolean),
        speakers: Object.entries(speakerNames.value).filter(([, n]) => n.trim()).map(([id, n]) => ({ id: Number(id), name: n.trim() })),
      },
    },
    { onSuccess: () => snackbar.show(t('models.detail.saved')), onError: onErr },
  );
}

/** What the file says about its training; rows only for what it carries. */
function provenanceRows(p: VoiceModel['provenance']): { label: string; value: string }[] {
  const rows: { label: string; value: string }[] = [];
  if (p.author) rows.push({ label: t('models.detail.author'), value: p.author });
  if (p.epoch != null) rows.push({ label: t('models.detail.trainedFor'), value: p.step != null ? t('models.detail.epochSteps', { epoch: p.epoch, step: p.step }) : t('models.detail.epochs', { epoch: p.epoch }) });
  if (p.dataset_length) rows.push({ label: t('models.detail.datasetLength'), value: p.dataset_length });
  if (p.created) rows.push({ label: t('models.detail.created'), value: p.created.replace('T', ' ').slice(0, 19) });
  return rows;
}

const info = computed(() =>
  voice.value
    ? [
        { label: t('models.detail.sampleRate'), value: formatRate(voice.value.sample_rate) },
        { label: t('models.detail.version'), value: voice.value.version },
        { label: t('models.detail.pitchGuidance'), value: voice.value.pitch_guidance ? t('common.yes') : t('common.no') },
        { label: t('models.detail.speakers'), value: voice.value.speakers.length ? voice.value.speakers.map((s) => `${s.id}: ${s.name}`).join(', ') : `${t('voice.singleSpeaker')} · ${t('models.detail.slots', { n: voice.value.speaker_slots })}` },
        { label: t('models.detail.file'), value: voice.value.model_path, mono: true },
        { label: t('models.detail.size'), value: formatBytes(voice.value.size) },
        { label: t('models.detail.info2'), value: voice.value.info || '—' },
        ...provenanceRows(voice.value.provenance),
      ]
    : [],
);

// Index
const indexKey = ref<string | null>('default');
const keyOptions = computed(() => [{ value: 'default', label: 'default' }, ...(voice.value?.speakers ?? []).map((s) => ({ value: `spk${s.id}`, label: `${s.name} (spk${s.id})` }))]);
const indexPath = ref('');
const browseIndex = ref(false);
const buildFrom = ref<string | null>(null);
const expOptions = computed(() => (experiments.data.value ?? []).map((e) => ({ value: e.name, label: e.name })));
function onIndexFiles(files: File[]) {
  const f = files[0];
  if (f) attachFile.mutate({ id: props.id, key: indexKey.value ?? 'default', file: f }, { onError: onErr });
}

// Default preset, edited in place.
const defaultPreset = computed(() => (presets.data.value ?? []).find((p) => p.voice_id === props.id && p.is_default) ?? null);
const params = ref<VoiceParams | null>(null);
watch(defaultPreset, (p) => p && !params.value && (params.value = { ...p.params }), { immediate: true });
function saveDefault() {
  if (defaultPreset.value && params.value) presetMut.update.mutate({ id: defaultPreset.value.id, patch: { params: params.value } }, { onSuccess: () => snackbar.show(t('models.detail.saved')), onError: onErr });
}

// Test strip
const testInputs = ref<InputItem[]>([]);
const testOutputs = useSessionOutputs(`model:${props.id}`);
const required = computed(() => ['hubert', ...pitchAssets(voice.value?.pitch_guidance, params.value?.f0_method)]);
function test() {
  const refs = readyRefs(testInputs.value);
  if (!refs.length || !params.value) return;
  convert.mutate({ inputs: refs, voice_id: props.id, params: params.value, preview_seconds: 15 }, { onError: onErr });
}
// Jobs of this screen are attributed to it, so the test results show here.
const jobs = useJobsStore();
watch(
  () => convert.data.value,
  (job) => job && jobs.upsert(job, `model:${props.id}`),
);

const deleteOpen = ref(false);
function doDelete() {
  remove.mutate(props.id, { onSuccess: () => router.push('/models'), onError: onErr });
}
</script>

<template>
  <Transition name="container" appear>
  <div class="detail" :style="motion">
    <ErrorNotice v-if="voiceQ.error.value" :error="voiceQ.error.value" />
    <template v-else-if="voice">
      <header class="head">
        <IconButton :icon="icons.ArrowLeft" :label="t('common.back')" @click="router.push('/models')" />
        <h2 class="type-headline-small title">{{ voice.name }}</h2>
        <IconButton :icon="icons.Download" :label="t('models.detail.download')" @click="downloadUrl(urls.modelArchive(id))" />
        <IconButton v-if="meta.data.value?.local" :icon="icons.FolderOpen" :label="t('common.reveal')" @click="revealModel(id).catch(onErr)" />
        <IconButton :icon="icons.Trash2" :label="voice.legacy ? t('models.detail.hide') : t('common.delete')" @click="deleteOpen = true" />
      </header>
      <ErrorNotice v-if="error" :error="error" />
      <div class="sections">
        <Surface :level="0" class="panel">
          <h3 class="type-title-medium sub">{{ t('models.detail.info') }}</h3>
          <div class="fields">
            <TextField v-model="name" :label="t('common.name')" />
            <TextField v-model="description" :label="t('common.description')" />
            <TextField v-model="tags" :label="t('models.detail.tags')" />
          </div>
          <div v-if="voice.speakers.length" class="speakers">
            <TextField v-for="s in voice.speakers" :key="s.id" :model-value="speakerNames[s.id] ?? ''" :label="`${t('voice.speaker')} ${s.id}`" @update:model-value="speakerNames[s.id] = String($event ?? '')" />
          </div>
          <div class="actions"><AppButton :icon="icons.Save" :loading="update.isPending.value" @click="save">{{ t('common.save') }}</AppButton></div>
          <DataList :rows="info" />
        </Surface>
        <Surface :level="0" class="panel">
          <h3 class="type-title-medium sub">{{ t('models.detail.index') }}</h3>
          <ul class="indexes">
            <li v-for="(p, k) in voice.indexes" :key="k" class="index">
              <span class="type-label-large">{{ k }}</span>
              <span class="type-body-small path">{{ p }}</span>
              <AppButton variant="text" @click="setIndexPath.mutate({ id, key: String(k), path: null })">{{ t('models.detail.detach') }}</AppButton>
            </li>
            <li v-if="!Object.keys(voice.indexes).length" class="type-body-medium muted">{{ t('models.detail.indexNone') }}</li>
          </ul>
          <SelectField v-if="keyOptions.length > 1" v-model="indexKey" :label="t('models.detail.index')" :options="keyOptions" />
          <DropZone :label="t('models.detail.attach')" hint=".index" accept=".index" :multiple="false" compact @files="onIndexFiles" />
          <div v-if="meta.data.value?.trusted" class="row">
            <PathField v-model="indexPath" :label="t('models.detail.attachPath')" />
            <AppButton variant="text" :disabled="!indexPath.trim()" @click="setIndexPath.mutate({ id, key: indexKey ?? 'default', path: indexPath.trim() }, { onError: onErr })">{{ t('common.add') }}</AppButton>
          </div>
          <div class="row">
            <SelectField v-model="buildFrom" :label="t('models.detail.build')" :options="expOptions" />
            <AppButton variant="tonal" :icon="icons.Hammer" :disabled="!buildFrom" @click="buildIndex.mutate({ id, experiment: buildFrom! }, { onError: onErr })">{{ t('models.detail.build') }}</AppButton>
          </div>
          <ServerPathDialog v-model:open="browseIndex" @select="(s) => s[0] && (indexPath = s[0].path)" />
        </Surface>
        <Surface v-if="params" :level="0" class="panel">
          <h3 class="type-title-medium sub">{{ t('models.detail.defaultPreset') }}</h3>
          <VoiceParamsPanel v-model="params" :voice="voice" />
          <div class="actions"><AppButton :icon="icons.Save" @click="saveDefault">{{ t('common.save') }}</AppButton></div>
        </Surface>
        <Surface :level="0" class="panel">
          <h3 class="type-title-medium sub">{{ t('models.detail.test') }}</h3>
          <p class="type-body-small muted">{{ t('models.detail.testHint') }}</p>
          <AudioSourceInput v-model="testInputs" :multiple="false" />
          <AssetGate :assets="required">
            <div class="actions"><AppButton :icon="icons.Play" :disabled="!readyRefs(testInputs).length" :loading="convert.isPending.value" @click="test">{{ t('convert.preview') }}</AppButton></div>
          </AssetGate>
          <ResultsList :outputs="testOutputs" />
        </Surface>
      </div>
      <ConfirmDialog
        v-model:open="deleteOpen"
        :title="voice.legacy ? t('models.detail.hide') : t('common.delete')"
        :message="voice.legacy ? t('models.detail.hideConfirm', { name: voice.name }) : t('models.detail.deleteConfirm', { name: voice.name })"
        :confirm-label="voice.legacy ? t('models.detail.hide') : t('common.delete')"
        :cancel-label="t('common.cancel')"
        danger
        @confirm="doDelete"
      />
    </template>
  </div>
  </Transition>
</template>

<style scoped>
.detail { display: flex; flex-direction: column; gap: var(--app-space-4); padding: var(--app-space-4); min-width: 0; transform-origin: var(--from-origin, center); }
.head { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.title { flex: 1; margin: 0; min-width: 0; overflow-wrap: anywhere; }
/* One column of sections (as Convert, Separate, Live); the fields inside wrap in a grid. */
.sections { display: flex; flex-direction: column; gap: var(--app-space-4); }
.fields { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(280px, 100%), 1fr)); gap: var(--app-space-3) var(--app-space-6); }
.panel { display: flex; flex-direction: column; gap: var(--app-space-3); padding: var(--app-space-4); min-width: 0; }
.sub { margin: 0; }
.speakers { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: var(--app-space-2); }
.actions { display: flex; justify-content: flex-end; }
.indexes { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--app-space-1); }
.index { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.path { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; direction: rtl; text-align: left; }
/* A field and its button (Add, Build): the button centred on the field. */
.row { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2); }
.row > :first-child { flex: 1; min-width: 0; }
</style>
