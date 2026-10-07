<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useRouter } from 'vue-router';
import { uploadRaw } from '@/api/client';
import { useBaseModels } from '@/api/queries/imports';
import { useSeparationPresets } from '@/api/queries/separate';
import { fetchMetrics, useCheckpoints, useExperiment, useExperimentMutations, useTrainingSamples } from '@/api/queries/train';
import type { Checkpoint, Dataset, DatasetReport as Report, Experiment, FitSettings, S } from '@/api/types';
import AssetGate from '@/components/AssetGate.vue';
import DatasetReport from '@/components/DatasetReport.vue';
import ErrorNotice from '@/components/ErrorNotice.vue';
import JobCard from '@/components/JobCard.vue';
import LossChart from '@/components/LossChart.vue';
import ResultsList from '@/components/ResultsList.vue';
import { embedderAsset, pitchAssets } from '@/components/paramFields';
import ServerPathDialog from '@/components/ServerPathDialog.vue';
import SpeakerTable from '@/components/SpeakerTable.vue';
import StageStepper, { type ExperimentStep } from '@/components/StageStepper.vue';
import { useI18n } from '@/i18n';
import { formatBytes, formatDate } from '@/format';
import { useHandoffStore } from '@/stores/handoff';
import { useJobsStore } from '@/stores/jobs';
import { useTrainStore } from '@/stores/train';
import { AppButton, AppCard, AppDialog, AxisPanes, Badge, ConfirmDialog, DropZone, EmptyState, ExpansionPanel, IconButton, ParamSlider, PathField, ProgressBar, SegmentedControl, SelectField, Surface, Switch, TextField, icons, useSnackbar } from '@/ui';

const props = defineProps<{ name: string }>();
const { t, tOr, locale } = useI18n();
const router = useRouter();
const snackbar = useSnackbar();
const exp = useExperiment(() => props.name);
const checkpoints = useCheckpoints(() => props.name);
const samples = useTrainingSamples(() => props.name);
const sepPresets = useSeparationPresets();
const m = useExperimentMutations();
const jobs = useJobsStore();
const train = useTrainStore();
const handoff = useHandoffStore();

const step = ref<ExperimentStep>('dataset');
const axisDir = ref(1);
const ORDER: ExperimentStep[] = ['dataset', 'settings', 'run', 'results'];
watch(step, (now, before) => (axisDir.value = ORDER.indexOf(now) >= ORDER.indexOf(before) ? 1 : -1));
const STAGES = ['clean', 'slice', 'f0', 'features', 'fit', 'index'] as const;

// Editable copies of the dataset and the settings; saved explicitly.
const dataset = ref<Dataset | null>(null);
const slicing = ref<S['SliceSettings-Output'] | null>(null);
const slicingOpen = ref(false);
const settingsForm = ref<{ sample_rate: string; version: string; pitch_guidance: boolean; f0_method: string; embedder: string; fit: FitSettings } | null>(null);
watch(
  () => exp.data.value,
  (e) => {
    if (!e) return;
    if (!dataset.value) dataset.value = JSON.parse(JSON.stringify(e.dataset));
    if (!slicing.value) slicing.value = { ...e.slicing };
    if (!settingsForm.value) settingsForm.value = { sample_rate: e.sample_rate, version: e.version, pitch_guidance: e.pitch_guidance, f0_method: e.f0_method, embedder: e.embedder, fit: { ...e.fit } };
  },
  { immediate: true },
);
const error = ref<unknown>(null);
const report = ref<Report | null>(null);
const browseOpen = ref(false);
const deleteOpen = ref(false);

const DATASET_MODES = ['single', 'multi'] as const;
const modes = computed(() => [
  { value: 'single' as const, label: t('train.mode.single') },
  { value: 'multi' as const, label: t('train.mode.multi') },
]);
const cleanOptions = computed(() => [{ value: '', label: t('train.cleanNone') }, ...(sepPresets.data.value ?? []).map((p) => ({ value: p.id, label: p.title }))]);
const running = computed(() => !!exp.data.value?.running_job);
const runningJob = computed(() => (exp.data.value?.running_job ? (jobs.byId[exp.data.value.running_job] ?? null) : null));
const onErr = (e: unknown) => (error.value = e);
const set = (e: Experiment) => ((dataset.value = JSON.parse(JSON.stringify(e.dataset))), snackbar.show(t('models.detail.saved')));

function saveDataset() {
  if (!dataset.value) return;
  error.value = null;
  m.update.mutate({ name: props.name, patch: { dataset: dataset.value, slicing: slicing.value } }, { onSuccess: set, onError: onErr });
}
// Audio from this computer, into the experiment's own dataset folder (a server folder is not needed).
const uploading = ref<{ name: string; done: number; total: number } | null>(null);
async function onDatasetFiles(files: File[]) {
  error.value = null;
  try {
    let latest: Experiment | null = null;
    for (const [i, file] of files.entries()) {
      uploading.value = { name: file.name, done: i, total: files.length };
      latest = await uploadRaw<Experiment>(`/api/v1/train/experiments/${encodeURIComponent(props.name)}/dataset/files`, { filename: file.name }, file);
    }
    if (latest) set(latest);
    await exp.refetch();
  } catch (e) {
    error.value = e;
  } finally {
    uploading.value = null;
  }
}
const uploadedHere = computed(() => !!dataset.value?.folder && !!exp.data.value && dataset.value.folder.startsWith(exp.data.value.path) && dataset.value.folder.endsWith('dataset_upload'));
function clearUploads() {
  m.clearUploads.mutate(props.name, { onSuccess: set, onError: onErr });
}

function scan() {
  error.value = null;
  m.scan.mutate(props.name, { onSuccess: (r) => (report.value = r), onError: onErr });
}
function fromFolders(folder: string) {
  m.fromFolders.mutate({ name: props.name, folder }, { onSuccess: set, onError: onErr });
}
function saveSettings() {
  const f = settingsForm.value;
  if (!f) return;
  error.value = null;
  m.update.mutate(
    { name: props.name, patch: { sample_rate: f.sample_rate as '40k', version: f.version as 'v2', pitch_guidance: f.pitch_guidance, f0_method: f.f0_method as 'rmvpe', embedder: f.embedder as 'contentvec', fit: f.fit } },
    { onSuccess: () => snackbar.show(t('models.detail.saved')), onError: onErr },
  );
}
function run(stages: string[] | null, force = false) {
  error.value = null;
  m.run.mutate({ name: props.name, body: { stages: stages as never, force } }, { onError: onErr });
}

// Metrics: REST once, then train_metrics events extend the store.
watch(
  () => [props.name, step.value] as const,
  async ([name, s]) => {
    if (s !== 'results' && s !== 'run') return;
    try {
      train.set(name, await fetchMetrics(name, 0));
    } catch {
      /* not trained yet */
    }
  },
  { immediate: true },
);
const metrics = computed(() => train.metrics[props.name] ?? []);

const exportOpen = ref(false);
const exportName = ref('');
const exportFrom = ref<string | null>(null);
function openExport(c: Checkpoint | null) {
  exportFrom.value = c?.name ?? null;
  exportName.value = props.name;
  exportOpen.value = true;
}
function doExport() {
  m.exportVoice.mutate({ name: props.name, body: { checkpoint: exportFrom.value, voice_name: exportName.value.trim() || null } }, { onSuccess: () => ((exportOpen.value = false), snackbar.show(t('train.exported'))), onError: onErr });
}
function tryIt(c: Checkpoint) {
  m.tryCheckpoint.mutate(
    { name: props.name, body: { checkpoint: c.name, voice_name: null } },
    { onSuccess: (voice) => (handoff.sendVoice(voice.id), router.push('/convert')), onError: onErr },
  );
}
function remove() {
  m.remove.mutate(props.name, { onSuccess: () => router.push('/train'), onError: onErr });
}

const cutOptions = computed(() => ['auto', 'fixed', 'none'].map((v) => ({ value: v, label: t(`train.cuts.${v}`) })));
const normalizeOptions = computed(() => ['slice', 'file', 'none'].map((v) => ({ value: v, label: t(`train.normalizes.${v}`) })));
const rateOptions = ['32k', '40k', '48k'].map((v) => ({ value: v, label: v }));
const versionOptions = ['v1', 'v2'].map((v) => ({ value: v, label: v }));
const f0Options = computed(() => ['rmvpe', 'pm', 'fcpe', 'crepe', 'crepe-tiny', 'swift'].map((v) => ({ value: v, label: t(`params.f0.${v}`) })));
const EMBEDDERS = ['contentvec', 'spin', 'spin-v2', 'chinese-hubert-base', 'japanese-hubert-base', 'korean-hubert-base'];
const V1_EMBEDDERS = ['contentvec', 'spin', 'spin-v2'];
const embedderOptions = computed(() => EMBEDDERS.filter((e) => settingsForm.value?.version !== 'v1' || V1_EMBEDDERS.includes(e)).map((v) => ({ value: v, label: t(`train.embedders.${v}`) })));
const precisionOptions = computed(() => ['auto', 'fp32', 'bf16'].map((v) => ({ value: v, label: t(`train.precisions.${v}`) })));
// Base models that fit the experiment: official and imported; none chosen means the official one.
const baseFilter = computed(() => (settingsForm.value ? { sample_rate: settingsForm.value.sample_rate, version: settingsForm.value.version, pitch_guidance: settingsForm.value.pitch_guidance } : null));
const baseModels = useBaseModels(baseFilter);
const baseOptions = computed(() => [
  { value: '', label: t('train.baseDefault') },
  ...(baseModels.data.value ?? []).filter((b) => b.source === 'imported').map((b) => ({ value: b.id, label: b.has_discriminator ? b.name : `${b.name} · ${t('train.baseNoD')}` })),
  ...(baseModels.data.value ?? []).filter((b) => b.source === 'community').map((b) => ({ value: b.id, label: `${b.name} · ${t('train.baseCommunity')}` })),
]);
watch(
  () => baseModels.data.value,
  (list) => {
    const f = settingsForm.value;
    if (list && f?.fit.base_model && !list.some((b) => b.id === f.fit.base_model)) f.fit.base_model = null;
  },
);
const importedBase = computed(() => !!settingsForm.value?.fit.base_model && !settingsForm.value.fit.base_model.startsWith('official-'));
const communityBase = computed(() => settingsForm.value?.fit.base_model?.startsWith('community-') ? [settingsForm.value.fit.base_model] : []);
const baseAssets = computed(() => {
  const f = settingsForm.value;
  if (!f) return [];
  return [...(importedBase.value ? communityBase.value : [`pretrained-${f.version}-${f.sample_rate}`]), embedderAsset(f.embedder), ...pitchAssets(f.pitch_guidance, f.f0_method)];
});
const smallCheckpoints = computed(() => (checkpoints.data.value ?? []).filter((c) => c.kind !== 'D'));
const stageTone = (status?: string) => (({ done: 'primary', running: 'primary', failed: 'error', stale: 'warning' }) as Record<string, 'primary' | 'error' | 'warning'>)[status ?? ''] ?? 'neutral';
const num = (v: unknown) => (v === '' || v === null || v === undefined ? null : Number(v));
</script>

<template>
  <div class="experiment">
    <ErrorNotice v-if="exp.error.value" :error="exp.error.value" />
    <template v-else-if="exp.data.value">
      <header class="head">
        <IconButton :icon="icons.ArrowLeft" :label="t('common.back')" @click="router.push('/train')" />
        <div class="titles">
          <h2 class="type-headline-small title">{{ exp.data.value.name }}</h2>
          <span class="type-body-small muted">{{ exp.data.value.sample_rate }} · {{ exp.data.value.version }} · {{ exp.data.value.path }}</span>
        </div>
        <IconButton :icon="icons.Trash2" :label="t('common.delete')" :disabled="running" @click="deleteOpen = true" />
      </header>
      <StageStepper v-model="step" :experiment="exp.data.value" />
      <ErrorNotice v-if="error" :error="error" />
      <div class="stage" :style="{ '--axis-dir': axisDir }">
        <Transition name="shared-axis-x" mode="out-in">
          <Surface v-if="step === 'dataset' && dataset" key="dataset" :level="0" class="panel">
            <SegmentedControl v-model="dataset.mode" :options="modes" />
            <AxisPanes :value="dataset.mode" :order="DATASET_MODES">
              <div v-if="dataset.mode === 'single'" key="single" class="folder">
                <PathField :model-value="dataset.folder ?? ''" :label="t('train.folder')" @update:model-value="dataset.folder = $event || null" />
                <AppButton variant="text" @click="browseOpen = true">{{ t('common.browse') }}</AppButton>
              </div>
              <SpeakerTable v-else key="multi" v-model="dataset.speakers" @from-folders="fromFolders" />
            </AxisPanes>
            <DropZone
              :label="dataset.mode === 'multi' ? t('train.uploadSpeakers') : t('train.upload')"
              :hint="dataset.mode === 'multi' ? '.zip' : t('train.uploadHint')"
              :accept="dataset.mode === 'multi' ? '.zip' : 'audio/*,.zip'"
              compact
              :disabled="running || !!uploading"
              @files="onDatasetFiles"
            />
            <div v-if="uploading" class="upload">
              <span class="type-body-small muted">{{ t('train.uploading', { name: uploading.name, n: uploading.done + 1, total: uploading.total }) }}</span>
              <ProgressBar :value="uploading.done / uploading.total" />
            </div>
            <AppButton v-if="uploadedHere" variant="text" :icon="icons.Trash2" :disabled="running" @click="clearUploads">{{ t('train.clearUploads') }}</AppButton>
            <SelectField :model-value="dataset.clean_preset ?? ''" :label="t('train.cleanPreset')" :options="cleanOptions" @update:model-value="dataset.clean_preset = $event || null" />
            <ExpansionPanel v-if="slicing" v-model:open="slicingOpen" :label="t('train.slicing')" :supporting-text="t('train.slicingHint')">
              <div class="form">
                <SelectField v-model="slicing.cut" :label="t('train.cut')" :options="cutOptions" />
                <TextField
                  type="number"
                  :min="0.5"
                  :max="10"
                  :step="0.1"
                  :model-value="slicing.chunk_seconds ?? ''"
                  :label="t('train.chunk')"
                  :supporting-text="t('train.chunkAuto')"
                  @update:model-value="slicing.chunk_seconds = num($event)"
                />
                <SelectField v-model="slicing.normalize" :label="t('train.normalize')" :options="normalizeOptions" />
              </div>
              <ParamSlider v-model="slicing.overlap" :label="t('train.overlap')" :min="0" :max="0.4" :step="0.05" unit="s" :default-value="0.3" :reset-label="t('common.reset')" />
              <ParamSlider v-model="slicing.denoise" :label="t('train.denoise')" :help="t('train.denoiseHelp')" :min="0" :max="1" :step="0.05" unit="" :default-value="0" :reset-label="t('common.reset')" />
              <Switch v-model="slicing.highpass" :label="t('train.highpass')" :supporting-text="t('train.highpassHint')" />
            </ExpansionPanel>
            <div class="actions">
              <AppButton variant="tonal" :icon="icons.Search" :loading="m.scan.isPending.value" @click="scan">{{ t('train.scan') }}</AppButton>
              <AppButton :icon="icons.Save" :loading="m.update.isPending.value" @click="saveDataset">{{ t('common.save') }}</AppButton>
            </div>
            <DatasetReport v-if="report" :report="report" />
            <ServerPathDialog v-model:open="browseOpen" folders :files="false" @select="(s) => s[0] && dataset && (dataset.folder = s[0].path)" />
          </Surface>
          <Surface v-else-if="step === 'settings' && settingsForm" key="settings" :level="0" class="panel">
            <div class="form">
              <SelectField v-model="settingsForm.sample_rate" :label="t('train.sampleRate')" :options="rateOptions" />
              <SelectField v-model="settingsForm.version" :label="t('train.version')" :options="versionOptions" />
              <SelectField v-model="settingsForm.f0_method" :label="t('train.f0Method')" :options="f0Options" :disabled="!settingsForm.pitch_guidance" />
              <SelectField v-model="settingsForm.embedder" :label="t('train.embedder')" :options="embedderOptions" :supporting-text="t('train.embedderHint')" />
              <TextField type="number" :min="1" :model-value="settingsForm.fit.epochs" :label="t('train.epochs')" @update:model-value="settingsForm.fit.epochs = Number($event)" />
              <TextField type="number" :min="1" :model-value="settingsForm.fit.batch_size ?? ''" :label="t('train.batchSize')" :supporting-text="t('train.batchAuto')" @update:model-value="settingsForm.fit.batch_size = num($event)" />
              <TextField type="number" :min="1" :model-value="settingsForm.fit.save_every" :label="t('train.saveEvery')" @update:model-value="settingsForm.fit.save_every = Number($event)" />
              <TextField :model-value="settingsForm.fit.gpus" :label="t('train.gpus')" @update:model-value="settingsForm.fit.gpus = String($event ?? 'auto')" />
            </div>
            <Switch v-model="settingsForm.pitch_guidance" :label="t('train.pitchGuidance')" />
            <Switch v-model="settingsForm.fit.cache_in_gpu" :label="t('train.cacheInGpu')" />
            <Switch v-model="settingsForm.fit.save_small_every" :label="t('train.saveSmallEvery')" />
            <Switch v-model="settingsForm.fit.save_latest_only" :label="t('train.saveLatestOnly')" />
            <Switch v-model="settingsForm.fit.previews" :label="t('train.previews')" :supporting-text="t('train.previewsHint')" />
            <h3 class="type-title-small sub">{{ t('train.advanced') }}</h3>
            <div class="form">
              <SelectField v-model="settingsForm.fit.precision" :label="t('train.precision')" :options="precisionOptions" :supporting-text="t('train.precisionHint')" />
            </div>
            <Switch v-model="settingsForm.fit.tf32" :label="t('train.tf32')" :supporting-text="t('train.tf32Hint')" />
            <Switch v-model="settingsForm.fit.checkpointing" :label="t('train.checkpointing')" :supporting-text="t('train.checkpointingHint')" />
            <Switch v-model="settingsForm.fit.fresh_speakers" :label="t('train.freshSpeakers')" :supporting-text="t('train.freshSpeakersHint')" />
            <h3 class="type-title-small sub">{{ t('train.baseModels') }}</h3>
            <SelectField
              :model-value="settingsForm.fit.base_model ?? ''"
              :label="t('train.baseModel')"
              :options="baseOptions"
              :supporting-text="t('train.baseHint')"
              @update:model-value="settingsForm.fit.base_model = $event || null"
            />
            <p v-if="settingsForm.fit.pretrained_g && !settingsForm.fit.base_model" class="type-body-small muted">{{ settingsForm.fit.pretrained_g }}</p>
            <AssetGate :assets="baseAssets">
              <div class="actions">
                <AppButton :icon="icons.Save" :loading="m.update.isPending.value" @click="saveSettings">{{ t('common.save') }}</AppButton>
              </div>
            </AssetGate>
          </Surface>
          <Surface v-else-if="step === 'run'" key="run" :level="0" class="panel">
            <div class="actions start">
              <AppButton :icon="icons.Play" :disabled="running" :loading="m.run.isPending.value" @click="run(null)">{{ t('train.runAll') }}</AppButton>
              <AppButton v-if="running" variant="tonal" :icon="icons.Square" @click="m.stop.mutate(name, { onError: onErr })">{{ t('train.stop') }}</AppButton>
            </div>
            <JobCard v-if="runningJob" :job="runningJob" />
            <div class="stages">
              <AppCard v-for="s in STAGES" :key="s" variant="outlined" class="stage-card" :class="exp.data.value.stages[s]?.status ?? 'pending'">
                <div class="stage-head">
                  <span class="type-title-small">{{ tOr(`train.stages.${s}`, s) }}</span>
                  <Badge :value="tOr(`train.stageStates.${exp.data.value.stages[s]?.status ?? 'pending'}`, '')" :tone="stageTone(exp.data.value.stages[s]?.status)" />
                </div>
                <span v-if="exp.data.value.stages[s]?.status === 'stale'" class="type-body-small warn">{{ t('train.staleNote') }}</span>
                <span v-if="exp.data.value.stages[s]?.error" class="type-body-small err">{{ exp.data.value.stages[s]?.error }}</span>
                <span class="type-body-small muted">
                  {{ Object.entries(exp.data.value.stages[s]?.counts ?? {}).map(([k, v]) => `${k}: ${v}`).join(' · ') }}
                  <template v-if="exp.data.value.stages[s]?.finished_at"> · {{ formatDate(exp.data.value.stages[s]?.finished_at, locale) }}</template>
                </span>
                <AppButton variant="text" :disabled="running" @click="run([s], exp.data.value.stages[s]?.status === 'done')">{{ exp.data.value.stages[s]?.status === 'done' ? t('train.rerun') : t('train.runStage') }}</AppButton>
              </AppCard>
            </div>
          </Surface>
          <Surface v-else-if="step === 'results'" key="results" :level="0" class="panel">
            <h3 class="type-title-small sub">{{ t('train.loss') }}</h3>
            <LossChart :metrics="metrics" />
            <h3 class="type-title-small sub">{{ t('train.gradients') }}</h3>
            <LossChart :metrics="metrics" gradients />
            <template v-if="samples.data.value?.length">
              <h3 class="type-title-small sub">{{ t('train.samples') }}</h3>
              <p class="type-body-small muted">{{ t('train.samplesHint') }}</p>
              <ResultsList :outputs="samples.data.value" />
            </template>
            <div class="ck-head">
              <h3 class="type-title-small sub">{{ t('train.checkpoints') }}</h3>
              <AppButton variant="tonal" :icon="icons.PackageOpen" :disabled="!smallCheckpoints.length" @click="openExport(null)">{{ t('train.export') }}</AppButton>
            </div>
            <EmptyState v-if="!smallCheckpoints.length" :icon="icons.Boxes" :title="t('train.noCheckpoints')" />
            <ul v-else class="checkpoints">
              <li v-for="c in smallCheckpoints" :key="c.path" class="ck">
                <Badge :value="c.kind" tone="neutral" />
                <span class="type-body-large name">{{ c.name }}</span>
                <span class="type-body-small muted">{{ c.epoch !== null ? `${t('train.epoch')} ${c.epoch}` : '' }} · {{ formatBytes(c.size) }}</span>
                <AppButton variant="text" :icon="icons.Play" @click="tryIt(c)">{{ t('train.try') }}</AppButton>
                <AppButton variant="text" :icon="icons.PackageOpen" @click="openExport(c)">{{ t('train.export') }}</AppButton>
              </li>
            </ul>
          </Surface>
        </Transition>
      </div>
      <AppDialog v-model:open="exportOpen" :title="t('train.export')" width="small" :close-label="t('common.close')">
        <p class="type-body-small muted">{{ exportFrom ?? '' }}</p>
        <TextField v-model="exportName" :label="t('train.exportName')" @enter="doExport" />
        <template #actions>
          <AppButton variant="text" @click="exportOpen = false">{{ t('common.cancel') }}</AppButton>
          <AppButton :disabled="!exportName.trim()" @click="doExport">{{ t('train.export') }}</AppButton>
        </template>
      </AppDialog>
      <ConfirmDialog v-model:open="deleteOpen" :title="t('common.delete')" :message="t('train.deleteConfirm', { name })" :confirm-label="t('common.delete')" :cancel-label="t('common.cancel')" danger @confirm="remove" />
    </template>
  </div>
</template>

<style scoped>
.experiment { display: flex; flex-direction: column; gap: var(--app-space-4); padding: var(--app-space-4); min-width: 0; }
.head { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.titles { flex: 1; min-width: 0; display: flex; flex-direction: column; }
.title { margin: 0; overflow-wrap: anywhere; }
.stage { position: relative; overflow: hidden; }
.panel { display: flex; flex-direction: column; gap: var(--app-space-3); padding: var(--app-space-4); min-width: 0; }
/* The folder field and Browse: the button centred on the field. */
.folder { display: flex; align-items: center; gap: var(--app-space-2); }
.folder > :first-child { flex: 1; min-width: 0; }
.form { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: var(--app-space-3); }
.actions { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: var(--app-space-2); }
.upload { display: flex; flex-direction: column; gap: var(--app-space-1); }
.actions.start { justify-content: flex-start; }
.sub { margin: var(--app-space-2) 0 0; }
.stages { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: var(--app-space-2); }
.stage-card { display: flex; flex-direction: column; gap: var(--app-space-1); padding: var(--app-space-3); min-width: 0; }
.stage-head { display: flex; align-items: center; justify-content: space-between; gap: var(--app-space-2); }
.warn { color: var(--md-sys-color-tertiary); }
.err { color: var(--md-sys-color-error); overflow-wrap: anywhere; }
.ck-head { display: flex; align-items: center; justify-content: space-between; gap: var(--app-space-2); }
.checkpoints { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--app-space-1); }
.ck { display: flex; align-items: center; flex-wrap: wrap; gap: var(--app-space-2); padding: var(--app-space-1) var(--app-space-2); border-radius: var(--md-sys-shape-corner-small); background: var(--md-sys-color-surface-container); min-width: 0; }
.ck .name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
