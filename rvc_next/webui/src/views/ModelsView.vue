<script setup lang="ts">
import { computed, ref } from 'vue';
import { useRouter } from 'vue-router';
import { useSettings } from '@/api/queries/app';
import { useAssets, useDeleteAsset, useDownloadAssets } from '@/api/queries/assets';
import { type BaseModel, type ImportResult, type SeparationModel, useBaseModelMutations, useBaseModels, useSeparationModelMutations, useSeparationModels } from '@/api/queries/imports';
import { useCatalogVoices, useDeleteModel, useDownloadCatalogVoices, useLegacyRoots, useModelJobs, useModels } from '@/api/queries/models';
import { useSeparationPresets } from '@/api/queries/separate';
import type { AssetStatus, CatalogVoice, VoiceModel } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import ImportReviewDialog from '@/components/ImportReviewDialog.vue';
import IndexInbox from '@/components/IndexInbox.vue';
import ResourceRow, { type ResourceState } from '@/components/ResourceRow.vue';
import LinkImport from '@/components/LinkImport.vue';
import { useModelImport } from '@/components/useModelImport';
import ServerPathDialog from '@/components/ServerPathDialog.vue';
import { useI18n } from '@/i18n';
import { formatBytes, formatRate } from '@/format';
import { AppButton, AppCard, AppDialog, Badge, ConfirmDialog, DropZone, EmptyState, IconButton, ParamSlider, PathField, ProgressBar, SearchField, SelectField, Surface, Switch, Tabs, TextField, TRANSITIONS, icons, staggerStyle, useAxisDirection, useSnackbar } from '@/ui';

type Tab = 'voices' | 'base' | 'separation' | 'assets' | 'tools';
const { t, tOr } = useI18n();
const router = useRouter();
const snackbar = useSnackbar();
const tab = ref<Tab>('voices');
const TABS = ['voices', 'base', 'separation', 'assets', 'tools'] as const;
const tabs = computed(() => TABS.map((v) => ({ value: v, label: t(`models.tabs.${v}`) })));
// Tabs slide the way they moved (shared-axis-x), as Settings' sections and Hanaikada's tabs.
const axisDir = useAxisDirection(tab, () => TABS);

// Voices
const showHidden = ref(false);
const query = ref('');
const onlyNoIndex = ref(false);
const models = useModels(showHidden);
// Ready-made voices from the download repository; listed with the other downloads in Assets.
const catalog = useCatalogVoices();
const downloadVoices = useDownloadCatalogVoices();
const missingDemo = computed(() => (catalog.data.value ?? []).filter((v) => !v.installed_voice_id));
const demoState = (v: CatalogVoice): ResourceState => (v.installed_voice_id ? 'installed' : v.job_id ? 'downloading' : 'missing');
function describeDemo(v: CatalogVoice) {
  return [formatRate(v.sample_rate), v.version, v.has_index ? t('models.demo.withIndex') : t('voice.noIndex'), formatBytes(v.size)].join(' · ');
}
function downloadDemo(ids: string[] | null) {
  downloadVoices.mutate(ids ? { ids } : { all: true }, { onSuccess: () => snackbar.show(t('models.demo.started')), onError: (e) => (error.value = e) });
}
const error = ref<unknown>(null);
const voices = computed(() => {
  const q = query.value.trim().toLowerCase();
  return (models.data.value ?? []).filter((v) => (!q || v.name.toLowerCase().includes(q) || v.tags.some((x) => x.toLowerCase().includes(q))) && (!onlyNoIndex.value || !v.has_index));
});

// Every drop zone imports through one session: the server identifies each file by its content and
// pairs them; the review dialog opens only when it cannot decide alone.
const importer = useModelImport(reportImport, (e) => (error.value = e));
function onFiles(files: File[]) {
  error.value = null;
  importer.importFiles(files);
}
function onLinks(text: string) {
  error.value = null;
  importer.importLinks(text);
}
function reportImport(r: ImportResult) {
  const name = (p: string) => p.split(/[\\/]/).pop() ?? p;
  if (r.voices.length) {
    const names = r.voices.map((v) => v.name).join(', ');
    const attached = r.attached.filter((a) => r.voices.some((v) => v.id === a.voice_id)).length;
    snackbar.show(attached ? t('models.importedWithIndex', { names, n: attached }) : t('models.imported', { names }));
  }
  const toExisting = r.attached.filter((a) => !r.voices.some((v) => v.id === a.voice_id));
  if (toExisting.length) snackbar.show(t('models.indexAttached', { names: toExisting.map((a) => a.name).join(', ') }));
  if (r.inbox.length) snackbar.show(t('models.indexToInbox', { names: r.inbox.map(name).join(', ') }));
  if (r.base_models.length) snackbar.show(t('models.baseImported', { n: r.base_models.length }));
  if (r.separation_models.length) snackbar.show(t('models.separationImported', { n: r.separation_models.length }));
  for (const c of r.checkpoints) snackbar.show(t('models.checkpointFound', { name: name(c) }));
  if (r.skipped.length) snackbar.show(t('models.skipped', { names: r.skipped.map(name).join(', ') }));
}
const describe = (v: VoiceModel) => [formatRate(v.sample_rate), v.version, v.pitch_guidance ? t('voice.pitchGuidance') : t('voice.noPitchGuidance')].join(' · ');
function open(v: VoiceModel, e: MouseEvent | KeyboardEvent) {
  const rect = (e.currentTarget as HTMLElement | null)?.getBoundingClientRect?.();
  if (rect) sessionStorage.setItem('rvc-next:fromRect', JSON.stringify({ left: rect.left, top: rect.top, width: rect.width, height: rect.height }));
  router.push(`/models/${encodeURIComponent(v.id)}`);
}

// Legacy roots
const settings = useSettings();
const legacy = useLegacyRoots();
const legacyOpen = ref(false);
const legacyPath = ref('');
const legacyBrowse = ref(false);
function addLegacy() {
  legacy.add.mutate({ path: legacyPath.value.trim(), name: '' }, { onSuccess: () => (legacyOpen.value = false), onError: (e) => (error.value = e) });
}

// Base models
const baseModels = useBaseModels();
const baseM = useBaseModelMutations();
const officialBase = computed(() => (baseModels.data.value ?? []).filter((b) => b.source === 'official'));
const importedBase = computed(() => (baseModels.data.value ?? []).filter((b) => b.source === 'imported'));
const communityBase = computed(() => (baseModels.data.value ?? []).filter((b) => b.source === 'community'));
/** A community base model: its licence and repository, from the catalog. */
function describeCommunity(b: BaseModel) {
  const info = b.asset_id ? assetById.value[b.asset_id]?.base_model : null;
  const licence = info?.license ? t('models.base.licence', { licence: info.license }) : t('models.base.noLicence');
  return [describeBase(b), licence, info?.homepage.replace('https://', '') ?? ''].filter(Boolean).join(' · ');
}
const describeBase = (b: BaseModel) => [b.version, b.sample_rate, b.pitch_guidance ? t('voice.pitchGuidance') : t('voice.noPitchGuidance'), b.has_discriminator ? 'G + D' : t('models.base.noD'), ...(b.size ? [formatBytes(b.size)] : [])].join(' · ');
const renaming = ref<{ kind: 'base'; id: string; name: string } | null>(null);
function rename() {
  const r = renaming.value;
  if (!r || !r.name.trim()) return;
  baseM.rename.mutate({ id: r.id, name: r.name.trim() }, { onSuccess: () => (renaming.value = null), onError: (e) => (error.value = e) });
}
// Deleting anything: imported models go to the trash; downloaded ones are removed and can be
// downloaded again (an asset, an official base model's own G and D, a demo voice in the library).
type Removable = { kind: 'base' | 'separation' | 'asset' | 'voice'; id: string; name: string };
const removing = ref<Removable | null>(null);
const sepM = useSeparationModelMutations();
const deleteAsset = useDeleteAsset();
const deleteVoice = useDeleteModel();
const removeBusy = computed(() => baseM.remove.isPending.value || sepM.remove.isPending.value || deleteAsset.isPending.value || deleteVoice.isPending.value);
function remove() {
  const r = removing.value;
  if (!r) return;
  const done = { onSuccess: () => (removing.value = null), onError: (e: unknown) => ((error.value = e), (removing.value = null)) };
  if (r.kind === 'base') baseM.remove.mutate(r.id, done);
  else if (r.kind === 'separation') sepM.remove.mutate(r.id, done);
  else if (r.kind === 'asset') deleteAsset.mutate(r.id, done);
  else deleteVoice.mutate(r.id, done);
}

// Assets: every downloadable model, and the demo voices, in one list.
const assets = useAssets();
const download = useDownloadAssets();
const assetById = computed(() => Object.fromEntries((assets.data.value ?? []).map((a) => [a.id, a])));
const GROUPS = ['inference', 'training', 'separation'] as const;
const groups = computed(() => GROUPS.map((g) => [g, (assets.data.value ?? []).filter((a) => a.group === g)] as const).filter(([, list]) => list.length));
const describeAsset = (a: AssetStatus) => [a.description, formatBytes(a.size), ...(a.verified ? [t('assets.verified')] : [])].filter(Boolean).join(' · ');
const fetchAssets = (ids: string[]) => download.mutate({ ids }, { onError: (e) => (error.value = e) });

// An official base model is part of its rate's asset: downloading fetches the asset (both pitch
// variants), deleting removes this one's G and D.
function baseState(b: BaseModel): ResourceState {
  const a = b.asset_id ? assetById.value[b.asset_id] : undefined;
  if (a?.state === 'downloading') return 'downloading';
  return b.installed ? 'installed' : 'missing';
}

// Separation models: the built-in ones are assets; each serves one or more presets.
const sepModels = useSeparationModels();
const sepPresets = useSeparationPresets();
const builtinSep = computed(() => (assets.data.value ?? []).filter((a) => a.group === 'separation'));
const presetsUsing = (assetId: string) => (sepPresets.data.value ?? []).filter((p) => p.assets.includes(assetId)).map((p) => p.title);
const describeBuiltinSep = (a: AssetStatus) => [a.description.replace(/\.$/, ''), t('models.separation.usedBy', { list: presetsUsing(a.id).join(', ') || '—' }), formatBytes(a.size)].join(' · ');
const describeSep = (s: SeparationModel) => [s.model_type, `${s.primary_label} / ${s.secondary_label}`, formatBytes(s.size)].join(' · ');

// Tools
const jobsApi = useModelJobs();
const voiceOptions = computed(() => (models.data.value ?? []).map((v) => ({ value: v.id, label: v.name })));
const mergeA = ref<string | null>(null);
const mergeB = ref<string | null>(null);
const alpha = ref(0.5);
const mergeName = ref('');
const extractPath = ref('');
const extractName = ref('');
const extractRate = ref<string | null>('');
const extractVersion = ref<string | null>('');
const rateOptions = computed(() => [{ value: '', label: t('models.extract.auto') }, ...['32k', '40k', '48k'].map((v) => ({ value: v, label: v }))]);
const versionOptions = computed(() => [{ value: '', label: t('models.extract.auto') }, { value: 'v1', label: 'v1' }, { value: 'v2', label: 'v2' }]);
function merge() {
  if (!mergeA.value || !mergeB.value) return;
  jobsApi.merge.mutate({ a: mergeA.value, b: mergeB.value, alpha: alpha.value, name: mergeName.value.trim(), info: '' }, { onError: (e) => (error.value = e) });
}
function extract() {
  jobsApi.extract.mutate(
    { checkpoint: extractPath.value.trim(), name: extractName.value.trim(), sample_rate: (extractRate.value || null) as '40k' | null, version: (extractVersion.value || null) as 'v2' | null, pitch_guidance: null, info: '' },
    { onError: (e) => (error.value = e) },
  );
}
</script>

<template>
  <div class="models">
    <Tabs v-model="tab" :tabs="tabs" />
    <ErrorNotice v-if="error" :error="error" />
    <div class="panes" :style="{ '--axis-dir': axisDir }">
      <Transition :name="TRANSITIONS.sharedAxisX">
        <section v-if="tab === 'voices'" key="voices" class="pane">
          <DropZone :label="t('models.import')" :hint="t('models.importHint')" accept=".pth,.zip,.index" compact @files="onFiles" />
          <ProgressBar v-if="importer.uploading.value" :value="importer.uploading.value.progress" :label="importer.uploading.value.name" />
          <LinkImport :fetching="importer.fetching.value" @links="onLinks" />
          <IndexInbox @error="error = $event" />
          <div class="filters">
            <SearchField v-model="query" :label="t('models.filter')" />
            <Switch v-model="onlyNoIndex" :label="t('models.noIndexFilter')" />
            <Switch v-model="showHidden" :label="t('models.showHidden')" />
          </div>
          <EmptyState v-if="models.isSuccess.value && !voices.length" :icon="icons.Library" :title="t('voice.none')" :text="t('voice.noneHint')">
            <AppButton v-if="missingDemo.length && !query && !onlyNoIndex" variant="tonal" :icon="icons.Download" @click="tab = 'assets'">{{ t('models.demo.get') }}</AppButton>
          </EmptyState>
          <TransitionGroup v-else :name="TRANSITIONS.list" tag="div" class="grid">
            <AppCard v-for="(v, i) in voices" :key="v.id" interactive variant="outlined" class="voice" :style="staggerStyle(i)" @activate="open(v, $event)">
              <span class="type-title-medium name">{{ v.name }}</span>
              <span class="type-body-small muted">{{ describe(v) }}</span>
              <div class="badges">
                <Badge v-if="!v.has_index" :value="t('voice.noIndex')" tone="warning" />
                <Badge v-if="v.speakers.length" :value="t('voice.speakers', { n: v.speakers.length })" tone="neutral" />
                <Badge v-if="v.legacy" :value="t('voice.legacy')" tone="neutral" />
              </div>
            </AppCard>
          </TransitionGroup>
          <section class="group">
            <div class="group-head">
              <div class="group-text">
                <h2 class="type-title-small title">{{ t('models.legacyRoots') }}</h2>
                <span class="type-body-small muted">{{ t('models.legacyHint') }}</span>
              </div>
              <AppButton variant="text" :icon="icons.Plus" @click="(legacyOpen = true), (legacyPath = '')">{{ t('models.addLegacy') }}</AppButton>
            </div>
            <div v-for="r in settings.data.value?.paths.legacy_roots ?? []" :key="r.id" class="row-item">
              <div class="row-text">
                <span class="type-body-large name">{{ r.name || r.id }}</span>
                <span class="type-body-small muted name">{{ r.path }}</span>
              </div>
              <div class="row-actions">
                <AppButton variant="text" :loading="legacy.scan.isPending.value" @click="legacy.scan.mutate(r.id)">{{ t('models.scan') }}</AppButton>
                <AppButton variant="tonal" @click="legacy.importAll.mutate(r.id)">{{ t('models.importAll') }}</AppButton>
              </div>
            </div>
          </section>
          <AppDialog v-model:open="legacyOpen" :title="t('models.addLegacy')" width="small" :close-label="t('common.close')">
            <div class="row">
              <PathField v-model="legacyPath" :label="t('models.legacyPath')" />
              <AppButton variant="text" @click="legacyBrowse = true">{{ t('common.browse') }}</AppButton>
            </div>
            <template #actions>
              <AppButton variant="text" @click="legacyOpen = false">{{ t('common.cancel') }}</AppButton>
              <AppButton :disabled="!legacyPath.trim()" :loading="legacy.add.isPending.value" @click="addLegacy">{{ t('common.add') }}</AppButton>
            </template>
          </AppDialog>
          <ServerPathDialog v-model:open="legacyBrowse" folders :files="false" @select="(s) => s[0] && (legacyPath = s[0].path)" />
        </section>

        <section v-else-if="tab === 'base'" key="base" class="pane">
          <p class="type-body-medium muted">{{ t('models.base.hint') }}</p>
          <DropZone :label="t('models.base.import')" :hint="t('models.base.importHint')" accept=".pth,.zip" compact @files="onFiles" />
          <ProgressBar v-if="importer.uploading.value" :value="importer.uploading.value.progress" :label="importer.uploading.value.name" />
          <LinkImport :fetching="importer.fetching.value" @links="onLinks" />
          <section class="group">
            <h2 class="type-title-small title">{{ t('models.base.imported') }}</h2>
            <p v-if="baseModels.isSuccess.value && !importedBase.length" class="type-body-small muted">{{ t('models.base.none') }}</p>
            <div v-for="b in importedBase" :key="b.id" class="row-item">
              <div class="row-text">
                <span class="type-body-large name">{{ b.name }}</span>
                <span class="type-body-small muted name">{{ describeBase(b) }}</span>
              </div>
              <div class="row-actions">
                <IconButton :icon="icons.Pencil" :label="t('models.base.rename')" @click="renaming = { kind: 'base', id: b.id, name: b.name }" />
                <IconButton :icon="icons.Trash2" :label="t('common.delete')" @click="removing = { kind: 'base', id: b.id, name: b.name }" />
              </div>
            </div>
          </section>
          <section class="group">
            <h2 class="type-title-small title">{{ t('models.base.official') }}</h2>
            <ResourceRow
              v-for="b in officialBase"
              :key="b.id"
              :title="b.name"
              :details="describeBase(b)"
              :state="baseState(b)"
              :job-id="b.asset_id ? assetById[b.asset_id]?.job_id : null"
              deletable
              @download="b.asset_id && fetchAssets([b.asset_id])"
              @delete="removing = { kind: 'base', id: b.id, name: b.name }"
            />
          </section>
          <section v-if="communityBase.length" class="group">
            <h2 class="type-title-small title">{{ t('models.base.community') }}</h2>
            <p class="type-body-small muted">{{ t('models.base.communityHint') }}</p>
            <ResourceRow
              v-for="b in communityBase"
              :key="b.id"
              :title="b.name"
              :details="describeCommunity(b)"
              :state="baseState(b)"
              :job-id="b.asset_id ? assetById[b.asset_id]?.job_id : null"
              deletable
              @download="b.asset_id && fetchAssets([b.asset_id])"
              @delete="removing = { kind: 'asset', id: b.id, name: b.name }"
            />
          </section>
        </section>

        <section v-else-if="tab === 'separation'" key="separation" class="pane">
          <p class="type-body-medium muted">{{ t('models.separation.hint') }}</p>
          <DropZone :label="t('models.separation.import')" :hint="t('models.separation.importHint')" accept=".ckpt,.pth,.bin,.safetensors,.yaml,.yml,.zip" compact @files="onFiles" />
          <ProgressBar v-if="importer.uploading.value" :value="importer.uploading.value.progress" :label="importer.uploading.value.name" />
          <LinkImport :fetching="importer.fetching.value" @links="onLinks" />
          <section class="group">
            <h2 class="type-title-small title">{{ t('models.separation.imported') }}</h2>
            <p v-if="sepModels.isSuccess.value && !sepModels.data.value?.length" class="type-body-small muted">{{ t('models.separation.none') }}</p>
            <div v-for="m in sepModels.data.value ?? []" :key="m.id" class="row-item">
              <div class="row-text">
                <span class="type-body-large name">{{ m.name }}</span>
                <span class="type-body-small muted name">{{ describeSep(m) }}</span>
              </div>
              <div class="row-actions">
                <IconButton :icon="icons.Trash2" :label="t('common.delete')" @click="removing = { kind: 'separation', id: m.id, name: m.name }" />
              </div>
            </div>
          </section>
          <section class="group">
            <h2 class="type-title-small title">{{ t('models.separation.builtin') }}</h2>
            <ResourceRow
              v-for="a in builtinSep"
              :key="a.id"
              :title="a.title"
              :details="describeBuiltinSep(a)"
              :state="a.state"
              :job-id="a.job_id"
              deletable
              @download="fetchAssets([a.id])"
              @delete="removing = { kind: 'asset', id: a.id, name: a.title }"
            />
          </section>
        </section>

        <section v-else-if="tab === 'assets'" key="assets" class="pane">
          <ErrorNotice v-if="assets.error.value" :error="assets.error.value" />
          <section v-for="[group, list] in groups" :key="group" class="group">
            <div class="group-head">
              <h2 class="type-title-small title">{{ tOr(`assets.groups.${group}`, group) }}</h2>
              <AppButton
                v-if="list.some((a) => a.state === 'missing' || a.state === 'partial')"
                variant="text"
                :icon="icons.Download"
                @click="download.mutate({ group }, { onError: (e) => (error = e) })"
                >{{ t('assets.downloadGroup', { group: tOr(`assets.groups.${group}`, group) }) }}</AppButton
              >
            </div>
            <ResourceRow
              v-for="a in list"
              :key="a.id"
              :title="a.title"
              :details="describeAsset(a)"
              :state="a.state"
              :job-id="a.job_id"
              verifiable
              deletable
              @download="fetchAssets([a.id])"
              @verify="download.mutate({ ids: [a.id], verify: true })"
              @delete="removing = { kind: 'asset', id: a.id, name: a.title }"
            />
          </section>
          <section v-if="catalog.data.value?.length" class="group">
            <div class="group-head">
              <div class="group-text">
                <h2 class="type-title-small title">{{ t('models.demo.title') }}</h2>
                <span class="type-body-small muted">{{ t('models.demo.hint') }}</span>
              </div>
              <AppButton v-if="missingDemo.length > 1" variant="text" :icon="icons.Download" :loading="downloadVoices.isPending.value" @click="downloadDemo(null)">{{ t('models.demo.all') }}</AppButton>
            </div>
            <ResourceRow
              v-for="v in catalog.data.value"
              :key="v.id"
              :title="v.name"
              :details="describeDemo(v)"
              :state="demoState(v)"
              :job-id="v.job_id"
              :open-label="t('models.demo.inLibrary')"
              deletable
              @download="downloadDemo([v.id])"
              @open="v.installed_voice_id && router.push(`/models/${encodeURIComponent(v.installed_voice_id)}`)"
              @delete="v.installed_voice_id && (removing = { kind: 'voice', id: v.installed_voice_id, name: v.name })"
            />
          </section>
        </section>

        <section v-else key="tools" class="pane">
          <Surface :level="1" class="tool">
            <h2 class="type-title-medium title">{{ t('models.merge.title') }}</h2>
            <div class="fields">
              <SelectField v-model="mergeA" :label="t('models.merge.a')" :options="voiceOptions" />
              <SelectField v-model="mergeB" :label="t('models.merge.b')" :options="voiceOptions" />
              <ParamSlider v-model="alpha" :label="t('models.merge.alpha')" :min="0" :max="1" :step="0.05" :default-value="0.5" :reset-label="t('common.reset')" />
              <TextField v-model="mergeName" :label="t('models.merge.name')" />
            </div>
            <div class="actions">
              <AppButton :icon="icons.Merge" :disabled="!mergeA || !mergeB || !mergeName.trim()" @click="merge">{{ t('models.merge.run') }}</AppButton>
            </div>
          </Surface>
          <Surface :level="1" class="tool">
            <h2 class="type-title-medium title">{{ t('models.extract.title') }}</h2>
            <div class="fields">
              <PathField v-model="extractPath" :label="t('models.extract.checkpoint')" />
              <TextField v-model="extractName" :label="t('models.extract.name')" />
              <SelectField v-model="extractRate" :label="t('train.sampleRate')" :options="rateOptions" />
              <SelectField v-model="extractVersion" :label="t('train.version')" :options="versionOptions" />
            </div>
            <div class="actions">
              <AppButton :icon="icons.PackageOpen" :disabled="!extractPath.trim() || !extractName.trim()" @click="extract">{{ t('models.extract.run') }}</AppButton>
            </div>
          </Surface>
        </section>
      </Transition>
    </div>
    <ImportReviewDialog :plan="importer.review.value" :busy="importer.committing.value" @commit="importer.review.value && importer.commit(importer.review.value, $event)" @discard="importer.discard()" />
    <AppDialog :open="!!renaming" :title="t('models.base.rename')" width="small" :close-label="t('common.close')" @update:open="!$event && (renaming = null)">
      <TextField v-if="renaming" v-model="renaming.name" :label="t('models.review.name')" />
      <template #actions>
        <AppButton variant="text" @click="renaming = null">{{ t('common.cancel') }}</AppButton>
        <AppButton :disabled="!renaming?.name.trim()" :loading="baseM.rename.isPending.value" @click="rename">{{ t('common.save') }}</AppButton>
      </template>
    </AppDialog>
    <ConfirmDialog
      :open="!!removing"
      :title="t('common.delete')"
      :message="removing?.kind === 'asset' || removing?.kind === 'voice' || removing?.id.startsWith('official-') ? t('models.deleteDownloaded', { name: removing?.name ?? '' }) : t('models.deleteConfirm', { name: removing?.name ?? '' })"
      :confirm-label="t('common.delete')"
      :cancel-label="t('common.cancel')"
      danger
      :loading="removeBusy"
      @update:open="!$event && (removing = null)"
      @confirm="remove"
    />
  </div>
</template>

<style scoped>
.models { display: flex; flex-direction: column; gap: var(--app-space-4); padding: var(--app-space-4); }
/* The leaving pane is taken out of the flow (shared-axis-x) and slides inside this box. */
.panes { position: relative; overflow-x: clip; }
.pane { display: flex; flex-direction: column; gap: var(--app-space-5); min-width: 0; }
.filters { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-4); }
.grid { position: relative; display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: var(--app-space-3); }
.voice { display: flex; flex-direction: column; gap: var(--app-space-1); padding: var(--app-space-4); min-width: 0; }
.name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.badges { display: flex; flex-wrap: wrap; gap: var(--app-space-1); }
.title { margin: 0; }
.muted { color: var(--md-sys-color-on-surface-variant); }
/* A titled list of rows, the same on every tab (ResourceRow and .row-item share the row style). */
.group { display: flex; flex-direction: column; gap: var(--app-space-2); min-width: 0; }
.group-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--app-space-2); min-height: 40px; }
.group-text { display: flex; flex-direction: column; flex: 1 1 280px; min-width: 0; }
.row-item {
  display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2) var(--app-space-3);
  min-height: 64px; padding: var(--app-space-2) var(--app-space-2) var(--app-space-2) var(--app-space-4);
  border-radius: var(--md-sys-shape-corner-medium); background: var(--md-sys-color-surface-container); min-width: 0;
}
.row-text { display: flex; flex-direction: column; gap: 2px; flex: 1 1 260px; min-width: 0; }
.row-actions { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: var(--app-space-2); margin-left: auto; }
/* Tools are one column of sections (as Convert, Separate, Live); their fields wrap in a grid. */
.tool { display: flex; flex-direction: column; gap: var(--app-space-3); padding: var(--app-space-4); }
.fields { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(280px, 100%), 1fr)); gap: var(--app-space-3) var(--app-space-6); align-items: center; }
/* A field and its button (Browse): the button centred on the field. */
.row { display: flex; flex-wrap: wrap; gap: var(--app-space-2); align-items: center; }
.row > :first-child { flex: 1; min-width: 0; }
.actions { display: flex; justify-content: flex-end; }
</style>
