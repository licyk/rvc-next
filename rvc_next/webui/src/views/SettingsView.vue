<script setup lang="ts">
import { useAssetRepositories } from '@/api/queries/assets';
import { computed, ref, watch } from 'vue';
import { useMeta, useSettings, useUpdateSettings, useVersion } from '@/api/queries/app';
import { useComputeDevices } from '@/api/queries/compute';
import { useSeparationPresets } from '@/api/queries/separate';
import type { LiveDevices, SettingsView } from '@/api/types';
import DevicePanel from '@/components/DevicePanel.vue';
import ErrorNotice from '@/components/ErrorNotice.vue';
import { useI18n } from '@/i18n';
import { formatMiB } from '@/format';
import { usePreferencesStore } from '@/stores/preferences';
import { SOURCE_COLORS } from '@/theme/scheme';
import { AppButton, Badge, DataList, PathField, SegmentedControl, SelectField, Surface, Switch, Tabs, TextField, TRANSITIONS, icons, useAxisDirection, useSnackbar } from '@/ui';

type Section = 'general' | 'compute' | 'audio' | 'storage' | 'downloads' | 'training' | 'appearance' | 'about';
const { t, localeOptions } = useI18n();
const prefs = usePreferencesStore();
const settings = useSettings();
const update = useUpdateSettings();
const meta = useMeta();
const version = useVersion();
const snackbar = useSnackbar();
const section = ref<Section>('general');
const SECTIONS = ['general', 'compute', 'audio', 'storage', 'downloads', 'training', 'appearance', 'about'] as const;
const sections = computed(() => SECTIONS.map((v) => ({ value: v, label: t(`settings.sections.${v}`) })));
// Sections slide the way the tabs moved (shared-axis-x), as Models' tabs and Hanaikada's tabs.
const axisDir = useAxisDirection(section, () => SECTIONS);
const error = ref<unknown>(null);
const s = computed<SettingsView | undefined>(() => settings.data.value);

function patch(p: Record<string, unknown>) {
  error.value = null;
  update.mutate(p, { onSuccess: () => snackbar.show(t('settings.saved')), onError: (e) => (error.value = e) });
}
// A value pinned by the application embedding rvc-next, or by an environment variable, cannot be changed here.
const pinnedBy = (key: string): 'host' | 'env' | null =>
  s.value?.pinned.includes(key) ? 'host' : s.value?.env_overrides.some((e) => e === `RVC_NEXT_${key.toUpperCase().replace('.', '__')}`) ? 'env' : null;

// General
const locale = computed({ get: () => prefs.prefs.locale, set: (v) => (prefs.prefs.locale = v as 'auto') });
const token = ref('');
// Compute
const devicesQ = useComputeDevices();
const deviceOptions = computed(() => [{ value: 'auto', label: t('settings.precisions.auto') }, ...(devicesQ.data.value?.devices ?? []).map((d) => ({ value: d.id, label: `${d.name}${d.memory_mb ? ` · ${formatMiB(d.memory_mb)}` : ''}${d.eligible ? '' : ' ✕'}` }))]);
const precisionOptions = computed(() => (['auto', 'fp32', 'fp16'] as const).map((v) => ({ value: v, label: t(`settings.precisions.${v}`) })));
// Audio
const liveDevices = ref<LiveDevices | null>(null);
watch(s, (v) => v && !liveDevices.value && (liveDevices.value = JSON.parse(JSON.stringify(v.live.devices))), { immediate: true });
// Storage
const folders = ref<Record<string, string>>({});
const browseRoots = ref('');
watch(
  s,
  (v) => {
    if (!v) return;
    folders.value = { models_dir: v.paths.models_dir, experiments_dir: v.paths.experiments_dir, outputs_dir: v.paths.outputs_dir, assets_dir: v.paths.assets_dir };
    browseRoots.value = v.paths.browse_roots.join('\n');
  },
  { immediate: true },
);
function saveFolders() {
  patch({ paths: { ...folders.value, browse_roots: browseRoots.value.split('\n').map((x) => x.trim()).filter(Boolean) } });
}
// Downloads
const repositories = useAssetRepositories();
const repositoryOptions = computed(() => (repositories.data.value ?? []).map((r) => ({ value: r.id, label: `${r.repo} · ${t(`settings.repositories.${r.id}`)}` })));
const sourceOptions = computed(() => (['huggingface', 'hf-mirror', 'custom'] as const).map((v) => ({ value: v, label: t(`settings.sources.${v}`) })));
const endpoint = ref('');
watch(s, (v) => v && (endpoint.value = v.downloads.endpoint), { immediate: true });
// Training defaults
const sepPresets = useSeparationPresets();
// Appearance
const themes = computed(() => (['light', 'dark', 'system'] as const).map((v) => ({ value: v, label: t(`settings.themes.${v}`) })));
const motions = computed(() => (['system', 'reduced'] as const).map((v) => ({ value: v, label: t(`settings.motions.${v}`) })));
const contrastOptions = computed(() => [
  { value: '0', label: t('settings.contrastLevels.standard') },
  { value: '0.5', label: t('settings.contrastLevels.medium') },
  { value: '1', label: t('settings.contrastLevels.high') },
]);
const contrast = computed({ get: () => String(prefs.prefs.contrast), set: (v) => (prefs.prefs.contrast = Number(v)) });
const COLORS = SOURCE_COLORS;
const theme = computed({ get: () => prefs.prefs.theme, set: (v) => (prefs.prefs.theme = v) });
const motion = computed({ get: () => prefs.prefs.motion, set: (v) => (prefs.prefs.motion = v) });
const previewShown = ref(true);
</script>

<template>
  <div class="settings">
    <Tabs v-model="section" :tabs="sections" />
    <ErrorNotice v-if="error" :error="error" />
    <ErrorNotice v-if="settings.error.value" :error="settings.error.value" />
    <div v-if="s" class="panes" :style="{ '--axis-dir': axisDir }">
      <Transition :name="TRANSITIONS.sharedAxisX">
        <Surface :key="section" :level="0" class="panel">
          <template v-if="section === 'general'">
            <SelectField v-model="locale" :label="t('settings.language')" :options="localeOptions" />
            <Switch :model-value="s.server.open_browser" :label="t('settings.openBrowser')" @update:model-value="patch({ server: { open_browser: $event } })" />
            <div class="row">
              <TextField :model-value="s.server.host" :label="t('settings.address')" :supporting-text="t('settings.restartNote')" @change="patch({ server: { host: $event } })" />
              <TextField type="number" :model-value="s.server.port" :label="t('settings.port')" @change="patch({ server: { port: Number($event) } })" />
            </div>
            <div class="row center">
              <Badge :value="s.server.access_token_configured ? t('settings.tokenSet') : t('settings.tokenNone')" :tone="s.server.access_token_configured ? 'primary' : 'neutral'" />
              <TextField v-model="token" class="grow" type="password" :label="t('settings.accessToken')" autocomplete="new-password" />
              <AppButton variant="tonal" :disabled="!token" @click="patch({ server: { access_token: token } }), (token = '')">{{ t('settings.setToken') }}</AppButton>
              <AppButton v-if="s.server.access_token_configured" variant="text" @click="patch({ server: { access_token: null } })">{{ t('settings.clearToken') }}</AppButton>
            </div>
          </template>
          <template v-else-if="section === 'compute'">
            <SelectField :model-value="s.compute.device" :label="t('settings.device')" :options="deviceOptions" @update:model-value="patch({ compute: { device: $event } })" />
            <SelectField :model-value="s.compute.precision" :label="t('settings.precision')" :options="precisionOptions" @update:model-value="patch({ compute: { precision: $event } })" />
            <Switch :model-value="s.compute.cuda_graph_offline" :label="t('settings.cudaGraphOffline')" @update:model-value="patch({ compute: { cuda_graph_offline: $event } })" />
            <Switch :model-value="s.compute.cuda_graph_live" :label="t('settings.cudaGraphLive')" @update:model-value="patch({ compute: { cuda_graph_live: $event } })" />
            <div class="row">
              <TextField type="number" :min="0" :model-value="s.compute.unload_after_minutes" :label="t('settings.unloadAfter')" @change="patch({ compute: { unload_after_minutes: Number($event) } })" />
              <TextField type="number" :min="1" :model-value="s.compute.gpu_jobs" :label="t('settings.gpuJobs')" @change="patch({ compute: { gpu_jobs: Number($event) } })" />
            </div>
            <DataList :rows="(devicesQ.data.value?.devices ?? []).map((d) => ({ label: d.id, value: `${d.name} · ${d.precision}${d.reason ? ' · ' + d.reason : ''}` }))" />
          </template>
          <template v-else-if="section === 'audio'">
            <DevicePanel v-if="liveDevices" v-model="liveDevices" @change="patch({ live: { devices: $event } })" />
            <Switch :model-value="s.live.enable_asio" :label="t('settings.asio')" @update:model-value="patch({ live: { enable_asio: $event } })" />
            <Switch :model-value="s.live.auto_reconnect" :label="t('settings.autoReconnect')" @update:model-value="patch({ live: { auto_reconnect: $event } })" />
            <Switch :model-value="s.live.show_meters" :label="t('settings.showMeters')" @update:model-value="patch({ live: { show_meters: $event } })" />
            <Switch
              :model-value="s.live.show_all_devices"
              :label="t('settings.showAllDevices')"
              :supporting-text="t('settings.showAllDevicesHint')"
              @update:model-value="patch({ live: { show_all_devices: $event } })"
            />
          </template>
          <template v-else-if="section === 'storage'">
            <h3 class="type-title-small sub">{{ t('settings.folders') }}</h3>
            <PathField
              v-for="key in ['models_dir', 'experiments_dir', 'outputs_dir', 'assets_dir']"
              :key="key"
              v-model="folders[key]"
              :label="t(`settings.${key.replace('_dir', 'Dir')}`)"
              :supporting-text="pinnedBy(`paths.${key}`) ? t(pinnedBy(`paths.${key}`) === 'host' ? 'settings.pinnedByHost' : 'settings.pinned') : t('settings.defaultFolder', { path: (s.resolved_paths as Record<string, unknown>)[key] as string })"
            />
            <label class="type-body-medium">{{ t('settings.browseRoots') }}</label>
            <textarea v-model="browseRoots" class="area type-body-medium" rows="4" :placeholder="s.resolved_paths.browse_roots.join('\n')" />
            <span class="type-body-small muted">{{ t('settings.browseRootsHint') }}</span>
            <TextField type="number" :min="0" :model-value="s.convert.keep_outputs_days" :label="t('settings.keepOutputs')" @change="patch({ convert: { keep_outputs_days: Number($event) } })" />
            <div class="actions"><AppButton :icon="icons.Save" @click="saveFolders">{{ t('common.save') }}</AppButton></div>
          </template>
          <template v-else-if="section === 'downloads'">
            <SelectField :model-value="s.downloads.repository" :label="t('settings.repository')" :options="repositoryOptions" @update:model-value="patch({ downloads: { repository: $event } })" />
            <span class="type-body-small muted">{{ t('settings.repositoryHint') }}</span>
            <SelectField :model-value="s.downloads.source" :label="t('settings.source')" :options="sourceOptions" @update:model-value="patch({ downloads: { source: $event } })" />
            <TextField v-if="s.downloads.source === 'custom'" v-model="endpoint" :label="t('settings.endpoint')" @change="patch({ downloads: { endpoint } })" />
            <Switch :model-value="s.downloads.verify_checksums" :label="t('settings.verifyChecksums')" @update:model-value="patch({ downloads: { verify_checksums: $event } })" />
          </template>
          <template v-else-if="section === 'training'">
            <div class="row">
              <SelectField :model-value="s.training.sample_rate" :label="t('train.sampleRate')" :options="['32k', '40k', '48k'].map((v) => ({ value: v, label: v }))" @update:model-value="patch({ training: { sample_rate: $event } })" />
              <SelectField :model-value="s.training.version" :label="t('train.version')" :options="['v1', 'v2'].map((v) => ({ value: v, label: v }))" @update:model-value="patch({ training: { version: $event } })" />
              <TextField type="number" :min="1" :model-value="s.training.epochs" :label="t('train.epochs')" @change="patch({ training: { epochs: Number($event) } })" />
              <TextField type="number" :min="1" :model-value="s.training.save_every" :label="t('train.saveEvery')" @change="patch({ training: { save_every: Number($event) } })" />
            </div>
            <Switch :model-value="s.training.pitch_guidance" :label="t('train.pitchGuidance')" @update:model-value="patch({ training: { pitch_guidance: $event } })" />
            <Switch :model-value="s.training.cache_in_gpu" :label="t('train.cacheInGpu')" @update:model-value="patch({ training: { cache_in_gpu: $event } })" />
            <SelectField :model-value="s.separation.default_preset" :label="t('settings.defaultPreset')" :options="(sepPresets.data.value ?? []).map((p) => ({ value: p.id, label: p.title }))" @update:model-value="patch({ separation: { default_preset: $event } })" />
            <SelectField :model-value="s.convert.output_format" :label="t('settings.outputFormat')" :options="['wav', 'flac', 'mp3', 'm4a'].map((v) => ({ value: v, label: v.toUpperCase() }))" @update:model-value="patch({ convert: { output_format: $event } })" />
          </template>
          <template v-else-if="section === 'appearance'">
            <!-- As Hanaikada's Appearance: the name on the left, the control at its own width on the right. -->
            <div class="field-row">
              <span class="type-body-large">{{ t('settings.theme') }}</span>
              <SegmentedControl v-model="theme" :options="themes" />
            </div>
            <div class="field-row">
              <span class="type-body-large">{{ t('settings.color') }}</span>
              <div class="swatches">
                <button v-for="c in COLORS" :key="c" type="button" class="swatch" :class="{ on: prefs.prefs.sourceColor === c }" :style="{ background: c }" :aria-label="c" @click="prefs.prefs.sourceColor = c" />
              </div>
            </div>
            <div class="field-row">
              <span class="type-body-large">{{ t('settings.contrast') }}</span>
              <SelectField v-model="contrast" :options="contrastOptions" />
            </div>
            <div class="field-row">
              <span class="type-body-large">{{ t('settings.motion') }}</span>
              <SegmentedControl v-model="motion" :options="motions" />
            </div>
            <AppButton variant="text" class="start" @click="previewShown = !previewShown">{{ t('settings.motionPreview') }}</AppButton>
            <Transition name="container"><div v-if="previewShown" class="preview type-body-medium">{{ t('settings.motionPreview') }}</div></Transition>
          </template>
          <template v-else>
            <DataList
              :rows="[
                { label: 'rvc-next', value: version.data.value?.version ?? '—' },
                { label: t('settings.dataDir'), value: meta.data.value?.data_dir ?? '—', mono: true },
                { label: t('settings.settingsFile'), value: s.settings_file, mono: true },
                { label: t('settings.device'), value: meta.data.value?.compute ?? '—' },
                { label: 'torch', value: devicesQ.data.value ? `${devicesQ.data.value.torch_version} (${devicesQ.data.value.backend ?? 'CPU'})` : '—' },
              ]"
            />
            <p class="type-body-small muted">RVC (MIT) · PyMSS (MIT) · rvc-next (GPL-3.0)</p>
          </template>
        </Surface>
      </Transition>
    </div>
  </div>
</template>

<style scoped>
.settings { display: flex; flex-direction: column; gap: var(--app-space-4); padding: var(--app-space-4); min-width: 0; }
/* The leaving section is taken out of the flow (shared-axis-x) and slides inside this box. */
.panes { position: relative; overflow-x: clip; }
.panel { display: flex; flex-direction: column; gap: var(--app-space-3); padding: var(--app-space-5); min-width: 0; }
.row { display: flex; flex-wrap: wrap; gap: var(--app-space-3); align-items: flex-start; }
.row:not(.center) > * { flex: 1 1 280px; min-width: 0; }
.row.center { align-items: center; }
.row.center > .grow { flex: 1 1 280px; min-width: 0; }
.sub { margin: 0; }
.actions { display: flex; justify-content: flex-end; }
.area { width: 100%; padding: var(--app-space-2) var(--app-space-3); border: 1px solid var(--md-sys-color-outline); border-radius: var(--md-sys-shape-corner-extra-small); background: transparent; color: var(--md-sys-color-on-surface); font: inherit; resize: vertical; }
.field-row { display: flex; align-items: center; justify-content: space-between; gap: var(--app-space-4); min-height: 56px; flex-wrap: wrap; }
.start { align-self: flex-start; }
.swatches { display: flex; flex-wrap: wrap; gap: var(--app-space-2); }
.swatch { width: 36px; height: 36px; border-radius: 50%; border: 2px solid transparent; cursor: pointer; }
.swatch.on { border-color: var(--md-sys-color-on-surface); }
.preview { padding: var(--app-space-4); border-radius: var(--md-sys-shape-corner-large); background: var(--md-sys-color-primary-container); color: var(--md-sys-color-on-primary-container); }
</style>
