<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { useMeta, useSettings } from '@/api/queries/app';
import { checkDevices, useLiveControl, useLiveDevices, useRefreshDevices } from '@/api/queries/live';
import type { DeviceCheck, DeviceList, DeviceSelection, LiveDevices } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import { DEFAULT_VALUE, channelOptions, deviceOptions, devicesFor, driverOptions, selectionFor, selectionValue, statusMessage, variantOf } from '@/components/devices';
import { useI18n } from '@/i18n';
import { useLiveStore } from '@/stores/live';
import { AppButton, DeviceMenu, ExpansionPanel, IconButton, LevelMeter, ParamSlider, SegmentedControl, SelectField, Switch, icons } from '@/ui';

/**
 * The device picker, on the Live screen and in Settings › Audio: physical devices with
 * full names, the server's host name in the title, a live input meter, a test sound, an optional
 * monitor, the driver under Advanced, and problems shown at the device they concern.
 */
const props = withDefaults(defineProps<{ running?: boolean }>(), { running: false });
const model = defineModel<LiveDevices>({ required: true });
const emit = defineEmits<{ change: [LiveDevices] }>();
const { t, tOr } = useI18n();
const meta = useMeta();
const settings = useSettings();
const live = useLiveStore();
const control = useLiveControl();
const list = useLiveDevices();
const refresh = useRefreshDevices();
const check = ref<DeviceCheck | null>(null);
const advanced = ref(false);

const deviceList = computed(() => list.data.value as DeviceList | undefined);
const inputs = computed(() => devicesFor(deviceList.value, 'input'));
const outputs = computed(() => devicesFor(deviceList.value, 'output'));
const labels = computed(() => ({
  systemDefault: t('devices.systemDefault'),
  default: t('devices.default'),
  virtual: t('devices.virtual'),
  notConnected: t('devices.notConnected'),
  inputDevice: t('devices.inputDevice'),
  outputDevice: t('devices.outputDevice'),
}));
const inputOptions = computed(() => deviceOptions(inputs.value, model.value.input, labels.value, 'input'));
const outputOptions = computed(() => deviceOptions(outputs.value, model.value.output, labels.value, 'output'));
const monitorOptions = computed(() => [{ value: '__none__', label: t('devices.monitorNone') }, ...deviceOptions(outputs.value, model.value.monitor, labels.value, 'output')]);
const host = computed(() => list.data.value?.host || meta.data.value?.host || '');

function commit(next: LiveDevices) {
  model.value = next;
  emit('change', next);
}
function setRole(role: 'input' | 'output' | 'monitor', value: string | null) {
  const pool = role === 'input' ? inputs.value : outputs.value;
  if (role === 'monitor') {
    commit({ ...model.value, monitor: value === '__none__' || value === null ? null : selectionFor(value, pool, model.value.monitor) });
    return;
  }
  commit({ ...model.value, [role]: selectionFor(value ?? DEFAULT_VALUE, pool, model.value[role]) });
}
function patchSel(role: 'input' | 'output' | 'monitor', p: Partial<DeviceSelection>) {
  const sel = model.value[role];
  if (!sel) return;
  commit({ ...model.value, [role]: { ...sel, ...p } });
}

const roleValue = (role: 'input' | 'output' | 'monitor') => (role === 'monitor' && !model.value.monitor ? '__none__' : selectionValue(model.value[role]));
const resolved = (role: string) => check.value?.resolved.find((r) => r.role === role);
const problems = (role: string) => check.value?.problems.filter((p) => p.role === role) ?? [];
const status = (role: 'input' | 'output' | 'monitor') =>
  problems(role).length
    ? null
    : statusMessage(resolved(role), (s, name) => tOr(`devices.statuses.${s}`, s, { name: model.value[role]?.name ?? name }));

// Channels and rate of the concrete device in this enumeration.
const variant = (role: 'input' | 'output') => variantOf(role === 'input' ? inputs.value : outputs.value, model.value[role]);
const channelChoices = (role: 'input' | 'output') =>
  channelOptions(variant(role)?.channels ?? 2, role, { automatic: (l) => t('devices.channelsDefault', { list: l }), channel: (n) => t('devices.channel', { n }), mix: (l) => t('devices.mix', { list: l }) });
const channelValue = (role: 'input' | 'output') => (model.value[role].channels ?? []).join(',');
const setChannels = (role: 'input' | 'output', v: string | null) => patchSel(role, { channels: v ? v.split(',').map(Number) : null });
const rateChoices = computed(() => [{ value: '', label: t('devices.automatic') }, ...(variant('output')?.supported_rates ?? [44100, 48000]).map((r) => ({ value: String(r), label: `${r / 1000} kHz` }))]);
const drivers = (role: 'input' | 'output') => driverOptions(role === 'input' ? inputs.value : outputs.value, model.value[role], (name) => t('devices.recommended', { name }));
function setDriver(role: 'input' | 'output', id: string | null) {
  const v = (role === 'input' ? inputs.value : outputs.value).flatMap((d) => d.variants).find((x) => x.id === id);
  if (v) patchSel(role, { device_id: v.id, host_api: v.host_api });
}
const isWasapi = (role: 'input' | 'output') => variant(role)?.host_api === 'wasapi';
const monitorSource = computed({ get: () => model.value.monitor_source, set: (v) => commit({ ...model.value, monitor_source: v }) });
const sourceOptions = computed(() => (['converted', 'input', 'both'] as const).map((v) => ({ value: v, label: t(`devices.monitorSource.${v}`) })));

function runAction(role: 'input' | 'output' | 'monitor', action: string | null | undefined) {
  if (action === 'use_48k') patchSel(role, { sample_rate: 48000 });
  else if (action === 'disable_exclusive') patchSel(role, { exclusive: false });
}

const statusLine = computed(() => {
  const c = check.value;
  if (!c?.sample_rate) return '';
  return t('devices.status', { rate: c.sample_rate / 1000, topology: c.topology ? t(`devices.topology.${c.topology}`) : '—', ms: Math.round(c.est_latency_ms ?? 0) });
});

// Validate the exact configuration on every change, so a refused format shows before Start.
let timer: ReturnType<typeof setTimeout> | undefined;
watch(
  () => [model.value, list.data.value],
  () => {
    clearTimeout(timer);
    if (!list.data.value) return;
    timer = setTimeout(async () => {
      try {
        check.value = await checkDevices(model.value);
      } catch {
        check.value = null;
      }
    }, 250);
  },
  { deep: true, immediate: true },
);

// The input meter runs while the panel is visible and Live is stopped (live.show_meters).
const meterWanted = computed(() => !props.running && (settings.data.value?.live.show_meters ?? true) && !!list.data.value);
watch(meterWanted, (on, before) => {
  if (on !== before) control.meter.mutate(on);
});
// Re-enumerate every 5 s while visible and stopped; the server emits devices_changed only on a difference.
let poll: ReturnType<typeof setInterval> | undefined;
onMounted(() => {
  if (meterWanted.value) control.meter.mutate(true);
  poll = setInterval(() => {
    if (!props.running && document.visibilityState === 'visible') refresh.mutate();
  }, 5000);
});
onBeforeUnmount(() => {
  clearInterval(poll);
  clearTimeout(timer);
  if (meterWanted.value) control.meter.mutate(false);
});
const passthrough = computed({ get: () => live.state.passthrough, set: (on) => control.passthrough.mutate(on) });
const inLevels = computed(() => (live.stats ? { rms: live.stats.input_rms_db, peak: live.stats.input_peak_db } : { rms: null, peak: null }));
</script>

<template>
  <section class="devices">
    <header class="head">
      <h2 class="type-title-medium title">{{ host ? t('devices.title', { host }) : t('devices.titleUnknown') }}</h2>
      <IconButton :icon="icons.RefreshCw" :spin="refresh.isPending.value" :label="t('devices.refresh')" @click="refresh.mutate()" />
    </header>
    <p v-if="meta.data.value && !meta.data.value.local" class="type-body-small muted">{{ t('devices.remoteNote') }}</p>
    <ErrorNotice v-if="list.error.value" :error="list.error.value" />
    <template v-else>
      <!-- One group per role; on a wide page they stand side by side. -->
      <div class="roles">
        <div class="group">
          <div class="role">
            <DeviceMenu :label="t('devices.input')" :model-value="roleValue('input')" :options="inputOptions" @update:model-value="setRole('input', $event)" />
            <LevelMeter :label="t('devices.meter')" :rms-db="inLevels.rms" :peak-db="inLevels.peak" />
          </div>
          <p v-if="status('input')" class="type-body-small warn">{{ status('input') }}</p>
          <div v-for="p in problems('input')" :key="p.reason" class="problem type-body-small">
            <span>{{ tOr(`devices.reasons.${p.reason}`, p.reason) }}: {{ p.message }}<template v-if="p.action === 'grant_permission' || p.action === 'choose'"> · {{ tOr(`devices.actions.${p.action}`, p.action) }}</template></span>
            <AppButton v-if="p.action === 'use_48k' || p.action === 'disable_exclusive'" variant="text" @click="runAction('input', p.action)">{{ tOr(`devices.actions.${p.action}`, p.action) }}</AppButton>
          </div>
        </div>
        <div class="group">
          <div class="role">
            <DeviceMenu :label="t('devices.output')" :model-value="roleValue('output')" :options="outputOptions" @update:model-value="setRole('output', $event)" />
            <AppButton variant="tonal" :icon="icons.Volume2" @click="control.testTone.mutate({ role: 'output', device: model.output })">{{ t('devices.test') }}</AppButton>
          </div>
          <p v-if="status('output')" class="type-body-small warn">{{ status('output') }}</p>
          <div v-for="p in problems('output')" :key="p.reason" class="problem type-body-small">
            <span>{{ tOr(`devices.reasons.${p.reason}`, p.reason) }}: {{ p.message }}<template v-if="p.action === 'grant_permission' || p.action === 'choose'"> · {{ tOr(`devices.actions.${p.action}`, p.action) }}</template></span>
            <AppButton v-if="p.action === 'use_48k' || p.action === 'disable_exclusive'" variant="text" @click="runAction('output', p.action)">{{ tOr(`devices.actions.${p.action}`, p.action) }}</AppButton>
          </div>
        </div>
        <div class="group">
          <div class="role">
            <DeviceMenu :label="t('devices.monitor')" :model-value="roleValue('monitor')" :options="monitorOptions" @update:model-value="setRole('monitor', $event)" />
            <AppButton v-if="model.monitor" variant="tonal" :icon="icons.Headphones" @click="control.testTone.mutate({ role: 'monitor', device: model.monitor })">{{ t('devices.test') }}</AppButton>
          </div>
          <div v-if="model.monitor" class="monitor">
            <SegmentedControl v-model="monitorSource" :options="sourceOptions" />
            <ParamSlider
              :model-value="model.monitor_gain_db"
              :label="t('devices.monitorGain')"
              :min="-60"
              :max="12"
              unit="dB"
              :default-value="0"
              :reset-label="t('common.reset')"
              @update:model-value="model = { ...model, monitor_gain_db: $event }"
              @commit="commit({ ...model, monitor_gain_db: $event })"
            />
          </div>
          <p v-if="status('monitor')" class="type-body-small warn">{{ status('monitor') }}</p>
        </div>
      </div>
      <Switch v-model="passthrough" :label="t('devices.hearYourself')" :supporting-text="t('devices.hearYourselfHint')" />

      <ExpansionPanel v-model:open="advanced" :label="t('devices.advanced')" :icon="icons.SlidersHorizontal">
        <div class="advanced">
          <SelectField v-if="drivers('input').length > 1" :label="`${t('devices.input')} · ${t('devices.driver')}`" :options="drivers('input')" :model-value="model.input.device_id" @update:model-value="setDriver('input', $event)" />
          <SelectField :label="`${t('devices.input')} · ${t('devices.channels')}`" :options="channelChoices('input')" :model-value="channelValue('input')" @update:model-value="setChannels('input', $event)" />
          <SelectField v-if="drivers('output').length > 1" :label="`${t('devices.output')} · ${t('devices.driver')}`" :options="drivers('output')" :model-value="model.output.device_id" @update:model-value="setDriver('output', $event)" />
          <SelectField :label="`${t('devices.output')} · ${t('devices.channels')}`" :options="channelChoices('output')" :model-value="channelValue('output')" @update:model-value="setChannels('output', $event)" />
          <SelectField :label="t('devices.sampleRate')" :options="rateChoices" :model-value="model.output.sample_rate ? String(model.output.sample_rate) : ''" @update:model-value="patchSel('output', { sample_rate: $event ? Number($event) : null })" />
          <Switch v-if="isWasapi('output')" :model-value="model.output.exclusive" :label="t('devices.exclusive')" @update:model-value="patchSel('output', { exclusive: $event })" />
          <ParamSlider
            :model-value="model.output_gain_db"
            :label="t('devices.outputGain')"
            :min="-60"
            :max="12"
            unit="dB"
            :default-value="0"
            :reset-label="t('common.reset')"
            @update:model-value="model = { ...model, output_gain_db: $event }"
            @commit="commit({ ...model, output_gain_db: $event })"
          />
        </div>
      </ExpansionPanel>
      <p v-if="statusLine" class="type-body-small muted status">{{ statusLine }}</p>
    </template>
  </section>
</template>

<style scoped>
.devices { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.head { display: flex; align-items: center; gap: var(--app-space-2); }
.title { flex: 1; margin: 0; min-width: 0; overflow-wrap: anywhere; }
.roles { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(360px, 100%), 1fr)); gap: var(--app-space-3) var(--app-space-6); align-items: start; }
.group { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.role { display: grid; grid-template-columns: minmax(0, 1fr) minmax(120px, 220px); gap: var(--app-space-3); align-items: center; }
.monitor { display: flex; flex-direction: column; gap: var(--app-space-2); padding-left: var(--app-space-3); }
.advanced { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(240px, 100%), 1fr)); gap: var(--app-space-3) var(--app-space-6); align-items: center; padding: var(--app-space-2) 0; }
.warn { margin: 0; color: var(--md-sys-color-tertiary); }
.problem { display: flex; align-items: center; flex-wrap: wrap; gap: var(--app-space-2); color: var(--md-sys-color-error); }
.status { margin: 0; font-variant-numeric: tabular-nums; }
@media (max-width: 599px) {
  .role { grid-template-columns: 1fr; }
}
</style>
