<script setup lang="ts">
import { computed, onActivated, onBeforeUnmount, onDeactivated, onMounted, ref, watch } from 'vue';
import { useMeta, useSettings } from '@/api/queries/app';
import { checkDevices, useLiveControl, useLiveDevices, useMeasureLatency, useMeterClaim, useRefreshDevices } from '@/api/queries/live';
import type { AudioDevice, DeviceCheck, DeviceList, DeviceSelection, LiveDevices, PhysicalDevice } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import { DEFAULT_VALUE, channelOptions, deviceOptions, devicesFor, driverOptions, selectionFor, selectionValue, statusMessage, variantOf } from '@/components/devices';
import { useI18n } from '@/i18n';
import { useLiveStore } from '@/stores/live';
import { formatRate } from '@/format';
import { AppButton, ExpansionPanel, IconButton, LevelMeter, ParamSlider, PickerMenu, SegmentedControl, SelectField, Switch, icons, type PickerOption } from '@/ui';

/**
 * The device picker, on the Live screen and in Settings › Audio: physical devices with
 * full names, the server's host name in the title, a level meter under each device (the input's
 * live whenever the panel is shown, running or not), a test sound, an optional monitor, the driver
 * under Advanced, problems shown at the device they concern, and a loopback measurement of the
 * real latency.
 */
const props = withDefaults(defineProps<{ running?: boolean }>(), { running: false });
/** A device list this long gets a search field (Windows lists each device under several drivers' names). */
const SEARCH_FROM = 9;
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
// The first enumeration is a subprocess on the server (most of a second, longer on Windows with many drivers).
const loading = computed(() => list.isLoading.value);
const hostApiName = (id: string) => deviceList.value?.host_apis.find((h) => h.id === id)?.name ?? id;
const inputs = computed(() => devicesFor(deviceList.value, 'input'));
const outputs = computed(() => devicesFor(deviceList.value, 'output'));
const labels = computed(() => ({
  systemDefault: t('devices.systemDefault'),
  default: t('devices.default'),
  virtual: t('devices.virtual'),
  notConnected: t('devices.notConnected'),
  loopback: t('devices.loopback'),
  systemDefaultHint: t('devices.systemDefaultHint'),
  describe: (d: PhysicalDevice, v: AudioDevice) =>
    [d.is_loopback ? t('devices.loopbackDriver') : hostApiName(v.host_api), t('devices.channelCount', { n: v.channels }), formatRate(v.default_sample_rate)].join(' · '),
}));
/** The picker props every device role shares: the loading state, and a search once the list is long. */
const pickerProps = (options: PickerOption[]) => ({
  options,
  loading: loading.value,
  placeholder: t('devices.loading'),
  searchLabel: t('devices.search'),
  searchFrom: SEARCH_FROM,
  noMatches: t('common.noMatches'),
});
const inputOptions = computed(() => deviceOptions(inputs.value, model.value.input, labels.value));
const outputOptions = computed(() => deviceOptions(outputs.value, model.value.output, labels.value));
const monitorOptions = computed(() => [{ value: '__none__', label: t('devices.monitorNone') }, ...deviceOptions(outputs.value, model.value.monitor, labels.value)]);
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

// The input meter: while Live runs, the session's own levels; otherwise the server opens the input
// alone (in the background, retrying a device that is not there) for as long as a panel is shown
// (live.show_meters).
useMeterClaim(() => settings.data.value?.live.show_meters ?? true);
// Re-enumerate every 5 s while shown and stopped; the server emits devices_changed only on a difference.
// Never a second request while one is out (an enumeration can outlast the interval on Windows), and
// none from a view kept alive in the background.
let poll: ReturnType<typeof setInterval> | undefined;
function startPoll() {
  clearInterval(poll);
  poll = setInterval(() => {
    if (!props.running && document.visibilityState === 'visible' && !refresh.isPending.value && !list.isFetching.value) refresh.mutate();
  }, 5000);
}
onMounted(startPoll);
onActivated(startPoll);
onDeactivated(() => clearInterval(poll));
onBeforeUnmount(() => {
  clearInterval(poll);
  clearTimeout(timer);
});
// Measured latency: bursts out of the output, found again in the input. The state keeps the last
// measurement while it still fits the devices and the block.
const measure = useMeasureLatency();
const measurement = computed(() => live.state.latency_test);
function runMeasure() {
  measure.mutate({ devices: model.value });
}
const measuredLine = computed(() => {
  const m = measurement.value;
  if (!m) return '';
  if (!m.ok) return tOr(`devices.measureFailed.${m.reason}`, m.reason ?? '');
  return t('devices.measured', { ms: Math.round(m.latency_ms ?? 0), rt: Math.round(m.round_trip_ms ?? 0), engine: Math.round(m.engine_ms), est: Math.round(m.estimated_ms) });
});
const passthrough = computed({ get: () => live.state.passthrough, set: (on) => control.passthrough.mutate(on) });
// What reaches each device: the microphone; the converted voice (or Hear yourself) on the output and
// the monitor. Silent ("—") while nothing plays there.
const levels = computed(() => {
  const s = live.stats;
  return {
    input: { rms: s?.input_rms_db ?? null, peak: s?.input_peak_db ?? null },
    output: { rms: s?.output_rms_db ?? null, peak: s?.output_peak_db ?? null },
    monitor: { rms: s?.monitor_rms_db ?? null, peak: s?.monitor_peak_db ?? null },
  };
});
</script>

<template>
  <section class="devices">
    <header class="head">
      <h2 class="type-title-medium title">{{ host ? t('devices.title', { host }) : t('devices.titleUnknown') }}</h2>
      <IconButton :icon="icons.RefreshCw" :spin="refresh.isPending.value || loading" :disabled="loading" :label="t('devices.refresh')" @click="refresh.mutate()" />
    </header>
    <p v-if="meta.data.value && !meta.data.value.local" class="type-body-small muted">{{ t('devices.remoteNote') }}</p>
    <ErrorNotice v-if="list.error.value" :error="list.error.value" />
    <template v-else>
      <!-- What the enumeration could not do (no ASIO driver, no way to record an output here), as the server words it. -->
      <p v-for="e in deviceList?.errors ?? []" :key="e" class="type-body-small warn">{{ e }}</p>
      <p v-if="loading" class="type-body-small muted" role="status">{{ t('devices.loadingHint') }}</p>
      <!-- One role per row: device names are long, so each picker takes the whole row, with its meter or test below. -->
      <div class="roles">
        <div class="group">
          <PickerMenu
            v-bind="pickerProps(inputOptions)"
            :label="t('devices.input')"
            :icon="icons.Mic"
            :model-value="roleValue('input')"
            @update:model-value="setRole('input', $event)"
          />
          <LevelMeter :label="t('devices.meter')" :rms-db="levels.input.rms" :peak-db="levels.input.peak" />
          <p v-if="status('input')" class="type-body-small warn">{{ status('input') }}</p>
          <div v-for="p in problems('input')" :key="p.reason" class="problem type-body-small">
            <span>{{ tOr(`devices.reasons.${p.reason}`, p.reason) }}: {{ p.message }}<template v-if="p.action === 'grant_permission' || p.action === 'choose'"> · {{ tOr(`devices.actions.${p.action}`, p.action) }}</template></span>
            <AppButton v-if="p.action === 'use_48k' || p.action === 'disable_exclusive'" variant="text" @click="runAction('input', p.action)">{{ tOr(`devices.actions.${p.action}`, p.action) }}</AppButton>
          </div>
        </div>
        <div class="group">
          <PickerMenu
            v-bind="pickerProps(outputOptions)"
            :label="t('devices.output')"
            :icon="icons.Speaker"
            :model-value="roleValue('output')"
            @update:model-value="setRole('output', $event)"
          />
          <div class="below">
            <LevelMeter class="level" :label="t('devices.outputMeter')" :rms-db="levels.output.rms" :peak-db="levels.output.peak" />
            <AppButton variant="tonal" :icon="icons.Volume2" :disabled="loading" @click="control.testTone.mutate({ role: 'output', device: model.output })">{{ t('devices.test') }}</AppButton>
          </div>
          <p v-if="status('output')" class="type-body-small warn">{{ status('output') }}</p>
          <div v-for="p in problems('output')" :key="p.reason" class="problem type-body-small">
            <span>{{ tOr(`devices.reasons.${p.reason}`, p.reason) }}: {{ p.message }}<template v-if="p.action === 'grant_permission' || p.action === 'choose'"> · {{ tOr(`devices.actions.${p.action}`, p.action) }}</template></span>
            <AppButton v-if="p.action === 'use_48k' || p.action === 'disable_exclusive'" variant="text" @click="runAction('output', p.action)">{{ tOr(`devices.actions.${p.action}`, p.action) }}</AppButton>
          </div>
        </div>
        <div class="group">
          <PickerMenu
            v-bind="pickerProps(monitorOptions)"
            :label="t('devices.monitor')"
            :icon="icons.Headphones"
            :model-value="roleValue('monitor')"
            @update:model-value="setRole('monitor', $event)"
          />
          <div v-if="model.monitor" class="below">
            <LevelMeter class="level" :label="t('devices.monitorMeter')" :rms-db="levels.monitor.rms" :peak-db="levels.monitor.peak" />
            <AppButton variant="tonal" :icon="icons.Headphones" :disabled="loading" @click="control.testTone.mutate({ role: 'monitor', device: model.monitor })">{{ t('devices.test') }}</AppButton>
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
          <ParamSlider
            :model-value="model.input_gain_db"
            :label="t('devices.inputGain')"
            :min="-24"
            :max="24"
            unit="dB"
            :default-value="0"
            :reset-label="t('common.reset')"
            @update:model-value="model = { ...model, input_gain_db: $event }"
            @commit="commit({ ...model, input_gain_db: $event })"
          />
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
      <div class="measure">
        <AppButton variant="tonal" :icon="icons.Timer" :loading="measure.isPending.value" :disabled="loading || running || live.state.passthrough" @click="runMeasure">{{ t('devices.measure') }}</AppButton>
        <p class="type-body-small muted hint">{{ measure.isPending.value ? t('devices.measuring') : t('devices.measureHint') }}</p>
      </div>
      <ErrorNotice v-if="measure.error.value" :error="measure.error.value" />
      <p v-else-if="measuredLine && !measure.isPending.value" class="type-body-small status" :class="measurement?.ok ? '' : 'warn'">
        {{ measuredLine }}<template v-if="measurement?.clipped"> · {{ t('devices.measureClipped') }}</template>
      </p>
    </template>
  </section>
</template>

<style scoped>
.devices { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.head { display: flex; align-items: center; gap: var(--app-space-2); }
.title { flex: 1; margin: 0; min-width: 0; overflow-wrap: anywhere; }
.roles { display: flex; flex-direction: column; gap: var(--app-space-5); }
.group { display: flex; flex-direction: column; gap: var(--app-space-2); min-width: 0; }
.below { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2) var(--app-space-4); }
/* The meter takes the row, as the input's does; the Test button keeps its size at the end. */
.level { flex: 1 1 240px; }
.monitor { display: flex; flex-direction: column; gap: var(--app-space-2); padding-left: var(--app-space-3); }
.advanced { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(240px, 100%), 1fr)); gap: var(--app-space-3) var(--app-space-6); align-items: center; padding: var(--app-space-2) 0; }
.warn { margin: 0; color: var(--md-sys-color-tertiary); }
.problem { display: flex; align-items: center; flex-wrap: wrap; gap: var(--app-space-2); color: var(--md-sys-color-error); }
.status { margin: 0; font-variant-numeric: tabular-nums; }
.measure { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2) var(--app-space-3); }
.hint { margin: 0; flex: 1 1 240px; min-width: 0; }
</style>
