<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { ApiError } from '@/api/client';
import { useMeta, useSettings, useUpdateSettings } from '@/api/queries/app';
import { useLiveControl } from '@/api/queries/live';
import type { LiveDevices, StreamParams, VoiceModel, VoiceParams } from '@/api/types';
import AssetGate from '@/components/AssetGate.vue';
import DevicePanel from '@/components/DevicePanel.vue';
import ErrorNotice from '@/components/ErrorNotice.vue';
import LiveStatsBar from '@/components/LiveStatsBar.vue';
import StreamParamsPanel from '@/components/StreamParamsPanel.vue';
import VoiceParamsPanel from '@/components/VoiceParamsPanel.vue';
import VoicePicker from '@/components/VoicePicker.vue';
import { defaults, pitchAssets } from '@/components/paramFields';
import { useI18n } from '@/i18n';
import { useLiveStore } from '@/stores/live';
import { AppButton, ConfirmDialog, EmptyState, PageFooter, Surface, icons } from '@/ui';

/**
 * Live: devices on top, voice and parameters in the middle (both hot), buffering below,
 * a sticky footer with Start/Stop and the stats. The session's config is saved by the server, so
 * Start with no changes resumes the previous session.
 */
const { t, tOr } = useI18n();
const meta = useMeta();
const settings = useSettings();
const updateSettings = useUpdateSettings();
const live = useLiveStore();
const control = useLiveControl();

const voiceId = ref<string | null>(null);
const voice = ref<VoiceModel | null>(null);
const params = ref<VoiceParams>(defaults<VoiceParams>('voice'));
const stream = ref<StreamParams>(defaults<StreamParams>('stream'));
const devices = ref<LiveDevices | null>(null);
const error = ref<unknown>(null);
const fallbackOpen = ref(false);
let seeded = false;

// Start from the running session's config, else the saved live settings.
watch(
  () => [settings.data.value, live.state.config] as const,
  ([s, config]) => {
    if (seeded || !s) return;
    seeded = true;
    const src = config ?? null;
    voiceId.value = src?.voice_id ?? s.live.last_voice ?? null;
    params.value = { ...(src?.params ?? s.live.last_params) };
    stream.value = { ...(src?.stream ?? s.live.stream) };
    devices.value = JSON.parse(JSON.stringify(src?.devices ?? s.live.devices));
  },
  { immediate: true },
);

const running = computed(() => live.active);
const speaker = computed({ get: () => params.value.speaker_id, set: (v: number) => onParams({ ...params.value, speaker_id: v }) });
const required = computed(() => ['hubert', ...pitchAssets(voice.value?.pitch_guidance, params.value.f0_method)]);

// While a session is active (running, loading, reconnecting…) a new voice is swapped in hot: the
// worker loads it beside the old one and switches between two blocks. The server's state says which
// voice is converting; if the new one is refused or will not load, the picker goes back to it.
function onVoice(v: VoiceModel | null) {
  const changed = voice.value && v && voice.value.id !== v.id;
  voice.value = v;
  if (!changed || !v || !live.active || v.id === live.state.voice_id) return;
  error.value = null;
  control.setVoice.mutate(v.id, {
    onError: (e) => {
      error.value = e;
      if (live.state.voice_id) voiceId.value = live.state.voice_id;
    },
  });
}
watch(
  () => live.state.voice_id,
  (id) => {
    if (id && live.active && id !== voiceId.value) voiceId.value = id;
  },
);
function onParams(p: VoiceParams) {
  params.value = p;
  if (live.active) control.updateVoice.mutate(p, { onError: (e) => (error.value = e) });
}
function onStream(s: StreamParams) {
  if (live.active) control.updateStream.mutate(s, { onError: (e) => (error.value = e) });
  else updateSettings.mutate({ live: { stream: s } });
}
function onDevices(d: LiveDevices) {
  if (live.active) control.setDevices.mutate(d);
  else updateSettings.mutate({ live: { devices: d } });
}
function increaseBlock(ms: number) {
  const next = { ...stream.value, block_ms: ms };
  stream.value = next;
  onStream(next);
}

function start(allowFallback = false) {
  if (!voiceId.value || !devices.value) return;
  error.value = null;
  control.start.mutate(
    { voice_id: voiceId.value, params: params.value, stream: stream.value, devices: devices.value, allow_output_fallback: allowFallback },
    {
      onError: (e) => {
        if (e instanceof ApiError && e.code === 'device_unavailable' && e.detail.reason === 'fallback') fallbackOpen.value = true;
        else error.value = e;
      },
    },
  );
}
// Between the click and the server's first state the request is under way: say so at once.
const requested = computed(() => control.start.isPending.value && !live.active);
const buttonText = computed(() => {
  if (live.busy) return tOr(live.statusKey, live.state.state);
  if (requested.value) return t('live.states.starting');
  return running.value ? t('live.stop') : t('live.start');
});
function toggle() {
  if (running.value) control.stop.mutate();
  else start();
}
</script>

<template>
  <div class="live">
    <EmptyState v-if="meta.data.value && !meta.data.value.live_available" :icon="icons.Unplug" :title="t('live.unavailable')" />
    <template v-else>
      <div class="sections">
        <Surface :level="0" shape="large" class="section">
          <DevicePanel v-if="devices" v-model="devices" :running="running" @change="onDevices" />
        </Surface>
        <Surface :level="0" shape="large" class="section">
          <VoicePicker v-model="voiceId" v-model:speaker="speaker" @voice="onVoice" />
          <VoiceParamsPanel :model-value="params" :voice="voice" live @update:model-value="params = $event" @change="onParams" />
        </Surface>
        <Surface :level="0" shape="large" class="section">
          <StreamParamsPanel v-model="stream" @change="onStream" />
        </Surface>
      </div>
      <PageFooter>
        <ErrorNotice v-if="error" :error="error" />
        <ErrorNotice v-else-if="live.state.error" :error="live.state.error" />
        <div class="bar">
          <AssetGate :assets="required">
            <AppButton class="start" :icon="running ? icons.Square : icons.Mic" :loading="live.busy || requested" :disabled="!voiceId && !running" @click="toggle">{{ buttonText }}</AppButton>
          </AssetGate>
          <span v-if="!voiceId && !running" class="type-body-small muted">{{ t('live.noVoice') }}</span>
          <LiveStatsBar class="stats" @increase-block="increaseBlock" />
        </div>
      </PageFooter>
      <ConfirmDialog v-model:open="fallbackOpen" :title="t('devices.output')" :message="t('live.confirmFallback')" :confirm-label="t('live.startAnyway')" :cancel-label="t('common.cancel')" @confirm="start(true)" />
    </template>
  </div>
</template>

<style scoped>
/* One column of sections, each as wide as the page (as Hanaikada's Settings); the footer stays in view. */
.live { display: flex; flex-direction: column; min-height: 100%; }
.sections { display: flex; flex-direction: column; gap: var(--app-space-4); flex: 1; padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); }
.section { display: flex; flex-direction: column; gap: var(--app-space-4); padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); min-width: 0; }
.bar { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-4); }
.start { --md-filled-button-container-height: 56px; min-width: 160px; }
.stats { flex: 1; min-width: 0; }
@container app-content (max-width: 599px) {
  .sections { padding: var(--app-space-3); }
  .section { padding: var(--app-space-4); }
}
</style>
