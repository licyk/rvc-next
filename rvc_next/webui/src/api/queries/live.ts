import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { computed, type MaybeRefOrGetter, onActivated, onBeforeUnmount, onDeactivated, toValue, watch } from 'vue';
import { api, unwrap } from '@/api/client';
import { keys } from '@/api/queries/keys';
import type { DeviceCheck, DeviceList, LatencyMeasurement, LiveState, S } from '@/api/types';
import { useLiveStore } from '@/stores/live';

export const fetchLiveState = () => unwrap(api.GET('/api/v1/live/state'));

export const useLiveDevices = (enabled: MaybeRefOrGetter<boolean> = true) =>
  useQuery({
    queryKey: keys.liveDevices,
    queryFn: async () => (await unwrap(api.GET('/api/v1/live/devices', { params: { query: { refresh: false } } }))) as DeviceList,
    enabled: computed(() => toValue(enabled)),
    retry: false,
  });

export function useRefreshDevices() {
  const qc = useQueryClient();
  return useMutation({ mutationFn: async () => (await unwrap(api.GET('/api/v1/live/devices', { params: { query: { refresh: true } } }))) as DeviceList, onSuccess: (list) => qc.setQueryData(keys.liveDevices, list) });
}

export const checkDevices = async (devices: S['LiveDevices']) => (await unwrap(api.POST('/api/v1/live/devices/check', { body: devices }))) as DeviceCheck;

/** A loopback measurement takes a few seconds; the result also reaches the state (``latency_test``) by the socket. */
export const useMeasureLatency = () =>
  useMutation({ mutationFn: async (body: S['LatencyTestRequest']) => (await unwrap(api.POST('/api/v1/live/devices/latency-test', { body }))) as LatencyMeasurement });

/** Every live control answers with the new state; the store keeps it, the socket keeps it current. */
export function useLiveControl() {
  const store = useLiveStore();
  const qc = useQueryClient();
  const apply = (state: LiveState) => {
    store.setState(state);
    qc.setQueryData(keys.liveState, state);
  };
  // The state's device records carry a [low, high] latency pair the client never indexes; read it as LiveState.
  const m = <A>(fn: (arg: A) => Promise<unknown>) => useMutation({ mutationFn: async (arg: A) => (await fn(arg)) as LiveState, onSuccess: apply });
  return {
    start: m((config: S['LiveConfig']) => unwrap(api.POST('/api/v1/live/start', { body: config }))),
    stop: m((_: void) => unwrap(api.POST('/api/v1/live/stop'))),
    updateVoice: m((params: S['VoiceParamsModel']) => unwrap(api.PATCH('/api/v1/live/voice', { body: params }))),
    updateStream: m((stream: S['StreamParamsModel']) => unwrap(api.PATCH('/api/v1/live/stream', { body: stream }))),
    setVoice: m((voiceId: string) => unwrap(api.PUT('/api/v1/live/voice-model', { body: { voice_id: voiceId } }))),
    setDevices: m((devices: S['LiveDevices']) => unwrap(api.PUT('/api/v1/live/devices', { body: devices }))),
    testTone: m((body: S['TestToneRequest']) => unwrap(api.POST('/api/v1/live/devices/test-tone', { body }))),
    passthrough: m((on: boolean) => unwrap(api.POST('/api/v1/live/passthrough', { body: { on } }))),
    startRecording: m((source: S['RecordingRequest']['source']) => unwrap(api.POST('/api/v1/live/recording', { body: { source } }))),
    stopRecording: m((_: void) => unwrap(api.DELETE('/api/v1/live/recording'))),
  };
}

/**
 * The input meter while Live is idle. The server keeps it for a 15 s lease and opens it in the
 * background (no request waits for the worker or the device); every shown device panel holds a
 * claim, and while any does and the page is visible the lease is renewed every 5 s. A hidden or
 * closed page lets the lease lapse, so the microphone is let go.
 */
const METER_RENEW_MS = 5000;
let meterClaims = 0;
let meterTimer: ReturnType<typeof setInterval> | undefined;

function askMeter(on: boolean) {
  api.POST('/api/v1/live/meter', { body: { on } }).catch(() => undefined); // the next renewal tries again
}
function renewMeter() {
  if (meterClaims > 0 && document.visibilityState === 'visible') askMeter(true);
}
function claimMeter(delta: 1 | -1) {
  meterClaims = Math.max(0, meterClaims + delta);
  if (delta > 0 && meterClaims === 1) {
    renewMeter();
    meterTimer = setInterval(renewMeter, METER_RENEW_MS);
    document.addEventListener('visibilitychange', renewMeter);
  } else if (delta < 0 && meterClaims === 0) {
    clearInterval(meterTimer);
    document.removeEventListener('visibilitychange', renewMeter);
    askMeter(false);
  }
}

/** Hold the input meter while ``wanted`` and the calling view is shown (not kept alive in the background). */
export function useMeterClaim(wanted: MaybeRefOrGetter<boolean>) {
  let held = false;
  let shown = true;
  const sync = () => {
    const want = shown && toValue(wanted);
    if (want !== held) claimMeter(want ? 1 : -1);
    held = want;
  };
  watch(() => toValue(wanted), sync, { immediate: true });
  onActivated(() => {
    shown = true;
    sync();
  });
  onDeactivated(() => {
    shown = false;
    sync();
  });
  onBeforeUnmount(() => {
    shown = false;
    sync();
  });
}
