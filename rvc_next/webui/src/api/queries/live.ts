import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { computed, type MaybeRefOrGetter, toValue } from 'vue';
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
    meter: m((on: boolean) => unwrap(api.POST('/api/v1/live/meter', { body: { on } }))),
    passthrough: m((on: boolean) => unwrap(api.POST('/api/v1/live/passthrough', { body: { on } }))),
    startRecording: m((source: S['RecordingRequest']['source']) => unwrap(api.POST('/api/v1/live/recording', { body: { source } }))),
    stopRecording: m((_: void) => unwrap(api.DELETE('/api/v1/live/recording'))),
  };
}
