import type { QueryClient, QueryKey } from '@tanstack/vue-query';
import { io, type Socket } from 'socket.io-client';
import { BASE_PATH, BASE_URL } from '@/api/baseUrl';
import { fetchActiveJobs } from '@/api/queries/jobs';
import { keys } from '@/api/queries/keys';
import { fetchLiveState } from '@/api/queries/live';
import type { DeviceList, LiveState, LiveStats, ServerEvents } from '@/api/types';
import { useAuthStore } from '@/stores/auth';
import { isFinished, useJobsStore } from '@/stores/jobs';
import { useLiveStore } from '@/stores/live';
import { useTrainStore } from '@/stores/train';

type Listeners = { [K in keyof ServerEvents]: (payload: ServerEvents[K]) => void };

let socket: Socket<Listeners> | null = null;

/** Invalidate a query key at most once per interval, so bursts of events cause one refetch. */
export function createInvalidator(qc: QueryClient, delay = 300) {
  const pending = new Map<string, QueryKey>();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const flush = () => {
    timer = undefined;
    for (const key of pending.values()) qc.invalidateQueries({ queryKey: key });
    pending.clear();
  };
  return (key: QueryKey) => {
    pending.set(JSON.stringify(key), key);
    if (!timer) timer = setTimeout(flush, delay);
  };
}

/** Listeners for ``job_log`` lines, keyed by job id (the open log viewers). */
const logListeners = new Map<string, Set<(offset: number, lines: string[]) => void>>();
export function onJobLog(jobId: string, listener: (offset: number, lines: string[]) => void): () => void {
  const set = logListeners.get(jobId) ?? new Set();
  set.add(listener);
  logListeners.set(jobId, set);
  return () => set.delete(listener);
}

/** Load what REST knows now: active jobs and the live state. Called on connect and reconnect. */
async function resync(qc: QueryClient) {
  const jobs = useJobsStore();
  const live = useLiveStore();
  try {
    jobs.loadActive((await fetchActiveJobs()).items);
  } catch {
    /* a token may be needed; the auth dialog handles it */
  }
  try {
    const state = (await fetchLiveState()) as LiveState;
    live.setState(state);
    qc.setQueryData(keys.liveState, state);
  } catch {
    /* Live not available on this server */
  }
}

/**
 * Connect once. REST stays the source of truth: events patch or invalidate the queries they affect,
 * high-frequency values go to the stores, and a reconnect refetches what is on screen.
 */
export function connectSocket(qc: QueryClient): Socket<Listeners> {
  if (socket) return socket;
  const jobs = useJobsStore();
  const live = useLiveStore();
  const train = useTrainStore();
  const invalidate = createInvalidator(qc);
  socket = io(new URL(BASE_URL).origin, {
    path: `${BASE_PATH}/ws/socket.io`,
    auth: (cb) => cb({ token: useAuthStore().token }),
    transports: ['websocket', 'polling'],
    reconnectionDelay: 1000,
    reconnectionDelayMax: 10_000,
  });

  socket.on('connect', () => resync(qc));
  socket.on('job_updated', ({ job }) => {
    if (!jobs.upsert(job)) return;
    qc.setQueryData(keys.job(job.id), job);
    // A voice download shows its progress on the catalog entry.
    if (job.kind === 'download' && job.state === 'running' && job.steps.some((s) => s.id.startsWith('voice:'))) invalidate(keys.catalogVoices);
    if (isFinished(job.state)) {
      invalidate(keys.jobs);
      if (job.kind === 'merge' || job.kind === 'extract' || job.kind === 'index') invalidate(keys.models);
      if (job.kind === 'train' || job.kind === 'index') invalidate(['train']);
      if (job.kind === 'download') invalidate(keys.models);
    }
  });
  socket.on('job_log', (e) => {
    for (const l of logListeners.get(e.job_id) ?? []) l(e.offset, e.lines);
  });
  socket.on('jobs_cleared', (e) => {
    jobs.remove(e.ids);
    invalidate(keys.jobs);
  });
  socket.on('outputs_added', () => invalidate(keys.outputs));
  socket.on('outputs_removed', () => invalidate(keys.outputs));
  socket.on('models_changed', (e) => {
    invalidate(keys.models);
    invalidate(keys.separationPresets);
    for (const id of [...(e.updated ?? []), ...(e.removed ?? [])]) invalidate(keys.model(id));
  });
  socket.on('presets_changed', () => invalidate(['presets']));
  socket.on('assets_changed', (e) => {
    qc.setQueryData(keys.assets, e.assets);
    // Official base models and the demo voices' state follow the files on disk.
    invalidate(keys.baseModels);
    invalidate(keys.catalogVoices);
  });
  socket.on('experiment_changed', (e) => {
    invalidate(keys.experiments);
    invalidate(keys.experiment(e.name));
    invalidate(keys.checkpoints(e.name));
  });
  socket.on('train_metrics', (e) => train.append(e.name, { epoch: e.epoch, step: e.step, losses: e.losses, lr: e.lr ?? null, time: null }));
  socket.on('live_state', ({ state }) => {
    live.setState(state as LiveState);
    qc.setQueryData(keys.liveState, state);
  });
  socket.on('live_stats', ({ stats }) => live.setStats(stats as LiveStats));
  socket.on('devices_changed', ({ devices }) => qc.setQueryData(keys.liveDevices, devices as DeviceList));
  socket.on('compute_changed', ({ usage }) => qc.setQueryData(keys.computeUsage, usage));
  socket.on('settings_changed', () => {
    invalidate(keys.settings);
    invalidate(keys.meta);
  });
  socket.io.on('reconnect', () => {
    resync(qc);
    for (const key of [keys.models, keys.outputs, keys.jobs, keys.assets, keys.experiments, keys.computeUsage, keys.liveDevices]) qc.invalidateQueries({ queryKey: key });
  });
  return socket;
}

export function disconnectSocket() {
  socket?.disconnect();
  socket = null;
}
