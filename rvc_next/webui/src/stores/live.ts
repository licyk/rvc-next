import { defineStore } from 'pinia';
import { computed, ref } from 'vue';
import type { LiveState, LiveStats } from '@/api/types';

const STOPPED: LiveState = { state: 'stopped', voice_id: null, config: null, resolved: [], topology: null, sample_rate: null, meter: false, passthrough: false, error: null, started_at: null, stage: null, latency_test: null };

/** Live session state and its 10 Hz stats. Stats never enter the query cache. */
export const useLiveStore = defineStore('live', () => {
  const state = ref<LiveState>({ ...STOPPED });
  const stats = ref<LiveStats | null>(null);
  const statsAt = ref(0);

  function setState(next: LiveState) {
    state.value = next;
    if (next.state === 'stopped' && !next.meter) stats.value = null;
  }
  function setStats(next: LiveStats) {
    stats.value = next;
    statsAt.value = Date.now();
  }

  const running = computed(() => state.value.state === 'running');
  const busy = computed(() => ['starting', 'loading', 'prewarming', 'stopping'].includes(state.value.state));
  const active = computed(() => state.value.state !== 'stopped' && state.value.state !== 'error');
  /** The i18n key that says what Live is doing: the step under way while starting, else the state. */
  const statusKey = computed(() => (state.value.stage ? `live.stages.${state.value.stage}` : `live.states.${state.value.state}`));
  /** Inference time against the block length; above 0.8 the session is close to falling behind. */
  const load = computed(() => (stats.value && stats.value.block_ms > 0 ? stats.value.infer_ms_p95 / stats.value.block_ms : 0));

  return { state, stats, statsAt, setState, setStats, running, busy, active, statusKey, load };
});
