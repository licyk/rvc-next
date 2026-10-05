import { defineStore } from 'pinia';
import { reactive } from 'vue';
import type { TrainMetric } from '@/api/types';

const MAX_POINTS = 20_000;

/** Training metrics per experiment, filled by REST and extended by ``train_metrics`` events. */
export const useTrainStore = defineStore('train', () => {
  const metrics = reactive<Record<string, TrainMetric[]>>({});

  function set(name: string, list: TrainMetric[]) {
    metrics[name] = list.slice(-MAX_POINTS);
  }
  function append(name: string, metric: TrainMetric) {
    const list = (metrics[name] ??= []);
    const last = list[list.length - 1];
    if (last && (metric.epoch < last.epoch || (metric.epoch === last.epoch && metric.step <= last.step))) return;
    list.push(metric);
    if (list.length > MAX_POINTS) list.splice(0, list.length - MAX_POINTS);
  }
  return { metrics, set, append };
});
