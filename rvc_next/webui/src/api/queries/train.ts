import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { computed, type MaybeRefOrGetter, toValue } from 'vue';
import { api, unwrap } from '@/api/client';
import { keys } from '@/api/queries/keys';
import type { Experiment, S } from '@/api/types';
import { useJobsStore } from '@/stores/jobs';

export const useExperiments = () => useQuery({ queryKey: keys.experiments, queryFn: () => unwrap(api.GET('/api/v1/train/experiments')) });

const path = (name: string) => ({ params: { path: { name } } });

export const useExperiment = (name: MaybeRefOrGetter<string>) =>
  useQuery({ queryKey: computed(() => keys.experiment(toValue(name))), queryFn: () => unwrap(api.GET('/api/v1/train/experiments/{name}', path(toValue(name)))), enabled: computed(() => !!toValue(name)) });

export const useCheckpoints = (name: MaybeRefOrGetter<string>) =>
  useQuery({ queryKey: computed(() => keys.checkpoints(toValue(name))), queryFn: () => unwrap(api.GET('/api/v1/train/experiments/{name}/checkpoints', path(toValue(name)))), enabled: computed(() => !!toValue(name)) });

export const fetchMetrics = (name: string, since = 0) => unwrap(api.GET('/api/v1/train/experiments/{name}/metrics', { params: { path: { name }, query: { since } } }));

export function useExperimentMutations() {
  const qc = useQueryClient();
  const jobs = useJobsStore();
  const set = (exp: Experiment) => {
    qc.setQueryData(keys.experiment(exp.name), exp);
    qc.invalidateQueries({ queryKey: keys.experiments });
  };
  const create = useMutation({ mutationFn: (body: S['ExperimentCreate']) => unwrap(api.POST('/api/v1/train/experiments', { body })), onSuccess: set });
  const importLegacy = useMutation({ mutationFn: (body: S['ImportExperimentRequest']) => unwrap(api.POST('/api/v1/train/import', { body })), onSuccess: set });
  const update = useMutation({ mutationFn: ({ name, patch }: { name: string; patch: S['ExperimentUpdate'] }) => unwrap(api.PATCH('/api/v1/train/experiments/{name}', { ...path(name), body: patch })), onSuccess: set });
  const remove = useMutation({ mutationFn: (name: string) => unwrap(api.DELETE('/api/v1/train/experiments/{name}', path(name))), onSuccess: () => qc.invalidateQueries({ queryKey: keys.experiments }) });
  const scan = useMutation({ mutationFn: (name: string) => unwrap(api.POST('/api/v1/train/experiments/{name}/dataset/scan', path(name))) });
  const setSpeakers = useMutation({ mutationFn: ({ name, speakers }: { name: string; speakers: S['SpeakerEntry-Input'][] }) => unwrap(api.PUT('/api/v1/train/experiments/{name}/speakers', { ...path(name), body: speakers })), onSuccess: set });
  const fromFolders = useMutation({ mutationFn: ({ name, folder }: { name: string; folder: string }) => unwrap(api.POST('/api/v1/train/experiments/{name}/speakers/from-folders', { ...path(name), body: { folder } })), onSuccess: set });
  const run = useMutation({ mutationFn: ({ name, body }: { name: string; body: S['RunRequest'] }) => unwrap(api.POST('/api/v1/train/experiments/{name}/run', { ...path(name), body })), onSuccess: (job) => jobs.upsert(job, 'train') });
  const stop = useMutation({ mutationFn: (name: string) => unwrap(api.POST('/api/v1/train/experiments/{name}/stop', path(name))), onSuccess: set });
  const exportVoice = useMutation({ mutationFn: ({ name, body }: { name: string; body: S['ExportRequest'] }) => unwrap(api.POST('/api/v1/train/experiments/{name}/export', { ...path(name), body })), onSuccess: (job) => jobs.upsert(job, 'train') });
  const tryCheckpoint = useMutation({ mutationFn: ({ name, body }: { name: string; body: S['ExportRequest'] }) => unwrap(api.POST('/api/v1/train/experiments/{name}/try', { ...path(name), body })) });
  return { create, importLegacy, update, remove, scan, setSpeakers, fromFolders, run, stop, exportVoice, tryCheckpoint };
}
