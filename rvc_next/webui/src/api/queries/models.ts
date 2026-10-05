import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { computed, type MaybeRefOrGetter, toValue } from 'vue';
import { api, unwrap, uploadRaw } from '@/api/client';
import { keys } from '@/api/queries/keys';
import type { S, VoiceModel } from '@/api/types';
import { useJobsStore } from '@/stores/jobs';

export const useModels = (includeHidden: MaybeRefOrGetter<boolean> = false) =>
  useQuery({
    queryKey: computed(() => [...keys.models, { hidden: toValue(includeHidden) }]),
    queryFn: () => unwrap(api.GET('/api/v1/models', { params: { query: { include_hidden: toValue(includeHidden) } } })),
  });

export const useModel = (id: MaybeRefOrGetter<string | null | undefined>) =>
  useQuery({
    queryKey: computed(() => keys.model(toValue(id) ?? '')),
    queryFn: () => unwrap(api.GET('/api/v1/models/{voice_id}', { params: { path: { voice_id: toValue(id)! } } })),
    enabled: computed(() => !!toValue(id)),
  });

function useInvalidateModels() {
  const qc = useQueryClient();
  return () => qc.invalidateQueries({ queryKey: keys.models });
}

/** Ready-made voices (the official RVC demo voices) that can be downloaded into the library. */
export const useCatalogVoices = () => useQuery({ queryKey: keys.catalogVoices, queryFn: () => unwrap(api.GET('/api/v1/models/catalog')) });

export function useDownloadCatalogVoices() {
  const jobs = useJobsStore();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { ids?: string[]; all?: boolean }) => unwrap(api.POST('/api/v1/models/catalog/download', { body: { ids: body.ids ?? [], all: body.all ?? false } })),
    onSuccess: (job) => {
      jobs.upsert(job, 'models');
      qc.invalidateQueries({ queryKey: keys.catalogVoices });
    },
  });
}

export function useImportModelPaths() {
  const invalidate = useInvalidateModels();
  return useMutation({ mutationFn: (body: S['ImportPathRequest']) => unwrap(api.POST('/api/v1/models/import-path', { body })), onSuccess: invalidate });
}

export function useUpdateModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, patch }: { id: string; patch: S['VoiceUpdate'] }) => unwrap(api.PATCH('/api/v1/models/{voice_id}', { params: { path: { voice_id: id } }, body: patch })),
    onSuccess: (voice: VoiceModel) => {
      qc.setQueryData(keys.model(voice.id), voice);
      qc.invalidateQueries({ queryKey: keys.models });
    },
  });
}

export function useDeleteModel() {
  const invalidate = useInvalidateModels();
  return useMutation({ mutationFn: (id: string) => unwrap(api.DELETE('/api/v1/models/{voice_id}', { params: { path: { voice_id: id } } })), onSuccess: invalidate });
}

export function useAttachIndexFile() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, key, file, onProgress }: { id: string; key: string; file: File; onProgress?: (l: number, t: number) => void }) =>
      uploadRaw<VoiceModel>(`/api/v1/models/${encodeURIComponent(id)}/index`, { key, filename: file.name }, file, onProgress),
    onSuccess: (voice) => {
      qc.setQueryData(keys.model(voice.id), voice);
      qc.invalidateQueries({ queryKey: keys.models });
    },
  });
}

export function useSetIndexPath() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, key, path }: { id: string; key: string; path: string | null }) =>
      unwrap(api.POST('/api/v1/models/{voice_id}/index', { params: { path: { voice_id: id } }, body: { key, path } })),
    onSuccess: (voice) => {
      qc.setQueryData(keys.model(voice.id), voice);
      qc.invalidateQueries({ queryKey: keys.models });
    },
  });
}

export function useModelJobs() {
  const jobs = useJobsStore();
  const merge = useMutation({ mutationFn: (body: S['MergeRequest']) => unwrap(api.POST('/api/v1/models/merge', { body })), onSuccess: (job) => jobs.upsert(job, 'models') });
  const extract = useMutation({ mutationFn: (body: S['ExtractRequest']) => unwrap(api.POST('/api/v1/models/extract', { body })), onSuccess: (job) => jobs.upsert(job, 'models') });
  const buildIndex = useMutation({
    mutationFn: ({ id, experiment }: { id: string; experiment: string }) => unwrap(api.POST('/api/v1/models/{voice_id}/index/build', { params: { path: { voice_id: id } }, body: { experiment } })),
    onSuccess: (job) => jobs.upsert(job, 'models'),
  });
  return { merge, extract, buildIndex };
}

export function useLegacyRoots() {
  const invalidate = useInvalidateModels();
  const add = useMutation({ mutationFn: (body: S['LegacyRootCreate']) => unwrap(api.POST('/api/v1/models/legacy-roots', { body })), onSuccess: invalidate });
  const scan = useMutation({ mutationFn: (id: string) => unwrap(api.POST('/api/v1/models/legacy-roots/{root_id}/scan', { params: { path: { root_id: id } } })), onSuccess: invalidate });
  const importAll = useMutation({ mutationFn: (id: string) => unwrap(api.POST('/api/v1/models/legacy-roots/{root_id}/import', { params: { path: { root_id: id } } })), onSuccess: invalidate });
  return { add, scan, importAll };
}

export const revealModel = (id: string) => unwrap(api.POST('/api/v1/models/{voice_id}/reveal', { params: { path: { voice_id: id } } }));
