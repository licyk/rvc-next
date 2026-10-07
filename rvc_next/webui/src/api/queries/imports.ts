import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { computed, type MaybeRefOrGetter, toValue } from 'vue';
import { api, unwrap, uploadRaw } from '@/api/client';
import { keys } from '@/api/queries/keys';
import type { S } from '@/api/types';

export type ImportPlan = S['ImportPlan'];
export type ImportDecisions = S['ImportDecisions'];
export type ImportResult = S['ImportResult'];
export type StagedFile = S['StagedFile'];
export type InboxIndex = S['InboxIndex'];
export type BaseModel = S['BaseModel'];
export type SeparationModel = S['SeparationModel'];

/** The import-session calls: stage files, read the plan, commit with decisions, or discard. */
export const importApi = {
  create: () => unwrap(api.POST('/api/v1/models/imports')),
  addFile: (sessionId: string, file: File, onProgress?: (loaded: number, total: number) => void) =>
    uploadRaw<StagedFile[]>(`/api/v1/models/imports/${encodeURIComponent(sessionId)}/files`, { name: file.name }, file, onProgress),
  addUrls: (sessionId: string, urls: string[]) => unwrap(api.POST('/api/v1/models/imports/{session_id}/urls', { params: { path: { session_id: sessionId } }, body: { urls } })),
  plan: (sessionId: string) => unwrap(api.GET('/api/v1/models/imports/{session_id}', { params: { path: { session_id: sessionId } } })),
  commit: (sessionId: string, decisions: ImportDecisions) => unwrap(api.POST('/api/v1/models/imports/{session_id}/commit', { params: { path: { session_id: sessionId } }, body: decisions })),
  discard: (sessionId: string) => unwrap(api.DELETE('/api/v1/models/imports/{session_id}', { params: { path: { session_id: sessionId } } })),
};

export const useInboxIndexes = () => useQuery({ queryKey: keys.inbox, queryFn: () => unwrap(api.GET('/api/v1/models/indexes')) });

export function useInboxMutations() {
  const qc = useQueryClient();
  const done = () => {
    qc.invalidateQueries({ queryKey: keys.models });
  };
  return {
    assign: useMutation({
      mutationFn: ({ id, voiceId, key }: { id: string; voiceId: string; key?: string }) =>
        unwrap(api.POST('/api/v1/models/indexes/{index_id}/assign', { params: { path: { index_id: id } }, body: { voice_id: voiceId, key: key ?? 'default' } })),
      onSuccess: done,
    }),
    remove: useMutation({ mutationFn: (id: string) => unwrap(api.DELETE('/api/v1/models/indexes/{index_id}', { params: { path: { index_id: id } } })), onSuccess: done }),
  };
}

export const useBaseModels = (filter: MaybeRefOrGetter<{ sample_rate?: string; version?: string; pitch_guidance?: boolean } | null> = null) =>
  useQuery({
    queryKey: computed(() => [...keys.baseModels, toValue(filter) ?? {}]),
    queryFn: () => unwrap(api.GET('/api/v1/models/base', { params: { query: { ...(toValue(filter) ?? {}) } } })),
  });

export function useBaseModelMutations() {
  const qc = useQueryClient();
  const done = () => qc.invalidateQueries({ queryKey: keys.baseModels });
  return {
    rename: useMutation({ mutationFn: ({ id, name }: { id: string; name: string }) => unwrap(api.PATCH('/api/v1/models/base/{base_id}', { params: { path: { base_id: id } }, body: { name } })), onSuccess: done }),
    remove: useMutation({ mutationFn: (id: string) => unwrap(api.DELETE('/api/v1/models/base/{base_id}', { params: { path: { base_id: id } } })), onSuccess: done }),
  };
}

export const useSeparationModels = () => useQuery({ queryKey: keys.separationModels, queryFn: () => unwrap(api.GET('/api/v1/models/separation')) });

export function useSeparationModelMutations() {
  const qc = useQueryClient();
  const done = () => {
    qc.invalidateQueries({ queryKey: keys.separationModels });
    qc.invalidateQueries({ queryKey: keys.separationPresets });
  };
  return {
    remove: useMutation({ mutationFn: (id: string) => unwrap(api.DELETE('/api/v1/models/separation/{model_id}', { params: { path: { model_id: id } } })), onSuccess: done }),
  };
}
