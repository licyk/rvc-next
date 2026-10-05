import { useMutation, useQuery } from '@tanstack/vue-query';
import { api, unwrap } from '@/api/client';
import { keys } from '@/api/queries/keys';
import type { S } from '@/api/types';
import { useJobsStore } from '@/stores/jobs';

export const useSeparationPresets = () => useQuery({ queryKey: keys.separationPresets, queryFn: () => unwrap(api.GET('/api/v1/separate/presets')), staleTime: Infinity });

export function useSeparate() {
  const jobs = useJobsStore();
  return useMutation({ mutationFn: (body: S['SeparateRequest']) => unwrap(api.POST('/api/v1/separate', { body })), onSuccess: (job) => jobs.upsert(job, 'separate') });
}
