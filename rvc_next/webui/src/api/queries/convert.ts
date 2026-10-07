import { useMutation, useQuery } from '@tanstack/vue-query';
import { api, unwrap } from '@/api/client';
import { keys } from '@/api/queries/keys';
import type { S } from '@/api/types';
import { useJobsStore } from '@/stores/jobs';

export function useConvert() {
  const jobs = useJobsStore();
  return useMutation({ mutationFn: (body: S['ConvertRequest']) => unwrap(api.POST('/api/v1/convert', { body })), onSuccess: (job) => jobs.upsert(job, 'convert') });
}

/** The effects a chain can hold, their parameters' ranges, and whether pedalboard is installed. */
export const useEffectsCatalog = () => useQuery({ queryKey: keys.effects, queryFn: () => unwrap(api.GET('/api/v1/convert/effects')), staleTime: Infinity });
