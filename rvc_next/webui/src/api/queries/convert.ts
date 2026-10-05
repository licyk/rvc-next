import { useMutation } from '@tanstack/vue-query';
import { api, unwrap } from '@/api/client';
import type { S } from '@/api/types';
import { useJobsStore } from '@/stores/jobs';

export function useConvert() {
  const jobs = useJobsStore();
  return useMutation({ mutationFn: (body: S['ConvertRequest']) => unwrap(api.POST('/api/v1/convert', { body })), onSuccess: (job) => jobs.upsert(job, 'convert') });
}
