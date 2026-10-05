import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { computed, type MaybeRefOrGetter, toValue } from 'vue';
import { api, unwrap } from '@/api/client';
import { keys } from '@/api/queries/keys';

export interface OutputFilter {
  job_id?: string;
  voice_id?: string;
  kind?: string;
}

export const useOutputs = (filter: MaybeRefOrGetter<OutputFilter>, enabled: MaybeRefOrGetter<boolean> = true) =>
  useQuery({
    queryKey: computed(() => keys.outputList({ ...toValue(filter) })),
    queryFn: () => unwrap(api.GET('/api/v1/outputs', { params: { query: { ...toValue(filter), limit: 200 } } })),
    enabled: computed(() => toValue(enabled)),
  });

export function useDeleteOutput() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => unwrap(api.DELETE('/api/v1/outputs/{output_id}', { params: { path: { output_id: id } } })),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.outputs }),
  });
}

export const revealOutput = (id: string) => unwrap(api.POST('/api/v1/outputs/{output_id}/reveal', { params: { path: { output_id: id } } }));
