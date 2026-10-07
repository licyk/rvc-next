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

/** A spectrogram and pitch curve of an output, or of the file it was made from (``source``). */
export const useOutputAnalysis = (id: MaybeRefOrGetter<string | null>, source: MaybeRefOrGetter<boolean> = false) =>
  useQuery({
    queryKey: computed(() => ['outputs', 'analysis', toValue(id), toValue(source)] as const),
    queryFn: () => {
      const outputId = toValue(id) ?? '';
      const params = { params: { path: { output_id: outputId } } };
      return toValue(source) ? unwrap(api.GET('/api/v1/outputs/{output_id}/source-analysis', params)) : unwrap(api.GET('/api/v1/outputs/{output_id}/analysis', params));
    },
    enabled: computed(() => !!toValue(id)),
    staleTime: Infinity,
  });
