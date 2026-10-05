import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { api, unwrap } from '@/api/client';
import { keys } from '@/api/queries/keys';

export const useComputeUsage = () => useQuery({ queryKey: keys.computeUsage, queryFn: () => unwrap(api.GET('/api/v1/compute/usage')), refetchInterval: 30_000 });

export const useComputeDevices = () => useQuery({ queryKey: keys.computeDevices, queryFn: () => unwrap(api.GET('/api/v1/compute/devices')), staleTime: 60_000 });

/** Free GPU memory: unload every cached model. */
export function useReleaseCompute() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => unwrap(api.POST('/api/v1/compute/release')),
    onSuccess: (usage) => qc.setQueryData(keys.computeUsage, usage),
  });
}
