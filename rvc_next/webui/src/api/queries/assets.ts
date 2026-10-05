import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { api, unwrap } from '@/api/client';
import { keys } from '@/api/queries/keys';
import { useJobsStore } from '@/stores/jobs';

export const useAssets = () => useQuery({ queryKey: keys.assets, queryFn: () => unwrap(api.GET('/api/v1/assets')) });

/** The Hugging Face repositories models can come from (`downloads.repository` picks one). */
export const useAssetRepositories = () => useQuery({ queryKey: keys.assetRepositories, queryFn: () => unwrap(api.GET('/api/v1/assets/repositories')) });

/** Download (or verify) assets; the job appears in the jobs sheet. */
export function useDownloadAssets() {
  const jobs = useJobsStore();
  return useMutation({
    mutationFn: (body: { ids?: string[]; group?: 'inference' | 'training' | 'separation' | 'all' | null; verify?: boolean }) =>
      unwrap(
        body.verify
          ? api.POST('/api/v1/assets/verify', { body: { ids: body.ids ?? [], group: body.group ?? null } })
          : api.POST('/api/v1/assets/download', { body: { ids: body.ids ?? [], group: body.group ?? null } }),
      ),
    onSuccess: (job) => jobs.upsert(job, 'models'),
  });
}

/** Remove a downloaded asset's files; it can be downloaded again. */
export function useDeleteAsset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => unwrap(api.DELETE('/api/v1/assets/{asset_id}', { params: { path: { asset_id: id } } })),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: keys.assets });
      qc.invalidateQueries({ queryKey: keys.baseModels });
    },
  });
}
