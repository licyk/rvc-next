import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { computed, type MaybeRefOrGetter, toValue } from 'vue';
import { api, unwrap } from '@/api/client';
import { keys } from '@/api/queries/keys';
import type { S } from '@/api/types';

/** A voice's presets (its default first) and the global ones. */
export const usePresets = (voiceId: MaybeRefOrGetter<string | null | undefined>) =>
  useQuery({
    queryKey: computed(() => keys.presets(toValue(voiceId))),
    queryFn: () => unwrap(api.GET('/api/v1/presets', { params: { query: { voice_id: toValue(voiceId) ?? undefined } } })),
  });

export function usePresetMutations() {
  const qc = useQueryClient();
  const invalidate = () => qc.invalidateQueries({ queryKey: ['presets'] });
  const create = useMutation({ mutationFn: (body: S['PresetCreate']) => unwrap(api.POST('/api/v1/presets', { body })), onSuccess: invalidate });
  const update = useMutation({
    mutationFn: ({ id, patch }: { id: string; patch: S['PresetUpdate'] }) => unwrap(api.PATCH('/api/v1/presets/{preset_id}', { params: { path: { preset_id: id } }, body: patch })),
    onSuccess: invalidate,
  });
  const remove = useMutation({ mutationFn: (id: string) => unwrap(api.DELETE('/api/v1/presets/{preset_id}', { params: { path: { preset_id: id } } })), onSuccess: invalidate });
  return { create, update, remove };
}
