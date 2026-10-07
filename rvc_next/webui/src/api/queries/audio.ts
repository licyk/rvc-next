import { useQuery } from '@tanstack/vue-query';
import { computed, type MaybeRefOrGetter, toValue } from 'vue';
import { api, unwrap, uploadRaw } from '@/api/client';
import { keys } from '@/api/queries/keys';
import type { AudioFile, Peaks, S } from '@/api/types';

export const uploadAudio = (file: File, onProgress?: (loaded: number, total: number) => void, signal?: AbortSignal) =>
  uploadRaw<AudioFile>('/api/v1/audio/uploads', { name: file.name }, file, onProgress, signal);

export const synthesizeSpeech = (body: S['TtsRequest']) => unwrap(api.POST('/api/v1/tts', { body }));

export const useTtsVoices = (enabled: MaybeRefOrGetter<boolean>) =>
  useQuery({ queryKey: keys.ttsVoices, queryFn: () => unwrap(api.GET('/api/v1/tts/voices')), enabled: computed(() => toValue(enabled)), staleTime: Infinity });

export const resolveServerPath = (path: string) => unwrap(api.POST('/api/v1/audio/resolve-path', { body: { path } }));

export const useBrowse = (path: MaybeRefOrGetter<string>, enabled: MaybeRefOrGetter<boolean> = true) =>
  useQuery({
    queryKey: computed(() => keys.browse(toValue(path))),
    queryFn: () => unwrap(api.GET('/api/v1/audio/browse', { params: { query: { path: toValue(path) } } })),
    enabled: computed(() => toValue(enabled)),
  });

export type PeaksSource = { kind: 'file' | 'output' | 'output-source'; id: string };

export function fetchPeaks(source: PeaksSource, points: number): Promise<Peaks> {
  const query = { points };
  if (source.kind === 'file') return unwrap(api.GET('/api/v1/audio/files/{file_id}/peaks', { params: { path: { file_id: source.id }, query } }));
  if (source.kind === 'output') return unwrap(api.GET('/api/v1/outputs/{output_id}/peaks', { params: { path: { output_id: source.id }, query } }));
  return unwrap(api.GET('/api/v1/outputs/{output_id}/source-peaks', { params: { path: { output_id: source.id }, query } }));
}

/** Peaks are computed once on the server and never change for a file; cache them for the session. */
export const usePeaks = (source: MaybeRefOrGetter<PeaksSource | null>, points = 1024) =>
  useQuery({
    queryKey: computed(() => {
      const s = toValue(source);
      return keys.peaks(s?.kind ?? '', s?.id ?? '', points);
    }),
    queryFn: () => fetchPeaks(toValue(source)!, points),
    enabled: computed(() => !!toValue(source)),
    staleTime: Infinity,
  });
