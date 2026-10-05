import { PARAM_SPECS, type ParamSpec } from '@/api/paramSpecs';

/** The voice parameters in their one order, with the slider step of each. */
export const VOICE_FIELDS = [
  { key: 'pitch', label: 'params.pitch', help: 'params.pitchHelp', step: 1, unit: 'st' },
  { key: 'formant', label: 'params.formant', help: 'params.formantHelp', step: 0.1, unit: 'st' },
  { key: 'index_rate', label: 'params.indexRate', help: 'params.indexHelp', step: 0.01, unit: '' },
  { key: 'protect', label: 'params.protect', help: 'params.protectHelp', step: 0.01, unit: '' },
  { key: 'rms_mix_rate', label: 'params.rmsMix', help: 'params.rmsHelp', step: 0.01, unit: '' },
] as const;

export const STREAM_FIELDS = [
  { key: 'block_ms', label: 'stream.block', help: 'stream.blockHelp', step: 10, unit: 'ms' },
  { key: 'crossfade_ms', label: 'stream.crossfade', help: 'stream.crossfadeHelp', step: 10, unit: 'ms' },
  { key: 'context_ms', label: 'stream.context', help: 'stream.contextHelp', step: 50, unit: 'ms' },
  { key: 'threshold_db', label: 'stream.threshold', help: 'stream.thresholdHelp', step: 1, unit: 'dB' },
] as const;

/** The server's range for a parameter; the panels never hard-code limits. */
export function spec(group: 'voice' | 'stream', key: string): Required<Pick<ParamSpec, 'min' | 'max'>> & ParamSpec {
  const s = PARAM_SPECS[group][key];
  return { ...s, min: s?.min ?? 0, max: s?.max ?? 1 };
}

export function defaults<T>(group: 'voice' | 'stream'): T {
  return Object.fromEntries(Object.entries(PARAM_SPECS[group]).map(([k, s]) => [k, s.default])) as T;
}

/** Each stream setting's share of the estimated latency, in ms. */
export function latencyShare(key: string, value: number, crossfade: number): number | null {
  if (key === 'block_ms') return 2 * value;
  if (key === 'crossfade_ms') return value;
  if (key === 'input_denoise') return value ? Math.min(crossfade, 40) : 0;
  return null;
}

/** The pitch model a voice needs for ``method``, as an asset id (the server's ``required_assets``); pm needs none. */
export function pitchAssets(pitchGuidance: boolean | undefined, method: string | undefined): string[] {
  if (!pitchGuidance) return [];
  return method === 'rmvpe' ? ['rmvpe'] : method === 'fcpe' ? ['fcpe'] : [];
}
