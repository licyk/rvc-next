import { describe, expect, it } from 'vitest';
import { ApiError } from '@/api/client';
import { errorReport, errorView } from '@/components/errors';
import en from '@/i18n/en';

function lookup(key: string): string | undefined {
  let node: unknown = en;
  for (const part of key.split('.')) node = (node as Record<string, unknown> | undefined)?.[part];
  return typeof node === 'string' ? node : undefined;
}
const t = (key: string) => lookup(key) ?? key;
const tOr = (key: string, fallback: string) => lookup(key) ?? fallback;

describe('error view', () => {
  it('names the device role and its reason', () => {
    const view = errorView(
      { code: 'device_unavailable', message: 'The output device “Speakers” is in use by another program, or refused to open: Device unavailable', detail: { reason: 'busy', role: 'output', device: 'Speakers', host_api: 'wasapi' } },
      t,
      tOr,
    );
    expect(view.title).toBe('The output device is not available');
    expect(view.hint).toBe('In use by another program');
    expect(view.fields.map((f) => [f.label, f.value])).toEqual([
      ['Reason', 'busy'],
      ['Role', 'output'],
      ['Device', 'Speakers'],
      ['Driver', 'wasapi'],
    ]);
  });

  it('says when a device was lost during the session, and from the API too', () => {
    const lost = errorView(new ApiError(409, 'device_unavailable', 'Lost the input device “Mic”', { role: 'input', reason: 'missing', lost: true }), t, tOr);
    expect(lost.title).toBe('The input device was lost');
    expect(lost.fields.find((f) => f.key === 'lost')?.value).toBe('Yes');
    // Neither device singled out: the generic title.
    expect(errorView({ code: 'device_unavailable', message: 'x', detail: { role: null, reason: 'busy' } }, t, tOr).title).toBe('The audio device is not available');
  });

  it('keeps tracebacks and process output readable, and copies everything', () => {
    const view = errorView(
      { code: 'internal_error', message: 'The live worker stopped unexpectedly (killed by SIGSEGV (signal 11))', detail: { exit_code: -11, log: 'loading\nFatal Python error: Segmentation fault' } },
      t,
      tOr,
    );
    expect(view.title).toBe('A background process stopped unexpectedly');
    expect(view.fields).toEqual([{ key: 'exit_code', label: 'Exit code', value: '-11' }]);
    expect(view.blocks).toEqual([{ key: 'log', label: 'Process output', value: 'loading\nFatal Python error: Segmentation fault' }]);
    expect(errorReport(view, 'Error code')).toBe(
      [
        'A background process stopped unexpectedly',
        'The live worker stopped unexpectedly (killed by SIGSEGV (signal 11))',
        '',
        'Error code: internal_error',
        'Exit code: -11',
        '',
        'Process output:',
        'loading\nFatal Python error: Segmentation fault',
      ].join('\n'),
    );
  });

  it('falls back to the code title and the raw key for unknown fields', () => {
    const view = errorView({ code: 'asset_missing', message: 'HuBERT is missing', detail: { assets: ['hubert', 'rmvpe'], other: 3 } }, t, tOr);
    expect(view.title).toBe('A required model file is not installed');
    expect(view.fields).toEqual([
      { key: 'assets', label: 'Missing files', value: 'hubert, rmvpe' },
      { key: 'other', label: 'other', value: '3' },
    ]);
  });
});
