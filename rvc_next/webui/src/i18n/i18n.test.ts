import { describe, expect, it } from 'vitest';
import en from '@/i18n/en';
import zhCN from '@/i18n/zh-CN';
import { detectLocale, translate } from '@/i18n';

/** Every English message has one in Chinese, and the placeholders agree. */
function flatten(tree: unknown, prefix = ''): Map<string, string> {
  const out = new Map<string, string>();
  for (const [key, value] of Object.entries(tree as Record<string, unknown>)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (typeof value === 'string') out.set(path, value);
    else for (const [k, v] of flatten(value, path)) out.set(k, v);
  }
  return out;
}
const placeholders = (text: string) => [...text.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();

describe('messages', () => {
  const english = flatten(en);
  const chinese = flatten(zhCN);
  it('has the same keys in zh-CN', () => {
    expect([...chinese.keys()].sort()).toEqual([...english.keys()].sort());
  });
  it('uses the same placeholders in zh-CN', () => {
    const mismatched = [...english.entries()].filter(([k, v]) => JSON.stringify(placeholders(v)) !== JSON.stringify(placeholders(chinese.get(k) ?? '')));
    expect(mismatched.map(([k]) => k)).toEqual([]);
  });
  it('keeps the fixed vocabulary', () => {
    expect(translate('en', 'voice.label')).toBe('Voice');
    expect(translate('zh-CN', 'voice.label')).toBe('音色');
    expect(translate('en', 'params.indexRate')).toBe('Index strength');
    expect(translate('en', 'params.protect')).toBe('Consonant protection');
    expect(translate('en', 'params.rmsMix')).toBe('Loudness match');
    const all = [...english.values(), ...chinese.values()].join('\n');
    for (const retired of ['检索特征占比', '响度因子', 'rms_mix_rate']) expect(all).not.toContain(retired);
  });
});

describe('detectLocale', () => {
  it('picks the translation for the system language', () => {
    expect(detectLocale('zh-CN')).toBe('zh-CN');
    expect(detectLocale('zh-Hant-TW')).toBe('zh-CN');
    expect(detectLocale('en-GB')).toBe('en');
    expect(detectLocale('ja')).toBe('en');
    expect(detectLocale(undefined)).toBe('en');
  });
});
