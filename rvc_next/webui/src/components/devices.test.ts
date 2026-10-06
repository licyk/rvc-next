import { describe, expect, it } from 'vitest';
import type { AudioDevice, PhysicalDevice } from '@/api/types';
import { DEFAULT_VALUE, channelOptions, deviceOptions, driverOptions, selectionFor, selectionValue, statusMessage, variantOf } from '@/components/devices';

const variant = (id: string, host: string, name: string): AudioDevice => ({
  id, physical_key: name.toLowerCase(), name, raw_name: name.slice(0, 31), host_api: host, direction: 'input', channels: 2, default_sample_rate: 48000,
  channel_counts: [], supported_rates: [44100, 48000], latency_ms: [10, 40], is_default: false, is_virtual: false, portaudio_index: 1, loopback_of: null, loopback_source: null,
});
const phys = (name: string, extra: Partial<PhysicalDevice> = {}): PhysicalDevice => {
  const variants = [variant(`${name}-wasapi`, 'wasapi', name), variant(`${name}-mme`, 'mme', name)];
  return { key: name.toLowerCase(), name, direction: 'input', is_default: false, is_virtual: false, is_loopback: false, variants, recommended_id: variants[0].id, ...extra };
};
const labels = { systemDefault: 'System default', default: 'default', virtual: 'virtual', notConnected: 'not connected' };

describe('device menu options', () => {
  const list = [phys('Zeta Mic'), phys('CABLE Output (VB-Audio Virtual Cable)', { is_virtual: true }), phys('Microphone (USB Audio Device)', { is_default: true })];

  it('lists the system default, then the default device, then the rest alphabetically, with badges', () => {
    const options = deviceOptions(list, null, labels);
    expect(options.map((o) => o.label)).toEqual(['System default', 'Microphone (USB Audio Device)', 'CABLE Output (VB-Audio Virtual Cable)', 'Zeta Mic']);
    expect(options[1].badges?.map((b) => b.text)).toEqual(['default']);
    expect(options[2].badges?.map((b) => b.text)).toEqual(['virtual']);
    expect(options[0].value).toBe(DEFAULT_VALUE);
  });

  it('puts output devices recorded as inputs last, marked as loopback', () => {
    const speakers = phys('Speakers (Realtek)', { key: 'loopback:speakers (realtek)', is_loopback: true });
    const options = deviceOptions([speakers, ...list], null, { ...labels, loopback: 'loopback' });
    expect(options.at(-1)).toMatchObject({ value: 'loopback:speakers (realtek)', label: 'Speakers (Realtek)', badges: [{ text: 'loopback', tone: 'neutral' }] });
    expect(options[1].label).toBe('Microphone (USB Audio Device)');
  });

  it('keeps a saved device that is gone, greyed and unselectable', () => {
    const options = deviceOptions(list, { physical_key: 'old headset', name: 'Old Headset', device_id: 'x', host_api: 'wasapi', channels: null, sample_rate: null, exclusive: false }, labels);
    const gone = options[options.length - 1];
    expect(gone).toMatchObject({ value: 'old headset', label: 'Old Headset', unavailable: true });
    expect(gone.badges?.[0].text).toBe('not connected');
  });

  it('chooses the recommended driver for a physical device, and back to default', () => {
    const sel = selectionFor('zeta mic', list);
    expect(sel).toMatchObject({ physical_key: 'zeta mic', device_id: 'Zeta Mic-wasapi', host_api: 'wasapi' });
    expect(selectionValue(sel)).toBe('zeta mic');
    expect(selectionFor(DEFAULT_VALUE, list, sel).physical_key).toBeNull();
    expect(selectionValue(selectionFor(DEFAULT_VALUE, list))).toBe(DEFAULT_VALUE);
    const drivers = driverOptions(list, sel, (n) => `${n} (recommended)`);
    expect(drivers.map((d) => d.label)).toEqual(['WASAPI (recommended)', 'MME']);
    expect(variantOf(list, { ...sel, device_id: 'Zeta Mic-mme' })?.host_api).toBe('mme');
  });

  it('offers channels and reports resolution statuses', () => {
    const labels = { automatic: (l: string) => `default ${l}`, channel: (n: number) => `ch${n}`, mix: (l: string) => `mix ${l}` };
    const ch = channelOptions(2, 'input', labels);
    expect(ch.map((c) => c.value)).toEqual(['', '1', '2', '1,2']);
    // The default names what the stream opens: input channel 1, output 1 and 2 (1 on a mono device).
    expect(ch[0].label).toBe('default 1');
    expect(channelOptions(2, 'output', labels)[0].label).toBe('default 1+2');
    expect(channelOptions(1, 'output', labels)[0].label).toBe('default 1');
    const fmt = (status: string, name: string) => `${status}:${name}`;
    expect(statusMessage({ role: 'input', status: 'exact', device: null, message: null }, fmt)).toBeNull();
    expect(statusMessage({ role: 'output', status: 'default_fallback', device: null, message: null }, fmt)).toBe('default_fallback:');
    expect(statusMessage({ role: 'output', status: 'matched', device: null, message: 'found under MME' }, fmt)).toBe('found under MME');
  });
});
