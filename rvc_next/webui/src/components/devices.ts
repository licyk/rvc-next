import type { AudioDevice, DeviceList, DeviceSelection, PhysicalDevice, ResolvedDevice } from '@/api/types';
import type { DeviceOption } from '@/ui';

/** The value the device menus use for "follow the system default". */
export const DEFAULT_VALUE = '__default__';

export interface DeviceLabels {
  systemDefault: string;
  default: string;
  virtual: string;
  notConnected: string;
  /** Badges for a device listed in the other direction's menu (``live.show_all_devices``). */
  inputDevice?: string;
  outputDevice?: string;
}

/**
 * The options of one device menu: "System default" first, then the default device,
 * then the rest alphabetically; virtual devices badged; a saved device that is not in the list is
 * kept, greyed, as "not connected". A physical device is chosen by its key.
 */
export function deviceOptions(devices: PhysicalDevice[], saved: DeviceSelection | null | undefined, labels: DeviceLabels, direction?: 'input' | 'output'): DeviceOption[] {
  // A device of the other direction (the server lists them with live.show_all_devices) comes after
  // the menu's own devices and says what it is.
  const foreign = (d: PhysicalDevice) => direction !== undefined && d.direction !== direction;
  const sorted = [...devices].sort(
    (a, b) => Number(foreign(a)) - Number(foreign(b)) || (a.is_default === b.is_default ? a.name.localeCompare(b.name) : a.is_default ? -1 : 1),
  );
  const options: DeviceOption[] = [{ value: DEFAULT_VALUE, label: labels.systemDefault }];
  for (const d of sorted) {
    const badges: DeviceOption['badges'] = [];
    const other = foreign(d) ? (d.direction === 'input' ? labels.inputDevice : labels.outputDevice) : undefined;
    if (other) badges.push({ text: other, tone: 'warning' });
    if (d.is_default && !foreign(d)) badges.push({ text: labels.default, tone: 'primary' });
    if (d.is_virtual) badges.push({ text: labels.virtual, tone: 'neutral' });
    options.push({ value: d.key, label: d.name, badges });
  }
  const key = saved?.physical_key;
  if (key && !devices.some((d) => d.key === key)) {
    options.push({ value: key, label: saved?.name ?? key, badges: [{ text: labels.notConnected, tone: 'warning' }], unavailable: true });
  }
  return options;
}

/** The menu value for a saved selection. */
export function selectionValue(sel: DeviceSelection | null | undefined): string {
  return sel?.physical_key ?? (sel?.device_id ? sel.device_id : DEFAULT_VALUE);
}

/** The selection for a menu value: the physical device's recommended variant, or the default. */
export function selectionFor(value: string, devices: PhysicalDevice[], previous?: DeviceSelection | null): DeviceSelection {
  const base: DeviceSelection = { device_id: null, physical_key: null, name: null, host_api: null, channels: null, sample_rate: previous?.sample_rate ?? null, exclusive: false };
  if (value === DEFAULT_VALUE) return base;
  const d = devices.find((x) => x.key === value);
  if (!d) return previous ?? base;
  const v = d.variants.find((x) => x.id === d.recommended_id) ?? d.variants[0];
  return { ...base, device_id: v?.id ?? null, physical_key: d.key, name: d.name, host_api: v?.host_api ?? null, channels: previous?.physical_key === d.key ? previous.channels : null, exclusive: false };
}

/** The variants of the selected physical device, recommended first, for the Driver menu. */
export function driverOptions(devices: PhysicalDevice[], sel: DeviceSelection | null | undefined, recommendedLabel: (name: string) => string): { value: string; label: string }[] {
  const d = devices.find((x) => x.key === sel?.physical_key);
  if (!d) return [];
  return d.variants.map((v) => ({ value: v.id, label: v.id === d.recommended_id ? recommendedLabel(hostApiLabel(v.host_api)) : hostApiLabel(v.host_api) }));
}

/** The concrete device of a selection in this enumeration, if present. */
export function variantOf(devices: PhysicalDevice[], sel: DeviceSelection | null | undefined): AudioDevice | null {
  const d = devices.find((x) => x.key === sel?.physical_key);
  if (!d) return null;
  return d.variants.find((v) => v.id === sel?.device_id) ?? d.variants.find((v) => v.id === d.recommended_id) ?? d.variants[0] ?? null;
}

const HOST_API_NAMES: Record<string, string> = { wasapi: 'WASAPI', mme: 'MME', directsound: 'DirectSound', 'wdm-ks': 'WDM-KS', asio: 'ASIO', coreaudio: 'Core Audio', alsa: 'ALSA', jack: 'JACK', pulse: 'PulseAudio', oss: 'OSS' };
export const hostApiLabel = (id: string) => HOST_API_NAMES[id] ?? id;

/** Channel choices: input channels are mixed to mono, output channels receive the converted voice. */
/**
 * The channel choices for a device with ``count`` channels. The first (no selection) is what the stream
 * opens by default (``Endpoint.selected``): input channel 1, output channels 1 and 2 (1 on a mono device).
 */
export function channelOptions(
  count: number,
  direction: 'input' | 'output',
  labels: { automatic: (list: string) => string; channel: (n: number) => string; mix: (list: string) => string },
): { value: string; label: string }[] {
  const out = [{ value: '', label: labels.automatic(direction === 'output' && count >= 2 ? '1+2' : '1') }];
  for (let i = 1; i <= Math.min(count, 16); i++) out.push({ value: String(i), label: labels.channel(i) });
  if (count >= 2) out.push({ value: '1,2', label: labels.mix('1+2') });
  return out;
}

/** The status message for a resolved device, or null when it was found as saved. */
export function statusMessage(resolved: ResolvedDevice | undefined, format: (status: string, name: string) => string): string | null {
  if (!resolved || resolved.status === 'exact') return null;
  return resolved.message ?? format(resolved.status, resolved.device?.name ?? '');
}

/** Every physical device of a list, by direction. */
export function devicesFor(list: DeviceList | undefined, direction: 'input' | 'output'): PhysicalDevice[] {
  if (!list) return [];
  return direction === 'input' ? list.inputs : list.outputs;
}
