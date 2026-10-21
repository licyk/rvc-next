import type { AudioDevice, DeviceList, DeviceSelection, PhysicalDevice, ResolvedDevice } from '@/api/types';
import type { PickerOption } from '@/ui';

/** The value the device menus use for "follow the system default". */
export const DEFAULT_VALUE = '__default__';
/** The Driver menu's value for "the recommended driver, then each other while they refuse". */
export const AUTO_DRIVER = 'auto';

export interface DeviceLabels {
  systemDefault: string;
  default: string;
  virtual: string;
  notConnected: string;
  /** Badge for an output device recorded as an input (listed with ``live.show_all_devices``). */
  loopback?: string;
  /** The second line of "System default". */
  systemDefaultHint?: string;
  /** The second line of a device: its recommended driver, channels and rate. */
  describe?: (device: PhysicalDevice, variant: AudioDevice) => string;
}

/**
 * The options of one device menu: "System default" first, then the default device,
 * then the rest alphabetically; virtual devices badged; a saved device that is not in the list is
 * kept, greyed, as "not connected". A physical device is chosen by its key.
 */
export function deviceOptions(devices: PhysicalDevice[], saved: DeviceSelection | null | undefined, labels: DeviceLabels): PickerOption[] {
  // An output device recorded as an input (a loopback) comes after the real inputs and says what it is.
  const sorted = [...devices].sort(
    (a, b) => Number(a.is_loopback) - Number(b.is_loopback) || (a.is_default === b.is_default ? a.name.localeCompare(b.name) : a.is_default ? -1 : 1),
  );
  const options: PickerOption[] = [{ value: DEFAULT_VALUE, label: labels.systemDefault, description: labels.systemDefaultHint }];
  for (const d of sorted) {
    const badges: PickerOption['badges'] = [];
    if (d.is_loopback && labels.loopback) badges.push({ text: labels.loopback, tone: 'neutral' });
    if (d.is_default) badges.push({ text: labels.default, tone: 'primary' });
    if (d.is_virtual) badges.push({ text: labels.virtual, tone: 'neutral' });
    const v = d.variants.find((x) => x.id === d.recommended_id) ?? d.variants[0];
    options.push({ value: d.key, label: d.name, description: v && labels.describe ? labels.describe(d, v) : undefined, badges });
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

/** The selection for a menu value: the physical device, its driver chosen automatically (recommended first), or the default. */
export function selectionFor(value: string, devices: PhysicalDevice[], previous?: DeviceSelection | null): DeviceSelection {
  const base: DeviceSelection = { device_id: null, physical_key: null, name: null, host_api: null, driver: AUTO_DRIVER, channels: null, sample_rate: previous?.sample_rate ?? null, exclusive: false };
  if (value === DEFAULT_VALUE) return base;
  const d = devices.find((x) => x.key === value);
  if (!d) return previous ?? base;
  const v = d.variants.find((x) => x.id === d.recommended_id) ?? d.variants[0];
  return { ...base, device_id: v?.id ?? null, physical_key: d.key, name: d.name, host_api: v?.host_api ?? null, channels: previous?.physical_key === d.key ? previous.channels : null, exclusive: false };
}

/**
 * Whether a selection's driver is chosen automatically (the server's ``devices.is_auto_driver``):
 * exclusive mode fixes it; one saved before the choice existed is automatic on the recommended driver.
 */
export function isAutoDriver(devices: PhysicalDevice[], sel: DeviceSelection | null | undefined): boolean {
  if (!sel || sel.exclusive) return false;
  if (sel.driver) return sel.driver === AUTO_DRIVER;
  const d = devices.find((x) => x.key === sel.physical_key);
  const recommended = d?.variants.find((v) => v.id === d.recommended_id);
  return !sel.host_api || !recommended || sel.host_api === recommended.host_api;
}

/** The Driver menu: Auto first, then the selected physical device's variants, recommended first. */
export function driverOptions(
  devices: PhysicalDevice[],
  sel: DeviceSelection | null | undefined,
  recommendedLabel: (name: string) => string,
  autoLabel: (name: string) => string = (name) => `Auto (${name})`,
): { value: string; label: string }[] {
  const d = devices.find((x) => x.key === sel?.physical_key);
  if (!d) return [];
  const recommended = d.variants.find((v) => v.id === d.recommended_id) ?? d.variants[0];
  return [
    { value: AUTO_DRIVER, label: autoLabel(hostApiLabel(recommended?.host_api ?? '')) },
    ...d.variants.map((v) => ({ value: v.id, label: v.id === d.recommended_id ? recommendedLabel(hostApiLabel(v.host_api)) : hostApiLabel(v.host_api) })),
  ];
}

/** The Driver menu's value for a selection: Auto, or the variant fixed. */
export function driverValue(devices: PhysicalDevice[], sel: DeviceSelection | null | undefined): string | null {
  if (!sel) return null;
  return isAutoDriver(devices, sel) ? AUTO_DRIVER : sel.device_id;
}

/** The selection with the Driver menu's choice: Auto (back on the recommended variant), or that variant alone. */
export function withDriver(devices: PhysicalDevice[], sel: DeviceSelection, value: string): DeviceSelection {
  const d = devices.find((x) => x.key === sel.physical_key);
  if (!d) return sel;
  if (value === AUTO_DRIVER) {
    const v = d.variants.find((x) => x.id === d.recommended_id) ?? d.variants[0];
    return { ...sel, device_id: v?.id ?? null, host_api: v?.host_api ?? null, driver: AUTO_DRIVER, exclusive: false };
  }
  const v = d.variants.find((x) => x.id === value);
  return v ? { ...sel, device_id: v.id, host_api: v.host_api, driver: 'fixed' } : sel;
}

/** "Opened through DirectSound" and why the drivers before it refused, for a device whose automatic driver fell back; null otherwise. */
export function fallbackNote(device: ResolvedDevice | undefined, format: (driver: string, refused: string) => string): string | null {
  if (!device?.device || !device.fallback?.length) return null;
  const refused = device.fallback.map((a) => `${hostApiLabel(a.host_api ?? '')}: ${a.error}`).join('; ');
  return format(hostApiLabel(device.device.host_api), refused);
}

/**
 * Whether WASAPI's exclusive mode applies to a selection: its driver fixed to WASAPI or, for Auto,
 * WASAPI being the driver the device opened through (``inUse``: the open device, else the one the
 * check would open), the first one to try when neither is known.
 */
export function exclusiveApplies(devices: PhysicalDevice[], sel: DeviceSelection | null | undefined, inUse: AudioDevice | null | undefined): boolean {
  const auto = isAutoDriver(devices, sel) && inUse != null && inUse.physical_key === sel?.physical_key;
  return (auto ? inUse.host_api : variantOf(devices, sel)?.host_api) === 'wasapi';
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
