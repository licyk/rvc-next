import { ApiError } from '@/api/client';

export interface ErrorField {
  key: string;
  label: string;
  value: string;
}

export interface ErrorView {
  code: string;
  title: string;
  /** Why, in a few translated words (a device's reason code), or empty. */
  hint: string;
  message: string;
  detail: Record<string, unknown>;
  /** The detail's short values, labelled, for the Details list. */
  fields: ErrorField[];
  /** Its long or multi-line values (a traceback, a process's last output), shown preformatted. */
  blocks: ErrorField[];
}

type Translate = (key: string) => string;
type TranslateOr = (key: string, fallback: string) => string;

const ROLES = ['input', 'output', 'monitor'];
/** A detail value longer than this is shown as a block, not on one line. */
const LINE_LIMIT = 160;

function title(code: string, detail: Record<string, unknown>, t: Translate, tOr: TranslateOr): string {
  const role = typeof detail.role === 'string' && ROLES.includes(detail.role) ? detail.role : null;
  // Say which device: the input, the output or the monitor, and whether it was lost mid-session.
  if (code === 'device_unavailable' && role) return t(`errors.${detail.lost ? 'deviceLost' : 'device'}.${role}`);
  if (code === 'internal_error' && 'exit_code' in detail) return t('errors.crashed');
  return tOr(`errors.${code}`, t('errors.title'));
}

function show(value: unknown): string {
  if (typeof value === 'string') return value;
  if (Array.isArray(value) && value.every((v) => typeof v !== 'object' || v === null)) return value.join(', ');
  if (typeof value === 'object') return JSON.stringify(value, null, 2);
  return String(value);
}

/**
 * How an error is shown: the domain title for its code (translated; for a device, which one), the
 * server's message, and the detail behind a "Details" expander, labelled field by field with
 * tracebacks and process output kept readable.
 */
export function errorView(error: unknown, t: Translate, tOr: TranslateOr): ErrorView {
  let code: string;
  let message: string;
  let detail: Record<string, unknown>;
  if (error instanceof ApiError) {
    ({ code, message, detail } = error);
  } else {
    const e = error as { code?: string; message?: string; detail?: Record<string, unknown> } | null;
    code = e?.code ?? 'internal_error';
    message = e?.message ?? String(error ?? '');
    detail = e?.detail ?? {};
  }
  const reason = code === 'device_unavailable' && typeof detail.reason === 'string' ? tOr(`devices.reasons.${detail.reason}`, '') : '';
  const fields: ErrorField[] = [];
  const blocks: ErrorField[] = [];
  for (const [key, value] of Object.entries(detail)) {
    if (value === null || value === undefined || value === '' || value === false) continue;
    const text = value === true ? t('common.yes') : show(value);
    const field = { key, label: tOr(`errors.fields.${key}`, key), value: text };
    (text.includes('\n') || text.length > LINE_LIMIT ? blocks : fields).push(field);
  }
  return { code, title: title(code, detail, t, tOr), hint: reason, message, detail, fields, blocks };
}

/** The whole error as plain text, to paste into a bug report. */
export function errorReport(view: ErrorView, codeLabel: string): string {
  const lines = [view.hint ? `${view.title} (${view.hint})` : view.title];
  if (view.message && view.message !== view.title) lines.push(view.message);
  lines.push('', `${codeLabel}: ${view.code}`, ...view.fields.map((f) => `${f.label}: ${f.value}`));
  for (const b of view.blocks) lines.push('', `${b.label}:`, b.value);
  return lines.join('\n');
}

/** Copy text; falls back to a selection where the Clipboard API is missing (plain HTTP from another machine). */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fall through to the selection
  }
  const area = document.createElement('textarea');
  area.value = text;
  area.setAttribute('readonly', '');
  area.style.position = 'fixed';
  area.style.opacity = '0';
  document.body.appendChild(area);
  area.select();
  try {
    return document.execCommand('copy');
  } catch {
    return false;
  } finally {
    area.remove();
  }
}
