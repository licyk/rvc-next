import { ApiError } from '@/api/client';

export interface ErrorView {
  code: string;
  title: string;
  message: string;
  detail: Record<string, unknown>;
}

/**
 * How an error is shown: the domain title for its code (translated), the server's message, and
 * the detail behind a "Details" expander. Tracebacks stay in job logs.
 */
export function errorView(error: unknown, t: (key: string) => string, tOr: (key: string, fallback: string) => string): ErrorView {
  if (error instanceof ApiError) return { code: error.code, title: tOr(`errors.${error.code}`, t('errors.title')), message: error.message, detail: error.detail };
  const e = error as { code?: string; message?: string; detail?: Record<string, unknown> } | null;
  const code = e?.code ?? 'internal_error';
  return { code, title: tOr(`errors.${code}`, t('errors.title')), message: e?.message ?? String(error ?? ''), detail: e?.detail ?? {} };
}
