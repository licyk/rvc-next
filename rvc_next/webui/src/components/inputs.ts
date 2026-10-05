import type { AudioRef } from '@/api/types';

/** One input of Convert, Separate or a dataset: an upload, a server file or folder, or an earlier result. */
export interface InputItem {
  key: string;
  name: string;
  ref: AudioRef | null;
  duration?: number | null;
  status: 'uploading' | 'ready' | 'failed';
  progress?: number;
  error?: string;
  folder?: boolean;
  /** An audio file id the server knows (uploads and resolved server files), for playing and peaks. */
  fileId?: string | null;
  outputId?: string | null;
}

let counter = 0;
export const inputKey = () => `in-${Date.now().toString(36)}-${counter++}`;

/** The references a job request takes, for the inputs that are ready. */
export const readyRefs = (items: InputItem[]): AudioRef[] => items.filter((i) => i.status === 'ready' && i.ref).map((i) => i.ref as AudioRef);
