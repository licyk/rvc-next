import { BASE_URL } from '@/api/baseUrl';
import { useAuthStore } from '@/stores/auth';

/**
 * Live with this browser's microphone and speakers: the microphone goes to the server over a
 * WebSocket (``/live/browser-audio``) as mono 16-bit PCM in 20 ms frames, and the converted audio
 * comes back the same way. An AudioWorklet does both ends on the audio thread: it cuts the
 * microphone into frames and plays what comes back from a jitter buffer that starts once 60 ms are
 * in, refills after an underrun and drops back to 60 ms when more than 250 ms pile up.
 */

const PROCESSOR = 'rvc-browser-io';

const WORKLET = `
class BrowserIo extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options.processorOptions;
    this.frame = new Float32Array(o.frame);
    this.filled = 0;
    this.buf = new Float32Array(o.capacity);
    this.read = 0;
    this.size = 0;
    this.target = o.target;
    this.max = o.max;
    this.playing = false;
    this.underruns = 0;
    this.dropped = 0;
    this.quanta = 0;
    this.port.onmessage = (e) => this.push(e.data);
  }
  push(x) {
    const cap = this.buf.length;
    for (let i = 0; i < x.length; i++) {
      if (this.size === cap) { this.read = (this.read + 1) % cap; this.size--; this.dropped++; }
      this.buf[(this.read + this.size) % cap] = x[i];
      this.size++;
    }
    if (this.size > this.max) {
      const drop = this.size - this.target;
      this.read = (this.read + drop) % cap;
      this.size -= drop;
      this.dropped += drop;
    }
  }
  process(inputs, outputs) {
    const mic = inputs[0] && inputs[0][0];
    if (mic) {
      let i = 0;
      while (i < mic.length) {
        const n = Math.min(mic.length - i, this.frame.length - this.filled);
        this.frame.set(mic.subarray(i, i + n), this.filled);
        this.filled += n;
        i += n;
        if (this.filled === this.frame.length) {
          this.port.postMessage(this.frame.slice());
          this.filled = 0;
        }
      }
    }
    const out = outputs[0];
    const q = out[0].length;
    if (!this.playing && this.size >= this.target) this.playing = true;
    if (this.playing && this.size >= q) {
      const cap = this.buf.length;
      for (let i = 0; i < q; i++) out[0][i] = this.buf[(this.read + i) % cap];
      this.read = (this.read + q) % cap;
      this.size -= q;
    } else {
      if (this.playing) { this.underruns++; this.playing = false; }
      out[0].fill(0);
    }
    for (let c = 1; c < out.length; c++) out[c].set(out[0]);
    if (++this.quanta % 200 === 0) this.port.postMessage({ buffered: this.size, underruns: this.underruns, dropped: this.dropped });
    return true;
  }
}
registerProcessor('${PROCESSOR}', BrowserIo);
`;

export interface BrowserAudioStats {
  /** Converted audio waiting to play, in ms (the jitter buffer). */
  bufferedMs: number;
  underruns: number;
  droppedMs: number;
}

export type BrowserAudioProblem = 'insecure' | 'unsupported';

/** Why this page cannot use the microphone, if it cannot: browsers allow it only over HTTPS or on localhost. */
export function browserAudioProblem(): BrowserAudioProblem | null {
  if (typeof window === 'undefined') return 'unsupported';
  if (!window.isSecureContext) return 'insecure';
  if (!navigator.mediaDevices?.getUserMedia || typeof AudioWorkletNode === 'undefined') return 'unsupported';
  return null;
}

export function browserAudioUrl(base = BASE_URL, token: string | null = useAuthStore().token): string {
  const url = `${base.replace(/^http/, 'ws')}/api/v1/live/browser-audio`;
  return token ? `${url}?token=${encodeURIComponent(token)}` : url;
}

export function toPcm16(x: Float32Array): Int16Array<ArrayBuffer> {
  const out = new Int16Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = Math.max(-1, Math.min(1, x[i]!)) * 32767;
  return out;
}

export function fromPcm16(data: ArrayBuffer): Float32Array {
  const pcm = new Int16Array(data, 0, data.byteLength >> 1);
  const out = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) out[i] = pcm[i]! / 32768;
  return out;
}

export class BrowserAudio {
  /** The connection to the server closed by itself (the server stopped, or the network went). */
  onClose: (() => void) | null = null;
  onStats: ((stats: BrowserAudioStats) => void) | null = null;
  private ctx: AudioContext | null = null;
  private media: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private ws: WebSocket | null = null;
  private stopping = false;

  get sampleRate(): number {
    return this.ctx?.sampleRate ?? 0;
  }

  /** Ask for the microphone, start the audio and connect; resolves with the audio rate. */
  async start(url = browserAudioUrl()): Promise<number> {
    this.stopping = false;
    try {
      // The browser's own processing would fight the conversion (and echo cancellation needs the far end).
      this.media = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } });
      const ctx = new AudioContext({ latencyHint: 'interactive' });
      this.ctx = ctx;
      const module = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }));
      try {
        await ctx.audioWorklet.addModule(module);
      } finally {
        URL.revokeObjectURL(module);
      }
      const rate = ctx.sampleRate;
      const node = new AudioWorkletNode(ctx, PROCESSOR, {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [2],
        processorOptions: { frame: Math.round(rate * 0.02), capacity: rate * 2, target: Math.round(rate * 0.06), max: Math.round(rate * 0.25) },
      });
      this.node = node;
      ctx.createMediaStreamSource(this.media).connect(node);
      node.connect(ctx.destination);
      const ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';
      this.ws = ws;
      await new Promise<void>((resolve, reject) => {
        ws.onopen = () => resolve();
        ws.onerror = () => reject(new Error('The server refused the browser audio connection'));
      });
      node.port.onmessage = (e: MessageEvent) => {
        if (e.data instanceof Float32Array) {
          if (ws.readyState === WebSocket.OPEN) ws.send(toPcm16(e.data).buffer);
        } else {
          const s = e.data as { buffered: number; underruns: number; dropped: number };
          this.onStats?.({ bufferedMs: (s.buffered / rate) * 1000, underruns: s.underruns, droppedMs: (s.dropped / rate) * 1000 });
        }
      };
      ws.onmessage = (e: MessageEvent<ArrayBuffer>) => {
        const samples = fromPcm16(e.data);
        node.port.postMessage(samples, [samples.buffer]);
      };
      ws.onclose = () => {
        if (!this.stopping) this.onClose?.();
      };
      await ctx.resume();
      return rate;
    } catch (e) {
      this.stop();
      throw e;
    }
  }

  stop(): void {
    this.stopping = true;
    this.ws?.close();
    this.ws = null;
    this.node?.disconnect();
    this.node = null;
    this.media?.getTracks().forEach((t) => t.stop());
    this.media = null;
    void this.ctx?.close().catch(() => undefined);
    this.ctx = null;
  }
}
