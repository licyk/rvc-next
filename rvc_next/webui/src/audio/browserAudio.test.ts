import { createPinia, setActivePinia } from 'pinia';
import { describe, expect, it } from 'vitest';
import { browserAudioUrl, fromPcm16, toPcm16 } from '@/audio/browserAudio';

describe('browser audio', () => {
  it('sends 16-bit PCM and reads it back', () => {
    const x = new Float32Array([0, 0.5, -0.5, 1.5, -1.5]);
    const back = fromPcm16(toPcm16(x).buffer);
    expect(Array.from(back).map((v) => Math.round(v * 1000) / 1000)).toEqual([0, 0.5, -0.5, 1, -1]);
  });

  it('connects under the deployment root, with the token when there is one', () => {
    setActivePinia(createPinia());
    expect(browserAudioUrl('https://host:7868/voice', null)).toBe('wss://host:7868/voice/api/v1/live/browser-audio');
    expect(browserAudioUrl('http://127.0.0.1:7868', 'a b')).toBe('ws://127.0.0.1:7868/api/v1/live/browser-audio?token=a%20b');
  });
});
