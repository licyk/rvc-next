import { describe, expect, it } from 'vitest';
import { AudioPlayer } from '@/audio/player';

/** A media element stand-in: tracks src, time and paused state, and fires the events the player uses. */
class FakeMedia extends EventTarget {
  src = '';
  currentTime = 0;
  duration = 30;
  paused = true;
  readyState = 4;
  preload = '';
  async play() {
    this.paused = false;
    this.dispatchEvent(new Event('play'));
  }
  pause() {
    this.paused = true;
    this.dispatchEvent(new Event('pause'));
  }
  removeAttribute(name: string) {
    if (name === 'src') this.src = '';
  }
}

const track = { key: 'o1', url: '/out.wav', label: 'out', compareUrl: '/src.wav' };

describe('A/B player', () => {
  it('plays, toggles and switches sides at the same position', async () => {
    const el = new FakeMedia();
    const player = new AudioPlayer(el as unknown as HTMLMediaElement);
    await player.play(track);
    expect(el.src).toBe('/out.wav');
    expect(player.snapshot.playing).toBe(true);
    el.currentTime = 12.5;
    await player.setSide('b');
    expect(el.src).toBe('/src.wav');
    expect(el.currentTime).toBe(12.5);
    expect(player.snapshot).toMatchObject({ side: 'b', playing: true, time: 12.5 });
    await player.play(track, 'b');
    expect(player.snapshot.playing).toBe(false);
    await player.setSide('a');
    expect(el.src).toBe('/out.wav');
    expect(player.snapshot.playing).toBe(false);
  });

  it('plays one thing at a time and ignores B without a source', async () => {
    const el = new FakeMedia();
    const player = new AudioPlayer(el as unknown as HTMLMediaElement);
    await player.play(track);
    await player.play({ key: 'o2', url: '/two.wav', label: 'two' });
    expect(player.snapshot.track?.key).toBe('o2');
    expect(player.snapshot.time).toBe(0);
    await player.setSide('b');
    expect(player.snapshot.side).toBe('a');
    player.stop();
    expect(player.snapshot.track).toBeNull();
  });
});
