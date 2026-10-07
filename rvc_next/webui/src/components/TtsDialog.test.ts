import { flushPromises, mount } from '@vue/test-utils';
import { QueryClient, VueQueryPlugin } from '@tanstack/vue-query';
import { createPinia } from 'pinia';
import { describe, expect, it, vi } from 'vitest';
import TtsDialog from '@/components/TtsDialog.vue';

vi.mock('@/api/client', () => ({
  api: {
    GET: vi.fn(async () => ({
      data: {
        available: true,
        voices: [
          { id: 'en-US-AriaNeural', name: 'Aria', locale: 'en-US', language: 'English (United States)', gender: 'Female' },
          { id: 'zh-CN-XiaoxiaoNeural', name: 'Xiaoxiao', locale: 'zh-CN', language: 'Chinese (Mainland)', gender: 'Female' },
        ],
      },
    })),
  },
  unwrap: async (p: Promise<{ data: unknown }>) => (await p).data,
}));

describe('TtsDialog', () => {
  it('picks the interface language’s voice and emits the request', async () => {
    const w = mount(TtsDialog, { props: { open: true }, global: { plugins: [createPinia(), [VueQueryPlugin, { queryClient: new QueryClient() }]] }, attachTo: document.body });
    await flushPromises();
    const area = document.querySelector('textarea')!;
    area.value = '  Hello  ';
    area.dispatchEvent(new Event('input'));
    await flushPromises();
    const add = [...document.querySelectorAll('button, md-filled-button')].find((b) => b.textContent?.trim() === 'Add') as HTMLElement;
    add.click();
    await flushPromises();
    expect(w.emitted('add')![0]![0]).toEqual({ text: 'Hello', voice: 'en-US-AriaNeural', rate: 0, pitch: 0 });
    w.unmount();
  });
});
