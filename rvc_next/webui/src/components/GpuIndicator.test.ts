import { mount } from '@vue/test-utils';
import { createPinia } from 'pinia';
import { describe, expect, it, vi } from 'vitest';
import { ref } from 'vue';
import GpuIndicator from '@/components/GpuIndicator.vue';
import AppMenu from '@/ui/AppMenu.vue';

const usage = ref<Record<string, unknown>>({});
vi.mock('@/api/queries/compute', () => ({
  useComputeUsage: () => ({ data: usage }),
  useReleaseCompute: () => ({ mutate: vi.fn(), isPending: ref(false) }),
}));

const freeItem = () => mount(GpuIndicator, { global: { plugins: [createPinia()] } }).findComponent(AppMenu).props('items')[0];

describe('GpuIndicator', () => {
  it('offers Free GPU memory whenever nothing is working, even with nothing listed as loaded', () => {
    usage.value = { device: 'cuda:0', cached: [], busy: [], can_release: true, lease_holder: null };
    expect(freeItem()).toMatchObject({ id: 'free', disabled: false, label: 'Free GPU memory' });
  });

  it('says what to wait for while a job or Live runs', () => {
    usage.value = { device: 'cuda:0', cached: ['hubert'], busy: ['jobs', 'live'], can_release: false, lease_holder: 'live' };
    expect(freeItem()).toMatchObject({ disabled: true, label: 'Free GPU memory (wait: a job is running, Live is running)' });
  });
});
