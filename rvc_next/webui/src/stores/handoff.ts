import { defineStore } from 'pinia';
import { ref } from 'vue';
import type { AudioRef, VoiceParams } from '@/api/types';

export interface HandoffInput {
  ref: AudioRef;
  name: string;
  duration?: number | null;
}

/**
 * Hand-offs between screens: "Use as input", "Convert this" (a stem) and "Re-run with these
 * settings" put something here, and the receiving screen takes it when it opens.
 */
export const useHandoffStore = defineStore('handoff', () => {
  const inputs = ref<Record<string, HandoffInput[]>>({});
  const voice = ref<{ id: string; params?: VoiceParams | null } | null>(null);

  function sendInputs(screen: string, items: HandoffInput[]) {
    inputs.value[screen] = [...(inputs.value[screen] ?? []), ...items];
  }
  function takeInputs(screen: string): HandoffInput[] {
    const items = inputs.value[screen] ?? [];
    delete inputs.value[screen];
    return items;
  }
  function sendVoice(id: string, params?: VoiceParams | null) {
    voice.value = { id, params };
  }
  function takeVoice() {
    const v = voice.value;
    voice.value = null;
    return v;
  }
  return { inputs, voice, sendInputs, takeInputs, sendVoice, takeVoice };
});
