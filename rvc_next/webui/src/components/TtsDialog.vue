<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useTtsVoices } from '@/api/queries/audio';
import type { S } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import { useI18n } from '@/i18n';
import { usePreferencesStore } from '@/stores/preferences';
import { AppButton, AppDialog, ParamSlider, SelectField } from '@/ui';

/** Speech from text with an online voice (Microsoft Edge's, through edge-tts on the server), added as an input. */
const open = defineModel<boolean>('open', { default: false });
const emit = defineEmits<{ add: [S['TtsRequest']] }>();
const { t, locale } = useI18n();
const prefs = usePreferencesStore();
const voices = useTtsVoices(open);

const text = ref('');
const language = ref<string | null>(null);
const voice = ref<string | null>(null);
const rate = ref(0);
const pitch = ref(0);

const list = computed(() => voices.data.value?.voices ?? []);
const available = computed(() => voices.data.value?.available ?? true);
const languageOptions = computed(() => {
  const seen = new Map<string, string>();
  for (const v of list.value) if (!seen.has(v.locale)) seen.set(v.locale, `${v.language} (${v.locale})`);
  return [...seen].map(([value, label]) => ({ value, label })).sort((a, b) => a.label.localeCompare(b.label));
});
const voiceOptions = computed(() => list.value.filter((v) => v.locale === language.value).map((v) => ({ value: v.id, label: `${v.name} · ${v.gender}` })));

// The last voice, else the interface language's usual one.
watch(
  list,
  (all) => {
    if (!all.length || voice.value) return;
    const usual = locale.value === 'zh-CN' ? 'zh-CN-XiaoxiaoNeural' : 'en-US-AriaNeural';
    const pick = all.find((v) => v.id === prefs.prefs.ttsVoice) ?? all.find((v) => v.id === usual) ?? all.find((v) => v.locale === usual.slice(0, 5)) ?? all[0]!;
    language.value = pick.locale;
    voice.value = pick.id;
  },
  { immediate: true },
);
watch(language, (l) => {
  if (!list.value.some((v) => v.id === voice.value && v.locale === l)) voice.value = list.value.find((v) => v.locale === l)?.id ?? null;
});

const canAdd = computed(() => !!voice.value && text.value.trim().length > 0);
function add() {
  if (!canAdd.value || !voice.value) return;
  prefs.prefs.ttsVoice = voice.value;
  emit('add', { text: text.value.trim(), voice: voice.value, rate: rate.value, pitch: pitch.value });
  text.value = '';
  open.value = false;
}
</script>

<template>
  <AppDialog v-model:open="open" :title="t('tts.title')" width="medium" :close-label="t('common.close')">
    <div class="tts">
      <p v-if="!available" class="type-body-medium">{{ t('tts.unavailable') }} <code>pip install rvc-next[tts]</code></p>
      <template v-else>
        <ErrorNotice v-if="voices.error.value" :error="voices.error.value" />
        <textarea v-model="text" class="type-body-large text" rows="5" maxlength="5000" :placeholder="t('tts.placeholder')" :aria-label="t('tts.text')" />
        <div class="fields">
          <SelectField v-model="language" :label="t('tts.language')" :options="languageOptions" :disabled="!list.length" />
          <SelectField v-model="voice" :label="t('tts.voice')" :options="voiceOptions" :disabled="!list.length" />
          <ParamSlider v-model="rate" :label="t('tts.rate')" :min="-50" :max="100" unit="%" :default-value="0" :reset-label="t('common.reset')" />
          <ParamSlider v-model="pitch" :label="t('tts.pitch')" :min="-50" :max="50" unit="Hz" :default-value="0" :reset-label="t('common.reset')" />
        </div>
        <p class="type-body-small muted">{{ t('tts.privacy') }}</p>
      </template>
    </div>
    <template #actions>
      <AppButton variant="text" @click="open = false">{{ t('common.cancel') }}</AppButton>
      <AppButton :disabled="!available || !canAdd" @click="add">{{ t('tts.add') }}</AppButton>
    </template>
  </AppDialog>
</template>

<style scoped>
.tts { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.text {
  box-sizing: border-box;
  width: 100%;
  resize: vertical;
  padding: var(--app-space-3);
  border: 1px solid var(--md-sys-color-outline);
  border-radius: var(--md-sys-shape-corner-extra-small);
  background: transparent;
  color: var(--md-sys-color-on-surface);
  font-family: inherit;
}
.text:focus { outline: 2px solid var(--md-sys-color-primary); outline-offset: -1px; }
.fields { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(240px, 100%), 1fr)); gap: var(--app-space-3) var(--app-space-4); align-items: center; }
.muted { color: var(--md-sys-color-on-surface-variant); margin: 0; }
</style>
