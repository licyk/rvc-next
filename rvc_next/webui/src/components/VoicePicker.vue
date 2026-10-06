<script setup lang="ts">
import { computed, watch } from 'vue';
import { useRouter } from 'vue-router';
import { useModels } from '@/api/queries/models';
import type { VoiceModel } from '@/api/types';
import { useI18n } from '@/i18n';
import { formatRate } from '@/format';
import { AppButton, PickerMenu, SelectField, icons, type PickerOption } from '@/ui';

/**
 * The only way to choose a voice: search by name or tag, with sample rate, version and
 * speakers shown, and voices without an index marked. Multi-speaker voices get a speaker menu.
 * The field and its list are ``PickerMenu``, the same as the audio device pickers.
 */
const props = withDefaults(defineProps<{ disabled?: boolean; label?: string }>(), { label: '' });
const model = defineModel<string | null>({ default: null });
const speaker = defineModel<number>('speaker', { default: 0 });
const emit = defineEmits<{ voice: [VoiceModel | null] }>();
const { t } = useI18n();
const router = useRouter();
const models = useModels();

const voices = computed(() => models.data.value ?? []);
const current = computed(() => voices.value.find((v) => v.id === model.value) ?? null);
const speakerOptions = computed(() => (current.value?.speakers ?? []).map((s) => ({ value: String(s.id), label: `${s.name} (${s.id})` })));
const speakerValue = computed({ get: () => String(speaker.value), set: (v) => (speaker.value = Number(v ?? 0)) });

watch(current, (v) => emit('voice', v), { immediate: true });
watch(current, (v, before) => {
  // A newly chosen voice starts at its first named speaker; a single-speaker voice at 0.
  if (v && v.id !== before?.id && (before !== null || !v.speakers.some((s) => s.id === speaker.value))) speaker.value = v.speakers[0]?.id ?? 0;
});

const describe = (v: VoiceModel) =>
  [formatRate(v.sample_rate), v.version, v.pitch_guidance ? t('voice.pitchGuidance') : t('voice.noPitchGuidance'), v.speakers.length ? t('voice.speakers', { n: v.speakers.length }) : ''].filter(Boolean).join(' · ');
const options = computed<PickerOption[]>(() =>
  voices.value.map((v) => ({
    value: v.id,
    label: v.name,
    description: describe(v),
    keywords: v.tags,
    badges: [...(v.has_index ? [] : [{ text: t('voice.noIndex'), tone: 'warning' as const }]), ...(v.legacy ? [{ text: t('voice.legacy'), tone: 'neutral' as const }] : [])],
  })),
);
</script>

<template>
  <div class="voice-picker">
    <div v-if="!models.isLoading.value && !voices.length" class="none">
      <span class="type-body-medium">{{ t('voice.none') }}</span>
      <span class="type-body-small muted">{{ t('voice.noneHint') }}</span>
      <AppButton variant="tonal" :icon="icons.Library" @click="router.push('/models')">{{ t('voice.goModels') }}</AppButton>
    </div>
    <template v-else>
      <PickerMenu
        v-model="model"
        :label="label || t('voice.label')"
        :options="options"
        :icon="icons.AudioLines"
        :placeholder="models.isLoading.value ? t('voice.loading') : t('voice.choose')"
        :loading="models.isLoading.value"
        :disabled="props.disabled"
        :search-label="t('voice.search')"
        :no-matches="t('common.noMatches')"
      />
      <SelectField v-if="speakerOptions.length" v-model="speakerValue" :label="t('voice.speaker')" :options="speakerOptions" :disabled="props.disabled" />
    </template>
  </div>
</template>

<style scoped>
.voice-picker { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.none { display: flex; flex-direction: column; align-items: flex-start; gap: var(--app-space-2); padding: var(--app-space-4); border-radius: var(--md-sys-shape-corner-medium); background: var(--md-sys-color-surface-container); }
</style>
