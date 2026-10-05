<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useRouter } from 'vue-router';
import { useModels } from '@/api/queries/models';
import type { VoiceModel } from '@/api/types';
import { useI18n } from '@/i18n';
import { formatRate } from '@/format';
import { AppButton, AppIcon, Badge, SelectField, TextField, icons, useLayer, useRevealWhileOpen } from '@/ui';

/**
 * The only way to choose a voice: search by name, with sample rate, version and
 * speakers shown, and voices without an index marked. Multi-speaker voices get a speaker menu.
 */
const props = withDefaults(defineProps<{ disabled?: boolean; label?: string }>(), { label: '' });
const model = defineModel<string | null>({ default: null });
const speaker = defineModel<number>('speaker', { default: 0 });
const emit = defineEmits<{ voice: [VoiceModel | null] }>();
const { t } = useI18n();
const router = useRouter();
const models = useModels();
const open = ref(false);
const query = ref('');
const root = ref<HTMLElement | null>(null);
const popover = ref<HTMLElement | null>(null);
useLayer(() => open.value, () => (open.value = false));
useRevealWhileOpen(() => open.value, popover);

const voices = computed(() => models.data.value ?? []);
const current = computed(() => voices.value.find((v) => v.id === model.value) ?? null);
const filtered = computed(() => {
  const q = query.value.trim().toLowerCase();
  return q ? voices.value.filter((v) => v.name.toLowerCase().includes(q) || v.tags.some((tag) => tag.toLowerCase().includes(q))) : voices.value;
});
const speakerOptions = computed(() => (current.value?.speakers ?? []).map((s) => ({ value: String(s.id), label: `${s.name} (${s.id})` })));
const speakerValue = computed({ get: () => String(speaker.value), set: (v) => (speaker.value = Number(v ?? 0)) });

watch(current, (v) => emit('voice', v), { immediate: true });
watch(current, (v, before) => {
  // A newly chosen voice starts at its first named speaker; a single-speaker voice at 0.
  if (v && v.id !== before?.id && (before !== null || !v.speakers.some((s) => s.id === speaker.value))) speaker.value = v.speakers[0]?.id ?? 0;
});

function pick(v: VoiceModel) {
  model.value = v.id;
  open.value = false;
  query.value = '';
}
const describe = (v: VoiceModel) =>
  [formatRate(v.sample_rate), v.version, v.pitch_guidance ? t('voice.pitchGuidance') : t('voice.noPitchGuidance'), v.speakers.length ? t('voice.speakers', { n: v.speakers.length }) : ''].filter(Boolean).join(' · ');
function onBlur(e: FocusEvent) {
  if (!root.value?.contains(e.relatedTarget as Node | null)) open.value = false;
}
</script>

<template>
  <div ref="root" class="voice-picker" @focusout="onBlur">
    <div v-if="!models.isLoading.value && !voices.length" class="none">
      <span class="type-body-medium">{{ t('voice.none') }}</span>
      <span class="type-body-small muted">{{ t('voice.noneHint') }}</span>
      <AppButton variant="tonal" :icon="icons.Library" @click="router.push('/models')">{{ t('voice.goModels') }}</AppButton>
    </div>
    <template v-else>
      <button type="button" class="field state-layer" :disabled="props.disabled" :aria-expanded="open" aria-haspopup="listbox" @click="open = !open">
        <AppIcon :icon="icons.AudioLines" :size="24" />
        <span class="text">
          <span class="type-body-small muted">{{ label || t('voice.label') }}</span>
          <span class="type-title-medium name">{{ current?.name ?? t('voice.choose') }}</span>
          <span v-if="current" class="type-body-small muted meta">{{ describe(current) }}</span>
        </span>
        <Badge v-if="current && !current.has_index" :value="t('voice.noIndex')" tone="warning" />
        <AppIcon :icon="icons.ChevronDown" :size="20" class="chevron" :class="{ open }" />
      </button>
      <Transition name="scrim">
        <div v-if="open" ref="popover" class="popover" role="dialog" :aria-label="t('voice.choose')">
          <TextField v-model="query" :label="t('voice.search')" :icon="icons.Search" type="search" />
          <ul class="list" role="listbox">
            <li v-for="v in filtered" :key="v.id">
              <button type="button" role="option" class="option state-layer" :aria-selected="v.id === model" @click="pick(v)">
                <span class="text">
                  <span class="type-body-large name">{{ v.name }}</span>
                  <span class="type-body-small muted meta">{{ describe(v) }}</span>
                </span>
                <Badge v-if="!v.has_index" :value="t('voice.noIndex')" tone="warning" />
                <Badge v-if="v.legacy" :value="t('voice.legacy')" tone="neutral" />
              </button>
            </li>
          </ul>
        </div>
      </Transition>
      <SelectField v-if="speakerOptions.length" v-model="speakerValue" :label="t('voice.speaker')" :options="speakerOptions" :disabled="props.disabled" />
    </template>
  </div>
</template>

<style scoped>
.voice-picker { position: relative; display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.field { display: flex; align-items: center; gap: var(--app-space-3); width: 100%; min-width: 0; padding: var(--app-space-3) var(--app-space-4); border: 1px solid var(--md-sys-color-outline-variant); border-radius: var(--md-sys-shape-corner-medium); background: var(--md-sys-color-surface-container); color: var(--md-sys-color-on-surface); font: inherit; text-align: start; cursor: pointer; }
.field:disabled { opacity: var(--app-disabled-content-opacity); cursor: default; }
.text { display: flex; flex-direction: column; flex: 1; min-width: 0; }
.name, .meta { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.popover { position: absolute; z-index: var(--app-z-dropdown); top: calc(100% + 4px); left: 0; right: 0; display: flex; flex-direction: column; gap: var(--app-space-2); padding: var(--app-space-3); background: var(--md-sys-color-surface-container); border-radius: var(--md-sys-shape-corner-medium); box-shadow: var(--app-elevation-2); }
.list { margin: 0; padding: 0; list-style: none; max-height: 320px; overflow: auto; }
.option { display: flex; align-items: center; gap: var(--app-space-2); width: 100%; padding: var(--app-space-2) var(--app-space-3); border: 0; border-radius: var(--md-sys-shape-corner-small); background: transparent; color: var(--md-sys-color-on-surface); font: inherit; text-align: start; cursor: pointer; min-width: 0; }
.option[aria-selected='true'] { background: var(--md-sys-color-secondary-container); color: var(--md-sys-color-on-secondary-container); }
.chevron { flex: none; transition: transform var(--md-sys-motion-duration-short4) var(--md-sys-motion-easing-standard); }
.chevron.open { transform: rotate(180deg); }
.none { display: flex; flex-direction: column; align-items: flex-start; gap: var(--app-space-2); padding: var(--app-space-4); border-radius: var(--md-sys-shape-corner-medium); background: var(--md-sys-color-surface-container); }
</style>
