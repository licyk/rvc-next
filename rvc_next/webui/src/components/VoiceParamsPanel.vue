<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { usePresetMutations, usePresets } from '@/api/queries/presets';
import type { Preset, VoiceModel, VoiceParams } from '@/api/types';
import { VOICE_FIELDS, spec } from '@/components/paramFields';
import { useI18n } from '@/i18n';
import { AppButton, AppDialog, AppMenu, Checkbox, ExpansionPanel, ParamSlider, SelectField, TextField, icons, useSnackbar, type MenuItem } from '@/ui';

/**
 * The one parameter panel: the same labels, ranges (from the server's schema),
 * defaults, help and order everywhere. ``live`` applies values while dragging (``change`` fires
 * then); otherwise values apply to the next job. Picking a voice loads its default preset.
 */
const props = withDefaults(defineProps<{ voice: VoiceModel | null; live?: boolean; disabled?: boolean }>(), { live: false });
const model = defineModel<VoiceParams>({ required: true });
const emit = defineEmits<{ change: [VoiceParams] }>();
const { t } = useI18n();
const snackbar = useSnackbar();
const presets = usePresets(() => props.voice?.id ?? null);
const mutations = usePresetMutations();
const loadedFor = ref<string | null>(null);

function setField(key: keyof VoiceParams, value: number | string, commit = true) {
  const next = { ...model.value, [key]: value } as VoiceParams;
  model.value = next;
  if (commit || props.live) emit('change', next);
}

// A newly picked voice sounds the same everywhere: load its default preset, keeping the speaker.
watch(
  () => [props.voice?.id, presets.data.value] as const,
  ([id, list]) => {
    if (!id || !list || loadedFor.value === id) return;
    const def = list.find((p) => p.voice_id === id && p.is_default);
    loadedFor.value = id;
    if (def) apply(def, false);
  },
  { immediate: true },
);

function apply(p: Preset, notify = true) {
  const next = { ...p.params, speaker_id: model.value.speaker_id } as VoiceParams;
  model.value = next;
  emit('change', next);
  if (notify) snackbar.show(p.name);
}

const noIndex = computed(() => !!props.voice && !props.voice.has_index);
const noPitch = computed(() => !!props.voice && !props.voice.pitch_guidance);
const fields = computed(() =>
  VOICE_FIELDS.map((f) => {
    const s = spec('voice', f.key);
    const disabledReason =
      f.key === 'index_rate' && noIndex.value
        ? t('params.noIndex')
        : f.key === 'protect' && noPitch.value
          ? t('params.noPitch')
          : f.key === 'protect' && model.value.unvoiced === 'original'
            ? t('params.protectOff')
            : '';
    return { ...f, min: s.min, max: s.max, default: s.default as number, disabledReason };
  }),
);
const f0Options = computed(() => (spec('voice', 'f0_method').options ?? ['rmvpe']).map((v) => ({ value: v, label: t(`params.f0.${v}`) })));
const f0 = computed({ get: () => model.value.f0_method as string | null, set: (v) => v && setField('f0_method', v) });

/** The choices under "Pitch details", options from the schema; some apply offline only, or to one pitch method. */
const CHOICES = [
  { key: 'unvoiced', label: 'params.unvoiced', help: 'params.unvoicedHelp', options: 'params.unvoicedOptions', offlineOnly: false, method: null },
  { key: 'f0_high_register', label: 'params.highRegister', help: 'params.highRegisterHelp', options: 'params.highRegisterOptions', offlineOnly: true, method: 'rmvpe' },
] as const;
const choices = computed(() =>
  CHOICES.filter((c) => !(c.offlineOnly && props.live)).map((c) => ({
    ...c,
    options: (spec('voice', c.key).options ?? []).map((v) => ({ value: v, label: t(`${c.options}.${v}`) })),
    off: !!c.method && model.value.f0_method !== c.method,
  })),
);
const ceiling = computed(() => {
  const s = spec('voice', 'f0_ceiling');
  return { min: s.min, max: s.max, default: s.default as number };
});
const showCeiling = computed(() => !props.live && model.value.f0_high_register === 'true_pitch' && model.value.f0_method === 'rmvpe');
const detailsOpen = ref(false);

const menuItems = computed<MenuItem[]>(() => [
  ...(presets.data.value ?? []).map((p) => ({ id: p.id, label: `${p.name}${p.is_default ? ` · ${t('params.defaultPreset')}` : p.voice_id ? '' : ` · ${t('params.global')}`}` })),
  { id: '__save', label: t('params.savePreset'), icon: icons.Save },
]);

const saveOpen = ref(false);
const saveName = ref('');
const saveDefault = ref(false);
const saveForVoice = ref(true);
function onMenu(id: string) {
  if (id === '__save') {
    saveName.value = '';
    saveDefault.value = false;
    saveForVoice.value = !!props.voice;
    saveOpen.value = true;
    return;
  }
  const p = presets.data.value?.find((x) => x.id === id);
  if (p) apply(p);
}
function save() {
  const name = saveName.value.trim();
  if (!name) return;
  mutations.create.mutate(
    { name, voice_id: saveForVoice.value ? (props.voice?.id ?? null) : null, params: { ...model.value }, is_default: saveForVoice.value && saveDefault.value },
    { onSuccess: () => ((saveOpen.value = false), snackbar.show(t('params.presetSaved', { name }))), onError: (e) => snackbar.error(e.message) },
  );
}
</script>

<template>
  <section class="params" :aria-label="t('params.title')">
    <header class="head">
      <h2 class="type-title-medium title">{{ t('params.title') }}</h2>
      <AppMenu :items="menuItems" @select="onMenu">
        <template #default="{ toggle }">
          <AppButton variant="text" :icon="icons.SlidersHorizontal" :disabled="disabled" @click="toggle">{{ t('params.presets') }}</AppButton>
        </template>
      </AppMenu>
    </header>
    <p class="type-body-small muted note">{{ live ? t('params.appliesLive') : t('params.appliesNext') }}</p>
    <div class="f0">
      <SelectField v-model="f0" :label="t('params.f0Method')" :options="f0Options" :supporting-text="t('params.f0Help')" :class="{ off: noPitch }" />
    </div>
    <div class="fields">
      <ParamSlider
        v-for="f in fields"
        :key="f.key"
        :model-value="model[f.key] as number"
        :label="t(f.label)"
        :help="t(f.help)"
        :min="f.min"
        :max="f.max"
        :step="f.step"
        :unit="f.unit"
        :default-value="f.default"
        :disabled="disabled || !!f.disabledReason"
        :disabled-reason="f.disabledReason"
        :reset-label="t('common.reset')"
        @update:model-value="setField(f.key, $event, false)"
        @commit="setField(f.key, $event)"
      />
    </div>
    <ExpansionPanel v-model:open="detailsOpen" :label="t('params.more')" :disabled="noPitch">
      <div class="fields">
        <SelectField
          v-for="c in choices"
          :key="c.key"
          :model-value="model[c.key] as string"
          :label="t(c.label)"
          :supporting-text="c.off ? t('params.onlyFor', { method: t(`params.f0.${c.method}`) }) : t(c.help)"
          :options="c.options"
          :disabled="disabled || c.off"
          @update:model-value="$event && setField(c.key, $event)"
        />
        <ParamSlider
          v-if="showCeiling"
          :model-value="model.f0_ceiling"
          :label="t('params.ceiling')"
          :help="t('params.ceilingHelp')"
          :min="ceiling.min"
          :max="ceiling.max"
          :step="10"
          unit="Hz"
          :default-value="ceiling.default"
          :disabled="disabled"
          :reset-label="t('common.reset')"
          @update:model-value="setField('f0_ceiling', $event, false)"
          @commit="setField('f0_ceiling', $event)"
        />
      </div>
    </ExpansionPanel>
    <AppDialog v-model:open="saveOpen" :title="t('params.savePreset')" width="small" :close-label="t('common.close')">
      <TextField v-model="saveName" :label="t('params.presetName')" @enter="save" />
      <Checkbox v-if="voice" v-model="saveForVoice" :label="t('params.forVoice')" />
      <Checkbox v-if="voice && saveForVoice" v-model="saveDefault" :label="t('params.makeDefault')" />
      <template #actions>
        <AppButton variant="text" @click="saveOpen = false">{{ t('common.cancel') }}</AppButton>
        <AppButton :disabled="!saveName.trim()" :loading="mutations.create.isPending.value" @click="save">{{ t('common.save') }}</AppButton>
      </template>
    </AppDialog>
  </section>
</template>

<style scoped>
.params { display: flex; flex-direction: column; gap: var(--app-space-2); min-width: 0; }
.head { display: flex; align-items: center; gap: var(--app-space-2); }
.title { flex: 1; margin: 0; }
.note { margin: 0; }
.f0 { display: grid; grid-template-columns: minmax(min(320px, 100%), max-content); }
.off { opacity: var(--app-disabled-content-opacity); }
/* The sliders wrap: a wide page shows more of them per row, never longer ones. */
.fields { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(320px, 100%), 1fr)); gap: var(--app-space-2) var(--app-space-6); align-items: start; }
</style>
