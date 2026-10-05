<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useModels } from '@/api/queries/models';
import type { ImportDecisions, ImportPlan, StagedFile } from '@/api/queries/imports';
import { useI18n } from '@/i18n';
import { formatBytes } from '@/format';
import { AppButton, AppDialog, Badge, SelectField, Switch, TextField } from '@/ui';

/**
 * Review an import the server could not settle on its own: which voice each index goes with
 * (only compatible voices are offered), names, and what to do with base and separation models.
 */
const props = defineProps<{ plan: ImportPlan | null; busy?: boolean }>();
const emit = defineEmits<{ commit: [ImportDecisions]; discard: [] }>();
const { t, tOr } = useI18n();
const library = useModels();

const open = computed({ get: () => !!props.plan, set: (v) => !v && emit('discard') });
const voices = ref<ImportPlan['voices']>([]);
const indexes = ref<ImportPlan['indexes']>([]);
const generators = ref<ImportPlan['generators']>([]);
const separations = ref<ImportPlan['separations']>([]);
watch(
  () => props.plan,
  (plan) => {
    // An editable copy; the plan is a reactive proxy, which structuredClone cannot copy.
    const copy = <T,>(v: T): T => JSON.parse(JSON.stringify(v));
    voices.value = copy(plan?.voices ?? []);
    indexes.value = copy(plan?.indexes ?? []);
    generators.value = copy(plan?.generators ?? []);
    separations.value = copy(plan?.separations ?? []);
  },
  { immediate: true },
);

const files = computed(() => Object.fromEntries((props.plan?.files ?? []).map((f) => [f.id, f])));
const baseName = (f: StagedFile | undefined) => (f ? f.name.split(/[\\/]/).pop() ?? f.name : '');
const other = computed(() => (props.plan?.files ?? []).filter((f) => f.kind === 'unsupported' || (f.kind === 'discriminator' && !generators.value.some((g) => g.discriminator === `file:${f.id}`))));

function describe(f: StagedFile | undefined): string {
  if (!f) return '';
  const parts: string[] = [];
  if (f.version) parts.push(f.version);
  if (f.sample_rate) parts.push(f.sample_rate);
  if (f.dim) parts.push(t('models.review.vectors', { n: (f.vectors ?? 0).toLocaleString() }));
  if (f.model_type) parts.push(f.model_type);
  parts.push(formatBytes(f.size));
  return parts.join(' · ');
}

function reason(code: string): string {
  if (code.startsWith('dimension:')) return t('models.review.reasons.dimension', { n: code.split(':')[1] });
  if (code.startsWith('no_') && code.endsWith('_voice')) return t('models.review.reasons.noVoice', { version: code.slice(3, 5) });
  return tOr(`models.review.reasons.${code}`, code);
}

function targetLabel(target: string, name: string): string {
  return target.startsWith('file:') ? t('models.review.newVoice', { name }) : t('models.review.libraryVoice', { name });
}
function indexOptions(i: ImportPlan['indexes'][number]) {
  return [{ value: '', label: t('models.review.keepUnassigned') }, ...i.candidates.map((c) => ({ value: c.target, label: targetLabel(c.target, c.name) }))];
}
function speakersOf(target: string | null | undefined): { id: number; name: string }[] {
  if (!target) return [];
  if (target.startsWith('file:')) return files.value[target.slice(5)]?.speakers ?? [];
  return (library.data.value ?? []).find((v) => `voice:${v.id}` === target)?.speakers ?? [];
}
function keyOptions(i: ImportPlan['indexes'][number]) {
  return [{ value: 'default', label: t('models.review.allSpeakers') }, ...speakersOf(i.target).map((s) => ({ value: `spk${s.id}`, label: `${s.id} · ${s.name}` }))];
}
function setTarget(i: ImportPlan['indexes'][number], value: string | null) {
  i.target = value || null;
  if (!speakersOf(i.target).some((s) => `spk${s.id}` === i.key)) i.key = 'default';
}
const statusTone = (s: string) => (({ unique: 'primary', evidence: 'primary', choose: 'warning', duplicate: 'warning', incompatible: 'error', empty: 'error' }) as Record<string, 'primary' | 'warning' | 'error'>)[s] ?? 'neutral';

const generatorActions = computed(() => (['base', 'extract', 'skip'] as const).map((v) => ({ value: v, label: t(`models.review.generator.${v}`) })));
function discOptions(g: ImportPlan['generators'][number]) {
  return [{ value: '', label: t('models.review.noDiscriminator') }, ...g.candidates.map((c) => ({ value: c, label: baseName(files.value[c.slice(5)]) }))];
}
function ckptOptions(s: ImportPlan['separations'][number]) {
  return [{ value: '', label: t('models.review.noCheckpoint') }, ...s.candidates.map((c) => ({ value: c, label: baseName(files.value[c]) }))];
}
function stemOptions(s: ImportPlan['separations'][number]) {
  return (files.value[s.config_id]?.instruments ?? []).map((i) => ({ value: i, label: i }));
}

function submit() {
  emit('commit', { voices: voices.value, indexes: indexes.value, generators: generators.value, separations: separations.value });
}
</script>

<template>
  <AppDialog v-model:open="open" :title="t('models.review.title')" width="large" :close-label="t('common.close')">
    <div v-if="plan" class="review">
      <p class="type-body-medium muted">{{ t('models.review.intro') }}</p>

      <section v-if="voices.length" class="block">
        <h3 class="type-title-small">{{ t('models.review.voices') }}</h3>
        <div v-for="v in voices" :key="v.file_id" class="item">
          <div class="item-text">
            <span class="type-body-medium name">{{ baseName(files[v.file_id]) }}</span>
            <span class="type-body-small muted">{{ describe(files[v.file_id]) }}</span>
          </div>
          <TextField v-model="v.name" :label="t('models.review.name')" :disabled="v.action === 'skip'" />
          <Switch class="toggle" :model-value="v.action === 'import'" :label="t('models.review.import')" @update:model-value="v.action = $event ? 'import' : 'skip'" />
        </div>
      </section>

      <section v-if="indexes.length" class="block">
        <h3 class="type-title-small">{{ t('models.review.indexes') }}</h3>
        <div v-for="i in indexes" :key="i.file_id" class="item">
          <div class="item-text">
            <span class="type-body-medium name">{{ baseName(files[i.file_id]) }}</span>
            <span class="type-body-small muted">{{ describe(files[i.file_id]) }}</span>
            <span class="reasons">
              <Badge :value="t(`models.review.status.${i.status}`)" :tone="statusTone(i.status)" />
              <span class="type-body-small muted">{{ i.reasons.map(reason).join(' · ') }}</span>
            </span>
          </div>
          <SelectField
            :model-value="i.target ?? ''"
            :label="t('models.review.voice')"
            :options="indexOptions(i)"
            :disabled="i.status === 'empty' || !i.candidates.length"
            @update:model-value="setTarget(i, $event)"
          />
          <SelectField v-if="speakersOf(i.target).length" v-model="i.key" :label="t('models.review.speaker')" :options="keyOptions(i)" />
        </div>
      </section>

      <section v-if="generators.length" class="block">
        <h3 class="type-title-small">{{ t('models.review.generators') }}</h3>
        <div v-for="g in generators" :key="g.file_id" class="item">
          <div class="item-text">
            <span class="type-body-medium name">{{ baseName(files[g.file_id]) }}</span>
            <span class="type-body-small muted">{{ describe(files[g.file_id]) }}</span>
          </div>
          <SelectField v-model="g.action" :label="t('models.review.useAs')" :options="generatorActions" />
          <TextField v-model="g.name" :label="t('models.review.name')" :disabled="g.action === 'skip'" />
          <SelectField
            v-if="g.action === 'base'"
            :model-value="g.discriminator ?? ''"
            :label="t('models.review.discriminator')"
            :options="discOptions(g)"
            @update:model-value="g.discriminator = $event || null"
          />
        </div>
      </section>

      <section v-if="separations.length" class="block">
        <h3 class="type-title-small">{{ t('models.review.separations') }}</h3>
        <div v-for="s in separations" :key="s.config_id" class="item sep">
          <div class="item-text">
            <span class="type-body-medium name">{{ baseName(files[s.config_id]) }}</span>
            <span class="type-body-small muted">{{ describe(files[s.config_id]) }}</span>
          </div>
          <SelectField :model-value="s.checkpoint_id ?? ''" :label="t('models.review.checkpoint')" :options="ckptOptions(s)" @update:model-value="s.checkpoint_id = $event || null" />
          <TextField v-model="s.name" :label="t('models.review.name')" />
          <SelectField v-model="s.primary" :label="t('models.review.primary')" :options="stemOptions(s)" />
          <TextField v-model="s.primary_label" :label="t('models.review.primaryLabel')" />
          <SelectField v-model="s.secondary" :label="t('models.review.secondary')" :options="stemOptions(s)" />
          <TextField v-model="s.secondary_label" :label="t('models.review.secondaryLabel')" />
          <Switch class="toggle" :model-value="s.action === 'import'" :label="t('models.review.import')" @update:model-value="s.action = $event ? 'import' : 'skip'" />
        </div>
      </section>

      <section v-if="other.length" class="block">
        <h3 class="type-title-small">{{ t('models.review.other') }}</h3>
        <p v-for="f in other" :key="f.id" class="type-body-small muted name">{{ baseName(f) }} — {{ f.note || t(`models.review.kinds.${f.kind}`) }}</p>
      </section>
    </div>
    <template #actions>
      <AppButton variant="text" @click="emit('discard')">{{ t('common.cancel') }}</AppButton>
      <AppButton :loading="busy" @click="submit">{{ t('models.review.commit') }}</AppButton>
    </template>
  </AppDialog>
</template>

<style scoped>
.review { display: flex; flex-direction: column; gap: var(--app-space-4); min-width: 0; }
.block { display: flex; flex-direction: column; gap: var(--app-space-2); min-width: 0; }
.block h3 { margin: 0; }
.item { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-3); padding: var(--app-space-3); border-radius: var(--md-sys-shape-corner-medium); background: var(--md-sys-color-surface-container); min-width: 0; }
.item > * { flex: 1 1 180px; min-width: 0; }
.item > .item-text { display: flex; flex-direction: column; flex: 2 1 240px; }
.item > .toggle { flex: none; }
.name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.reasons { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2); }
.muted { color: var(--md-sys-color-on-surface-variant); }
</style>
