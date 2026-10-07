<script setup lang="ts">
import { computed } from 'vue';
import { useEffectsCatalog } from '@/api/queries/convert';
import type { EffectModel, EffectSpec } from '@/api/types';
import { useI18n } from '@/i18n';
import { AppButton, AppMenu, IconButton, ParamSlider, TRANSITIONS, icons, type MenuItem } from '@/ui';

/** A chain of effects over the converted voice (pedalboard on the server), applied top to bottom. */
/** ``live``: offer only the effects that suit a stream. */
const props = defineProps<{ disabled?: boolean; live?: boolean }>();
const model = defineModel<EffectModel[]>({ required: true });
/** A finished change (a slider let go, an effect added, moved or removed): Live sends it on. */
const emit = defineEmits<{ change: [EffectModel[]] }>();
const { t, tOr } = useI18n();
const catalog = useEffectsCatalog();

const MAX = 16;
const specs = computed(() => new Map((catalog.data.value?.effects ?? []).map((s) => [s.kind, s])));
const available = computed(() => catalog.data.value?.available ?? true);
const addItems = computed<MenuItem[]>(() => (catalog.data.value?.effects ?? []).filter((s) => !props.live || s.live).map((s) => ({ id: s.kind, label: kindLabel(s.kind) })));
// Keys for the transition: an effect keeps its key when another moves.
let nextKey = 0;
const keys = new WeakMap<EffectModel, number>();
function keyOf(e: EffectModel) {
  if (!keys.has(e)) keys.set(e, nextKey++);
  return keys.get(e)!;
}

const kindLabel = (kind: string) => tOr(`effects.kind.${kind}`, kind);
const paramLabel = (name: string) => tOr(`effects.param.${name}`, name);
function value(e: EffectModel, spec: EffectSpec, name: string) {
  return e.params[name] ?? spec.params.find((p) => p.name === name)?.default ?? 0;
}
function step(min: number, max: number) {
  const span = max - min;
  return span <= 2 ? 0.01 : span <= 20 ? 0.1 : span <= 200 ? 1 : 10;
}
function commit(next: EffectModel[]) {
  model.value = next;
  emit('change', next);
}
function setParam(i: number, name: string, v: number, done: boolean) {
  const next = model.value.slice();
  const old = next[i]!;
  next[i] = { ...old, params: { ...old.params, [name]: v } };
  keys.set(next[i]!, keyOf(old));
  if (done) commit(next);
  else model.value = next;
}
function add(kind: string) {
  if (model.value.length >= MAX) return;
  commit([...model.value, { kind: kind as EffectModel['kind'], params: {} }]);
}
function move(i: number, by: -1 | 1) {
  const next = model.value.slice();
  const [e] = next.splice(i, 1);
  next.splice(i + by, 0, e!);
  commit(next);
}
function remove(i: number) {
  commit(model.value.filter((_, j) => j !== i));
}
</script>

<template>
  <section class="effects" :aria-label="t('effects.title')">
    <div class="head">
      <h3 class="type-title-small title">{{ t('effects.title') }}</h3>
      <AppMenu v-if="available" :items="addItems" align="start" @select="add">
        <template #default="{ toggle }">
          <AppButton variant="text" :icon="icons.Plus" :disabled="props.disabled || model.length >= MAX" @click="toggle">{{ t('effects.add') }}</AppButton>
        </template>
      </AppMenu>
    </div>
    <p v-if="!available" class="type-body-small muted">{{ t('effects.unavailable') }} <code>pip install rvc-next[effects]</code></p>
    <p v-else-if="!model.length" class="type-body-small muted">{{ t('effects.empty') }}</p>
    <TransitionGroup :name="TRANSITIONS.list" tag="ol" class="chain">
      <li v-for="(e, i) in model" :key="keyOf(e)" class="effect">
        <div class="effect-head">
          <span class="type-label-large">{{ i + 1 }}. {{ kindLabel(e.kind) }}</span>
          <span class="tools">
            <IconButton :icon="icons.ChevronUp" :label="t('effects.up')" :disabled="props.disabled || i === 0" @click="move(i, -1)" />
            <IconButton :icon="icons.ChevronDown" :label="t('effects.down')" :disabled="props.disabled || i === model.length - 1" @click="move(i, 1)" />
            <IconButton :icon="icons.Trash2" :label="t('effects.remove')" :disabled="props.disabled" @click="remove(i)" />
          </span>
        </div>
        <div v-if="specs.get(e.kind)" class="fields">
          <ParamSlider
            v-for="p in specs.get(e.kind)!.params"
            :key="p.name"
            :model-value="value(e, specs.get(e.kind)!, p.name)"
            :label="paramLabel(p.name)"
            :min="p.min"
            :max="p.max"
            :step="step(p.min, p.max)"
            :unit="p.unit"
            :default-value="p.default"
            :disabled="props.disabled"
            :reset-label="t('common.reset')"
            @update:model-value="setParam(i, p.name, $event, false)"
            @commit="setParam(i, p.name, $event, true)"
          />
        </div>
      </li>
    </TransitionGroup>
  </section>
</template>

<style scoped>
.effects { display: flex; flex-direction: column; gap: var(--app-space-2); min-width: 0; }
.head { display: flex; align-items: center; justify-content: space-between; gap: var(--app-space-2); }
.title { margin: 0; }
.muted { color: var(--md-sys-color-on-surface-variant); margin: 0; }
.chain { position: relative; list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--app-space-2); }
.effect { border: 1px solid var(--md-sys-color-outline-variant); border-radius: var(--md-sys-shape-corner-medium); padding: var(--app-space-2) var(--app-space-4) var(--app-space-3); }
.effect-head { display: flex; align-items: center; justify-content: space-between; gap: var(--app-space-2); }
.tools { display: flex; gap: var(--app-space-1); }
/* The sliders wrap: a wide page shows more of them per row, never longer ones. */
.fields { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(280px, 100%), 1fr)); gap: var(--app-space-2) var(--app-space-6); align-items: start; }
</style>
