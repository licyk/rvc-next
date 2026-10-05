<script setup lang="ts">
import { computed, ref } from 'vue';
import AppIcon from '@/ui/AppIcon.vue';
import Badge from '@/ui/Badge.vue';
import { Check, ChevronDown } from '@/ui/icons';
import { useRevealWhileOpen } from '@/ui/reveal';
import { useLayer } from '@/ui/useLayer';

export interface DeviceOption {
  value: string;
  label: string;
  /** Short tags shown after the name: "default", "virtual", "not connected". */
  badges?: { text: string; tone?: 'primary' | 'warning' | 'neutral' | 'error' }[];
  /** A saved device that is not connected: listed, greyed, not selectable. */
  unavailable?: boolean;
}

/** A device picker: a field showing the chosen device and its badges, opening a list. */
const props = defineProps<{ label: string; options: DeviceOption[]; placeholder?: string; disabled?: boolean }>();
const model = defineModel<string | null>({ default: null });
const open = ref(false);
const root = ref<HTMLElement | null>(null);
const list = ref<HTMLElement | null>(null);
const current = computed(() => props.options.find((o) => o.value === model.value) ?? null);
useLayer(() => open.value, () => (open.value = false));
useRevealWhileOpen(() => open.value, list);

function pick(o: DeviceOption) {
  if (o.unavailable) return;
  model.value = o.value;
  open.value = false;
}
function onBlur(e: FocusEvent) {
  if (!root.value?.contains(e.relatedTarget as Node | null)) open.value = false;
}
</script>

<template>
  <div ref="root" class="device-menu" @focusout="onBlur">
    <button type="button" class="field state-layer" :disabled="disabled" :aria-expanded="open" aria-haspopup="listbox" @click="open = !open">
      <span class="type-body-small muted caption">{{ label }}</span>
      <span class="row">
        <span class="type-body-large name" :class="{ muted: !current }">{{ current?.label ?? placeholder ?? '' }}</span>
        <Badge v-for="b in current?.badges ?? []" :key="b.text" :value="b.text" :tone="b.tone ?? 'neutral'" />
        <AppIcon :icon="ChevronDown" :size="20" class="chevron" :class="{ open }" />
      </span>
    </button>
    <Transition name="scrim">
      <ul v-if="open" ref="list" class="list" role="listbox" :aria-label="label">
        <li v-for="o in options" :key="o.value">
          <button type="button" role="option" class="option state-layer" :class="{ unavailable: o.unavailable }" :aria-selected="o.value === model" :aria-disabled="o.unavailable" @click="pick(o)">
            <AppIcon class="check" :icon="Check" :size="18" :style="{ visibility: o.value === model ? 'visible' : 'hidden' }" />
            <span class="type-body-large label">{{ o.label }}</span>
            <Badge v-for="b in o.badges ?? []" :key="b.text" :value="b.text" :tone="b.tone ?? 'neutral'" />
          </button>
        </li>
      </ul>
    </Transition>
  </div>
</template>

<style scoped>
.device-menu { position: relative; min-width: 0; }
.field {
  display: flex; flex-direction: column; align-items: stretch; gap: 2px; width: 100%; min-width: 0; padding: var(--app-space-2) var(--app-space-3);
  border: 1px solid var(--md-sys-color-outline); border-radius: var(--md-sys-shape-corner-extra-small); background: transparent; color: var(--md-sys-color-on-surface); font: inherit; text-align: start; cursor: pointer;
}
.field:disabled { opacity: var(--app-disabled-content-opacity); cursor: default; }
.caption { line-height: 1; }
.row { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.name { flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.list {
  position: absolute; z-index: var(--app-z-dropdown); top: calc(100% + 4px); left: 0; right: 0; margin: 0; padding: var(--app-space-1) 0; list-style: none;
  max-height: 320px; overflow: auto; background: var(--md-sys-color-surface-container); border-radius: var(--md-sys-shape-corner-extra-small); box-shadow: var(--app-elevation-2);
}
.option { display: flex; align-items: center; gap: var(--app-space-2); width: 100%; padding: var(--app-space-2) var(--app-space-3); border: 0; background: transparent; color: var(--md-sys-color-on-surface); font: inherit; text-align: start; cursor: pointer; min-width: 0; }
.option.unavailable { opacity: var(--app-disabled-content-opacity); cursor: default; }
.label { flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.check { flex: none; color: var(--md-sys-color-primary); }
.chevron { flex: none; transition: transform var(--md-sys-motion-duration-short4) var(--md-sys-motion-easing-standard); }
.chevron.open { transform: rotate(180deg); }
</style>
