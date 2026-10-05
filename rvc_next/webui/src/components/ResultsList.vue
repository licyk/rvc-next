<script setup lang="ts">
import { computed } from 'vue';
import { useMeta } from '@/api/queries/app';
import { revealOutput, useDeleteOutput } from '@/api/queries/outputs';
import { downloadOutputsZip, downloadUrl, urls } from '@/api/client';
import type { Output } from '@/api/types';
import type { Track } from '@/audio/player';
import ABCompare from '@/components/ABCompare.vue';
import ResultWave from '@/components/ResultWave.vue';
import { useI18n } from '@/i18n';
import { formatDuration } from '@/format';
import { usePlayerStore } from '@/stores/player';
import { AppButton, AppMenu, Badge, EmptyState, IconButton, TRANSITIONS, icons, staggerStyle, useSnackbar, type MenuItem } from '@/ui';

/**
 * The one results pattern for conversion and separation: a waveform, play, A/B
 * against the source, download, reveal (on the server's machine), "Use as input" and "Re-run with
 * these settings". ``groupBySource`` groups stems under their input.
 */
const props = withDefaults(defineProps<{ outputs: Output[]; groupBySource?: boolean; emptyText?: string }>(), { groupBySource: false, emptyText: '' });
const emit = defineEmits<{ useAsInput: [Output]; rerun: [Output]; convertThis: [Output] }>();
const { t, tOr } = useI18n();
const meta = useMeta();
const player = usePlayerStore();
const remove = useDeleteOutput();
const snackbar = useSnackbar();

const groups = computed(() => {
  if (!props.groupBySource) return [{ key: 'all', title: '', items: props.outputs }];
  const map = new Map<string, Output[]>();
  for (const o of props.outputs) {
    const k = o.source_name ?? o.job_id ?? o.id;
    map.set(k, [...(map.get(k) ?? []), o]);
  }
  return [...map.entries()].map(([key, items]) => ({ key, title: key, items }));
});

const trackOf = (o: Output): Track => ({ key: o.id, url: urls.output(o.id), label: o.name, compareUrl: o.source_path ? urls.outputSource(o.id) : null });
const progressOf = (o: Output) => (player.state.track?.key === o.id && player.state.duration ? player.state.time / player.state.duration : 0);
function seek(o: Output, fraction: number) {
  if (player.state.track?.key !== o.id) player.play(trackOf(o)).then(() => player.seek(fraction * (o.duration ?? 0)));
  else player.seek(fraction * (player.state.duration || o.duration || 0));
}

const menu = (o: Output): MenuItem[] => [
  { id: 'download', label: t('common.download'), icon: icons.Download },
  ...(meta.data.value?.local ? [{ id: 'reveal', label: t('common.reveal'), icon: icons.FolderOpen }] : []),
  { id: 'input', label: t('results.useAsInput'), icon: icons.ArrowRight },
  ...(o.kind === 'stem' ? [{ id: 'convert', label: t('results.convertThis'), icon: icons.AudioLines }] : []),
  ...(o.model_id && o.voice ? [{ id: 'rerun', label: t('results.rerun'), icon: icons.Repeat }] : []),
  { id: 'delete', label: t('common.delete'), icon: icons.Trash2, danger: true },
];
function onMenu(o: Output, id: string) {
  if (id === 'download') downloadUrl(urls.output(o.id, true));
  else if (id === 'reveal') revealOutput(o.id).catch((e) => snackbar.error(e.message));
  else if (id === 'input') emit('useAsInput', o);
  else if (id === 'convert') emit('convertThis', o);
  else if (id === 'rerun') emit('rerun', o);
  else if (id === 'delete') {
    if (player.state.track?.key === o.id) player.stop();
    remove.mutate(o.id, { onSuccess: () => snackbar.show(t('results.deleted')) });
  }
}
function downloadAll() {
  downloadOutputsZip(props.outputs.filter((o) => o.exists).map((o) => o.id)).catch((e) => snackbar.error(e.message));
}
</script>

<template>
  <section class="results" :aria-label="t('results.title')">
    <header class="head">
      <h2 class="type-title-medium title">{{ t('results.title') }}</h2>
      <AppButton v-if="outputs.length > 1" variant="text" :icon="icons.Download" @click="downloadAll">{{ t('results.downloadAll') }}</AppButton>
    </header>
    <EmptyState v-if="!outputs.length" :icon="icons.AudioWaveform" :title="emptyText || t('results.empty')" />
    <div v-for="g in groups" :key="g.key" class="group">
      <h3 v-if="g.title" class="type-title-small group-title">{{ g.title }}</h3>
      <TransitionGroup :name="TRANSITIONS.list" tag="ul" class="list">
        <li v-for="(o, i) in g.items" :key="o.id" class="row" :class="{ gone: !o.exists }" :style="staggerStyle(i)">
          <IconButton :icon="player.isPlaying(o.id) ? icons.Pause : icons.Play" :label="player.isPlaying(o.id) ? t('results.pause') : t('results.play')" :disabled="!o.exists" @click="player.play(trackOf(o))" />
          <div class="main">
            <div class="line">
              <span class="type-body-large name">{{ o.name }}</span>
              <Badge :value="tOr(`results.kinds.${o.kind}`, o.kind)" :tone="o.kind === 'preview' ? 'warning' : 'neutral'" />
              <span v-if="o.kind === 'stem'" class="type-label-medium muted">{{ o.label }}</span>
              <span class="type-body-small muted dur">{{ o.duration ? formatDuration(o.duration) : '' }}</span>
            </div>
            <ResultWave v-if="o.exists" :output="o" :progress="progressOf(o)" @seek="seek(o, $event)" />
            <span v-else class="type-body-small muted">{{ t('results.missing') }}</span>
            <ABCompare v-if="o.source_path && o.exists" :track="trackOf(o)" class="ab" />
          </div>
          <AppMenu :items="menu(o)" @select="onMenu(o, $event)">
            <template #default="{ toggle }"><IconButton :icon="icons.MoreVertical" :label="t('common.actions')" @click="toggle" /></template>
          </AppMenu>
        </li>
      </TransitionGroup>
    </div>
  </section>
</template>

<style scoped>
.results { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.head { display: flex; align-items: center; gap: var(--app-space-2); }
.title { flex: 1; margin: 0; }
.group { display: flex; flex-direction: column; gap: var(--app-space-2); }
.group-title { margin: var(--app-space-2) 0 0; overflow-wrap: anywhere; }
.list { position: relative; list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--app-space-2); }
.row { display: flex; align-items: center; gap: var(--app-space-2); padding: var(--app-space-2); border-radius: var(--md-sys-shape-corner-medium); background: var(--md-sys-color-surface-container); min-width: 0; }
.row.gone { opacity: var(--app-disabled-content-opacity); }
.main { display: flex; flex-direction: column; gap: var(--app-space-1); flex: 1; min-width: 0; }
.line { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ab { align-self: flex-start; }
.row { align-items: flex-start; }
.dur { margin-left: auto; flex: none; font-variant-numeric: tabular-nums; }
</style>
