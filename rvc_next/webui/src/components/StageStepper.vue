<script setup lang="ts">
import { computed } from 'vue';
import type { Experiment } from '@/api/types';
import { useI18n } from '@/i18n';
import { useJobsStore } from '@/stores/jobs';
import { ProgressBar, Stepper, type StepItem } from '@/ui';

export type ExperimentStep = 'dataset' | 'settings' | 'run' | 'results';

/** The experiment's four steps, with the run's live progress in the header. */
const props = defineProps<{ experiment: Experiment }>();
const model = defineModel<ExperimentStep>({ required: true });
const { t, tOr } = useI18n();
const jobs = useJobsStore();
const job = computed(() => (props.experiment.running_job ? (jobs.byId[props.experiment.running_job] ?? null) : null));
const stages = computed(() => Object.values(props.experiment.stages));
const runState = computed<StepItem['state']>(() => {
  if (props.experiment.running_job) return 'running';
  if (stages.value.some((s) => s.status === 'failed')) return 'failed';
  if (stages.value.some((s) => s.status === 'stale')) return 'stale';
  const fit = props.experiment.stages.fit;
  return fit?.status === 'done' ? 'done' : 'pending';
});
const steps = computed<StepItem<ExperimentStep>[]>(() => [
  { id: 'dataset', label: t('train.steps.dataset'), state: props.experiment.dataset.folder || props.experiment.dataset.speakers.length ? 'done' : 'pending' },
  { id: 'settings', label: t('train.steps.settings'), state: 'done' },
  { id: 'run', label: t('train.steps.run'), state: runState.value, note: job.value?.step ? tOr(`train.stages.${job.value.step}`, job.value.step) : undefined },
  { id: 'results', label: t('train.steps.results'), state: props.experiment.stages.fit?.status === 'done' ? 'done' : 'pending' },
]);
</script>

<template>
  <div class="stage-stepper">
    <Stepper v-model="model" :steps="steps" />
    <ProgressBar v-if="job" :value="job.progress" :label="job.title" />
  </div>
</template>

<style scoped>
.stage-stepper { display: flex; flex-direction: column; gap: var(--app-space-2); min-width: 0; }
</style>
