import { useQueryClient } from '@tanstack/vue-query';
import { ref, watch } from 'vue';
import { ApiError } from '@/api/client';
import { type ImportDecisions, type ImportPlan, type ImportResult, importApi } from '@/api/queries/imports';
import { keys } from '@/api/queries/keys';
import type { Job } from '@/api/types';
import { useJobsStore } from '@/stores/jobs';

/**
 * Import files of any type through an import session: upload, read the plan the server proposes,
 * and commit it at once when nothing needs a decision. Otherwise ``review`` holds the plan for the
 * review dialog, which commits or discards it.
 */
export function useModelImport(onDone: (result: ImportResult) => void, onError: (e: unknown) => void) {
  const qc = useQueryClient();
  const uploading = ref<{ name: string; progress: number } | null>(null);
  const review = ref<ImportPlan | null>(null);
  const committing = ref(false);
  /** The download job of links being added, while it runs. */
  const fetching = ref<string | null>(null);
  const jobs = useJobsStore();

  async function propose(sessionId: string) {
    const plan = await importApi.plan(sessionId);
    if (plan.confident) await commit(plan, {});
    else review.value = plan;
  }

  /** Wait for a job through the jobs store (the socket keeps it current). */
  function finished(job: Job): Promise<Job> {
    jobs.upsert(job);
    return new Promise((resolve) => {
      const stop = watch(
        () => jobs.byId[job.id],
        (j) => {
          if (j && ['completed', 'failed', 'cancelled', 'interrupted'].includes(j.state)) {
            stop();
            resolve(j);
          }
        },
        { immediate: true },
      );
    });
  }

  /** Links (Hugging Face, Google Drive, any http(s) file): the server downloads them into a session. */
  async function importLinks(text: string) {
    const urls = text.split(/\s+/).map((u) => u.trim()).filter(Boolean);
    if (!urls.length) return;
    let sessionId: string | null = null;
    try {
      sessionId = (await importApi.create()).session_id;
      const job = await importApi.addUrls(sessionId, urls);
      fetching.value = job.id;
      const done = await finished(job);
      fetching.value = null;
      if (done.state !== 'completed') {
        if (done.error) throw new ApiError(0, done.error.code, done.error.message, done.error.detail ?? {});
        return;
      }
      await propose(sessionId);
    } catch (e) {
      fetching.value = null;
      if (sessionId) importApi.discard(sessionId).catch(() => undefined);
      onError(e);
    }
  }

  async function importFiles(files: File[]) {
    if (!files.length) return;
    let sessionId: string | null = null;
    try {
      sessionId = (await importApi.create()).session_id;
      for (const file of files) {
        uploading.value = { name: file.name, progress: 0 };
        await importApi.addFile(sessionId, file, (loaded, total) => (uploading.value = { name: file.name, progress: total ? loaded / total : 0 }));
      }
      uploading.value = null;
      await propose(sessionId);
    } catch (e) {
      uploading.value = null;
      if (sessionId) importApi.discard(sessionId).catch(() => undefined);
      onError(e);
    }
  }

  async function commit(plan: ImportPlan, decisions: ImportDecisions) {
    committing.value = true;
    try {
      const result = await importApi.commit(plan.session_id, decisions);
      review.value = null;
      qc.invalidateQueries({ queryKey: keys.models });
      qc.invalidateQueries({ queryKey: keys.separationPresets });
      onDone(result);
    } catch (e) {
      onError(e);
    } finally {
      committing.value = false;
    }
  }

  function discard() {
    if (review.value) importApi.discard(review.value.session_id).catch(() => undefined);
    review.value = null;
  }

  return { uploading, fetching, review, committing, importFiles, importLinks, commit, discard };
}
