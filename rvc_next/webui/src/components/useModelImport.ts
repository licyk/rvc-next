import { useQueryClient } from '@tanstack/vue-query';
import { ref } from 'vue';
import { type ImportDecisions, type ImportPlan, type ImportResult, importApi } from '@/api/queries/imports';
import { keys } from '@/api/queries/keys';

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
      const plan = await importApi.plan(sessionId);
      if (plan.confident) await commit(plan, {});
      else review.value = plan;
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

  return { uploading, review, committing, importFiles, commit, discard };
}
