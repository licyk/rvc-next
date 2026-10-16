import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

/**
 * Rows that pair a name with a control must let the name shrink.
 *
 * An audio file or voice name is often one long unbreakable word (a take number, an experiment
 * name with its epoch), so the row's min-content width is the whole name. A flex item keeps `min-width: auto` by
 * default and refuses to go below that, which pushes the row's buttons out of view. `min-width: 0`
 * on the item that holds the text is what makes the ellipsis work and keeps the control in place.
 */
const SRC = join(process.cwd(), 'src');
const ROWS: [file: string, selector: string][] = [
  ['ui/Snackbar.vue', '.text'],
  ['ui/Chip.vue', '.label'],
  ['ui/ContextMenu.vue', '.label'],
  ['ui/PickerMenu.vue', '.text'],
  ['components/AudioSourceInput.vue', '.name'],
  ['components/ResultsList.vue', '.name'],
  ['components/JobCard.vue', '.titles'],
  ['components/ServerPathDialog.vue', '.name'],
  ['components/DatasetReport.vue', '.path'],
  ['views/ModelsView.vue', '.name'],
  ['views/ModelsView.vue', '.row-text'],
  ['components/ResourceRow.vue', '.text'],
  ['components/ResourceRow.vue', '.name'],
  ['views/TrainView.vue', '.name'],
  ['views/ExperimentView.vue', '.titles'],
];

function rule(file: string, selector: string): string {
  const css = readFileSync(join(SRC, file), 'utf8');
  const match = new RegExp(String.raw`(?:^|\n)\s*${selector.replace('.', '\\.')}\s*\{([^}]*)\}`).exec(css);
  expect(match, `${file} has no rule for ${selector}`).not.toBeNull();
  return match![1];
}

describe('rows that hold a file name', () => {
  it.each(ROWS)('%s %s can shrink below the name', (file, selector) => {
    expect(rule(file, selector)).toContain('min-width: 0');
  });
});

/**
 * Convert, Separate and Live are one column of sections, as Hanaikada's Settings: a wider page
 * widens each section, whose controls wrap in grids, and never sets the sections side by side.
 * Their run buttons sit in the sticky PageFooter, so they stay in view however long the column.
 */
const PAGES = ['views/ConvertView.vue', 'views/SeparateView.vue', 'views/LiveView.vue'];

describe('one-column pages', () => {
  it.each(PAGES)('%s stacks its sections and keeps its actions in the footer', (file) => {
    expect(rule(file, '.sections')).toContain('flex-direction: column');
    expect(readFileSync(join(SRC, file), 'utf8')).toContain('<PageFooter>');
  });
});

/** A field with its buttons beside it (Add, Browse, Build): the buttons centre on the field's box. */
const FIELD_ROWS = ['views/ModelDetailView.vue', 'views/ModelsView.vue', 'views/ExperimentView.vue', 'views/TrainView.vue', 'components/LinkImport.vue', 'ui/TokenField.vue'];

describe('field rows', () => {
  // A field with supporting text: its buttons centre on the field's box, not on the box and the text under it.
  it('FieldRow centres its end on the field box alone', () => {
    expect(rule('ui/FieldRow.vue', '.field-row')).toContain('align-items: flex-start');
    const end = rule('ui/FieldRow.vue', '.end');
    expect(end).toContain('align-items: center');
    expect(end).toContain('min-height: var(--app-field-height)');
  });
  it.each(FIELD_ROWS)('%s puts its fields and their buttons in a FieldRow', (file) => {
    expect(readFileSync(join(SRC, file), 'utf8')).toContain('<FieldRow');
  });
  // Fields side by side in a stretching grid keep their own boxes (Train's settings, Models › Tools).
  it.each([['ui/TextField.vue', '.text-field'], ['ui/SelectField.vue', '.select']] as const)('%s keeps its own height', (file, selector) => {
    expect(rule(file, selector)).toContain('height: fit-content');
  });
});

/** Pages of sections never set them side by side: the model detail page and Models › Tools too. */
describe('one column of sections', () => {
  it('the model detail page stacks its sections', () => {
    expect(rule('views/ModelDetailView.vue', '.sections')).toContain('flex-direction: column');
  });
  it('no page grid fits sections side by side', () => {
    for (const file of ['views/ModelDetailView.vue', 'views/ModelsView.vue']) expect(readFileSync(join(SRC, file), 'utf8')).not.toMatch(/auto-fit, minmax\((320|340)px/);
  });
});
