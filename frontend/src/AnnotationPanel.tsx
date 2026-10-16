import { useEffect, useRef, useState } from "react";
import { groupAnnotations, groupKey, normalize } from "./annotationGroups";
import type { CSSProperties, FormEvent } from "react";
import type { Annotation, AnnotationInput, Position, RegionData } from "./types";

interface Props {
  records: Annotation[];
  selectedGroup: string | null;
  onSelectGroup: (key: string) => void;
  position: Position | null;
  region: RegionData | null;
  selected: Annotation | null;
  picking: boolean;
  /** Reason annotation is unavailable for the active representation, or null. */
  unsupported: string | null;
  enabled: boolean;
  busy: boolean;
  canPick: boolean;
  error: string;
  notice: string;
  canRecover: boolean;
  onRecover: () => Promise<void>;
  onNew: () => void;
  onCancel: () => void;
  onSelect: (annotation: Annotation) => void;
  onSave: (input: AnnotationInput) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}

function groupCount(members: { record: Annotation }[]) {
  const regions = members.filter(({ record }) => record.kind === "region").length;
  const points = members.length - regions;
  const pointText = `${points} ${points === 1 ? "point" : "points"}`;
  const regionText = `${regions} ${regions === 1 ? "region" : "regions"}`;
  return regions ? (points ? `${pointText} · ${regionText}` : regionText) : pointText;
}

export function AnnotationPanel(props: Props) {
  const { selected, position } = props;
  const [category, setCategory] = useState("Landmark");
  const [label, setLabel] = useState("");
  const [note, setNote] = useState("");
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [confirmRecovery, setConfirmRecovery] = useState(false);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const groups = groupAnnotations(props.records);
  const suggestions = [...new Map(props.records
    .filter((record) => normalize(record.category) === normalize(category))
    .map((record) => [normalize(record.label), record.label.trim()])).values()];
  const labelInput = useRef<HTMLInputElement>(null);
  useEffect(() => { if (!props.canRecover) setConfirmRecovery(false); }, [props.canRecover]);
  useEffect(() => {
    setCategory(selected?.category ?? "Landmark");
    setLabel(selected?.label ?? "");
    setNote(selected?.note ?? "");
    setConfirmDelete(null);
    if (selected) setCollapsed((keys) => {
      const next = new Set(keys); next.delete(groupKey(selected.category, selected.label)); return next;
    });
    if (position) {
      labelInput.current?.focus({ preventScroll: true });
      if (window.innerWidth <= 800)
        labelInput.current?.scrollIntoView({ block: "center" });
    }
  }, [selected, position]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!position || !category.trim() || !label.trim()) return;
    await props.onSave({
      kind: props.region ? "region" : "point",
      region: props.region,
      x: position.x,
      y: position.y,
      z: position.z,
      category: category.trim(),
      label: label.trim(),
      note: note.trim(),
    });
  }

  return (
    <aside
      className="annotation-panel panel"
      aria-labelledby="annotation-heading"
    >
      <div className="panel-heading">
        <div>
          <p className="eyebrow">SPATIAL NOTES</p>
          <h2 id="annotation-heading">
            Annotations <span className="count">{props.records.length}</span>
          </h2>
        </div>
        <span className="panel-icon">⌖</span>
      </div>
      <p className="subtle">Mark a detail. Give it meaning.</p>
      {props.unsupported && (
        <p className="alert unsupported-notice" role="status" data-testid="annotation-unsupported">
          {props.unsupported}
        </p>
      )}
      <button
        className={`primary full-width ${props.picking ? "is-picking" : ""}`}
        onClick={props.onNew}
        disabled={!props.enabled || props.busy || !props.canPick}
        aria-pressed={props.picking}
      >
        {props.picking ? "◎ Picking… click a point" : "+ Pick point"}
      </button>
      {!props.unsupported && (
        <ol className="workflow">
          <li>Pick a point, or use Rectangle / Lasso to select a part.</li>
          <li>Add a category and label, then save.</li>
          <li>Select a group to highlight it; select a member to focus or edit.</li>
        </ol>
      )}

      {props.error && (
        <p className="alert error" role="alert">
          {props.error}
        </p>
      )}
      {props.canRecover && (
        <div className="storage-recovery">
          {confirmRecovery ? <>
            <p>Discard the damaged local notes for this scene? This cannot be undone.</p>
            <button className="danger" disabled={props.busy} onClick={() => { void props.onRecover(); setConfirmRecovery(false); }}>Confirm reset local notes</button>
            <button disabled={props.busy} onClick={() => setConfirmRecovery(false)}>Keep data</button>
          </> : <button disabled={props.busy} onClick={() => setConfirmRecovery(true)}>Reset this scene's local notes</button>}
        </div>
      )}
      {props.notice && (
        <p className="alert success" role="status">
          {props.notice}
        </p>
      )}
      {position ? (
        <form
          onSubmit={(event) => void submit(event)}
          className="annotation-form"
        >
          <div className="form-title">
            <h3>{selected ? "Edit annotation" : "New annotation"}</h3>
            <span className="marker-key">
              {props.region ? `Region · ${props.region.selected_count.toLocaleString()} points` : selected ? "● Selected" : "● New point"}
            </span>
          </div>
          <div className="coordinates" aria-label="Selected coordinates">
            {(["x", "y", "z"] as const).map((axis) => (
              <span key={axis}>
                {axis.toUpperCase()} <b>{position[axis].toFixed(3)}</b>
              </span>
            ))}
          </div>
          <label htmlFor="annotation-category">Category</label>
          <input
            id="annotation-category"
            value={category}
            onChange={(event) => setCategory(event.target.value)}
            list="category-options"
            required
            maxLength={40}
            disabled={props.busy}
          />
          <datalist id="category-options">
            <option value="Landmark" />
            <option value="Structure" />
            <option value="Surface" />
            <option value="Observation" />
          </datalist>
          <label htmlFor="annotation-label">
            Label <span>required</span>
          </label>
          <input
            ref={labelInput}
            id="annotation-label"
            list="label-options"
            placeholder="e.g. Upper edge of the torus"
            value={label}
            onChange={(event) => setLabel(event.target.value)}
            required
            maxLength={80}
            disabled={props.busy}
          />
          <datalist id="label-options">{suggestions.map((value) => <option key={normalize(value)} value={value} />)}</datalist>
          <label htmlFor="annotation-note">
            Note <span>optional</span>
          </label>
          <textarea
            id="annotation-note"
            placeholder={props.region ? "What makes this region interesting?" : "What makes this point interesting?"}
            value={note}
            onChange={(event) => setNote(event.target.value)}
            maxLength={1000}
            rows={3}
            disabled={props.busy}
          />
          <div className="form-actions">
            <button
              className="primary"
              type="submit"
              disabled={
                props.busy ||
                !props.enabled ||
                !label.trim() ||
                !category.trim()
              }
            >
              {props.busy
                ? "Working…"
                : selected
                  ? "Save changes"
                  : "Save annotation"}
            </button>
            <button
              type="button"
              onClick={props.onCancel}
              disabled={props.busy}
            >
              Cancel
            </button>
          </div>
        </form>
      ) : (
        props.picking && (
          <p className="selection-tip" role="status">
            The next click near a point places a visible marker. Empty space
            will not create an annotation.
          </p>
        )
      )}

      <div className="saved-heading">
        <h3>Saved annotations</h3>
        <span>{props.records.length.toString().padStart(2, "0")}</span>
      </div>
      {props.records.length === 0 ? (
        <div className="empty-notes">
          <span>⌖</span>
          <h3>Your observations start here</h3>
          <p>Pick a point to leave your first spatial note.</p>
        </div>
      ) : (
        <div className="annotation-groups">
          {groups.map((group, groupIndex) => (
            <section key={group.key} className={`annotation-group ${props.selectedGroup === group.key ? "group-selected" : ""}`}
              style={{ "--group-color": group.color } as CSSProperties} data-testid="annotation-group">
              <div className="group-header">
                <button className="group-toggle" aria-label={`${collapsed.has(group.key) ? "Expand" : "Collapse"} ${group.category}: ${group.label}`}
                  aria-expanded={!collapsed.has(group.key)} aria-controls={`group-members-${groupIndex}`}
                  onClick={() => setCollapsed((keys) => {
                    const next = new Set(keys);
                    if (next.has(group.key)) next.delete(group.key); else next.add(group.key);
                    return next;
                  })}>{collapsed.has(group.key) ? "▸" : "▾"}</button>
                <button className="group-select" onClick={() => props.onSelectGroup(group.key)} disabled={props.busy}
                  aria-label={`Highlight group ${group.category}: ${group.label}`} aria-pressed={props.selectedGroup === group.key}>
                  <span className="category-tag">{group.category}</span>
                  <strong>{group.label}</strong><span className="group-count">{groupCount(group.members)}</span>
                </button>
              </div>
              <ul id={`group-members-${groupIndex}`} className="annotation-list" hidden={collapsed.has(group.key)}>
          {group.members.map(({ record, number }) => (
            <li
              key={record.id}
              className={selected?.id === record.id ? "selected" : ""}
            >
              <button
                className="annotation-select"
                aria-label={`Focus ${record.kind === "region" ? "region" : "point"} ${number}: ${record.label}`}
                onClick={() => props.onSelect(record)}
                disabled={props.busy}
                aria-pressed={selected?.id === record.id}
              >
                <span className="annotation-number">{number}</span>
                <span>
                  <strong>{record.kind === "region" ? "Region" : "Point"} {number}{record.region ? ` · ${record.region.selected_count.toLocaleString()} points` : ""}</strong>
                  {record.note && (
                    <span className="annotation-note">{record.note}</span>
                  )}
                  <span className="small-coordinates">
                    {record.x.toFixed(2)}, {record.y.toFixed(2)},{" "}
                    {record.z.toFixed(2)}
                  </span>
                </span>
              </button>
              <div className="delete-row">
                {confirmDelete === record.id ? (
                  <>
                    <span>Delete this note?</span>
                    <button
                      className="danger"
                      disabled={props.busy}
                      onClick={() => {
                        void props.onDelete(record.id);
                        setConfirmDelete(null);
                      }}
                    >
                      Confirm delete
                    </button>
                    <button
                      disabled={props.busy}
                      onClick={() => setConfirmDelete(null)}
                    >
                      Keep
                    </button>
                  </>
                ) : (
                  <button
                    className="text-button"
                    disabled={props.busy}
                    onClick={() => setConfirmDelete(record.id)}
                    aria-label={`Delete ${record.label}`}
                  >
                    Delete
                  </button>
                )}
              </div>
            </li>
          ))}
              </ul>
            </section>
          ))}
        </div>
      )}
      <p className="annotation-footer">
        Markers stay attached to scene coordinates as you navigate.
      </p>
    </aside>
  );
}
