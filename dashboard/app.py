"""Run with: uv run --project dashboard python dashboard/app.py [--runs PATH]."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
from dash import Dash, Input, Output, State, ClientsideFunction, ctx, dcc, html, no_update
from flask import Response

try:
    from .appearance import apply_theme, remember_plot_state
    from .report import build_dashboard_pdf
    from .core import (annotation_candidates, clear_caches, discover_runs, group_profiles,
                       label_distribution, load_dataset, profile_keys, project, select_documents, profile_label_summary,
                       label_count_summary, profile_outcome_summary)
except ImportError:
    from appearance import apply_theme, remember_plot_state
    from report import build_dashboard_pdf
    from core import (annotation_candidates, clear_caches, discover_runs, group_profiles,
                      label_distribution, load_dataset, profile_keys, project, select_documents, profile_label_summary,
                      label_count_summary, profile_outcome_summary)

HERE = Path(__file__).resolve().parent


def empty_figure(message="Choose an analysis to explore its binary codes."):
    fig = go.Figure()
    fig.add_annotation(text=message, x=.5, y=.5, xref="paper", yref="paper", showarrow=False,
                       font={"color": "#98a8b9", "size": 15})
    fig.update_layout(template="plotly_dark", paper_bgcolor="#101c2a", plot_bgcolor="#101c2a",
                      margin=dict(l=20, r=20, t=20, b=20), xaxis_visible=False, yaxis_visible=False)
    return fig


def field(label, control, hint=None):
    return html.Div([html.Label(label), control, html.Small(hint) if hint else None], className="field")


def metric_card(value, label):
    return html.Div([html.Strong(value), html.Span(label)], className="metric-card")


def make_figure(coordinates, keys, counts, dimensions, revision, colouring=None):
    # Marker area encodes sampled document count. Minimum diameter preserves click targets.
    sizes = np.maximum(6, 32 * np.sqrt(counts / counts.max()))
    activity = np.array([key.count("1") for key in keys])
    custom = [[key, int(count), int(active), "",
               ", ".join(f"{i:02d}: 100%" for i, value in enumerate(key) if value == "1") or "None"]
              for key, count, active in zip(keys, counts, activity)]
    groups = [(None, np.arange(len(keys)), None)]
    continuous = bool(colouring and colouring.get("kind") == "continuous")
    if continuous:
        summaries = colouring["profiles"]
        valid = np.array([s["median"] is not None for s in summaries])
        groups = []
        if valid.any():
            groups.append(("Profile median", np.flatnonzero(valid), None))
        if (~valid).any():
            groups.append(("Missing / nonnumeric", np.flatnonzero(~valid), "#738295"))
        for i, summary in enumerate(summaries):
            values = (f"<br>Median: {summary['median']:,.6g}<br>Mean: {summary['mean']:,.6g}"
                      if summary["median"] is not None else "<br>No finite numeric values")
            custom[i][3] = (f"{escape(colouring['field'])} · sampled profile members" + values
                           + f"<br>{summary['valid']} observed, {summary['missing']} missing / nonnumeric")
    elif colouring:
        summaries = colouring["profiles"]
        palette = {label: qualitative.Dark24[i] if len(colouring["categories"]) <= 24
                   else f"hsl({(i * 137.508) % 360:.2f},65%,65%)"
                   for i, label in enumerate(colouring["categories"])}
        category_keys = [(s["kind"], s["label"]) for s in summaries]
        groups = []
        for key in sorted(set(category_keys), key=lambda k: (k[0] != "label", k[0], k[1] or "")):
            kind, label = key
            name = label if kind == "label" else "Tied most frequent labels" if kind == "tie" else "Missing / inapplicable"
            color = palette[label] if kind == "label" else "#d4c2df" if kind == "tie" else "#738295"
            groups.append((escape(name), np.array([i for i, k in enumerate(category_keys) if k == key]), color))
        for i, summary in enumerate(summaries):
            mix = "<br>".join(f"{escape(label)}: {count} ({100*count/summary['valid']:.1f}% of observed)"
                              for label, count in list(summary["counts"].items())[:8])
            if len(summary["counts"]) > 8:
                mix += f"<br>… and {len(summary['counts']) - 8} more labels"
            custom[i][3] = (f"{escape(colouring['field'])} · sampled profile members"
                           f"<br>{summary['valid']} observed, {summary['missing']} missing"
                           + ("<br>" + mix if mix else ""))
    fig = go.Figure()
    for name, indices, color in groups:
        marker = dict(size=sizes[indices].tolist(), opacity=.85, line=dict(width=.5, color="#b2e4df"))
        if continuous and color is None:
            scale = colouring["scale"]
            colorbar = dict(title=dict(text=escape(colouring["field"]) + "<br>Profile median"
                            + (" (log colours)" if scale.get("mode") == "log" else "")), thickness=12)
            if scale.get("tickvals") is not None:
                colorbar.update(tickmode="array", tickvals=scale["tickvals"], ticktext=scale["ticktext"])
            marker.update(color=[colouring["color_values"][i] for i in indices],
                          colorscale=scale["colorscale"], cmin=scale["cmin"], cmax=scale["cmax"],
                          showscale=True, colorbar=colorbar)
        elif colouring:
            marker["color"] = color
        else:
            marker.update(color=activity.tolist(), colorscale="Tealgrn",
                          colorbar=dict(title="Active bits", thickness=12))
        common = dict(mode="markers", marker=marker, customdata=[custom[i] for i in indices], name=name,
                      showlegend=bool(colouring) and (not continuous or color is not None),
                      hovertemplate="%{customdata[1]} sampled documents"
                                    "<br>Active bits in this profile: %{customdata[4]}"
                                    + ("<br>%{customdata[3]}" if colouring else "") + "<extra></extra>")
        if dimensions == 3:
            trace = go.Scatter3d(x=coordinates[indices, 0].tolist(), y=coordinates[indices, 1].tolist(),
                                z=coordinates[indices, 2].tolist(), **common)
        else:
            trace = go.Scattergl(x=coordinates[indices, 0].tolist(), y=coordinates[indices, 1].tolist(), **common)
        fig.add_trace(trace)
    fig.update_layout(template="plotly_dark", paper_bgcolor="#101c2a", plot_bgcolor="#101c2a",
                      margin=dict(l=12, r=12, t=12, b=12), uirevision=revision, showlegend=bool(colouring),
                      legend=dict(orientation="h", y=-.03, x=0, title_text=escape(colouring["field"]) if colouring else ""),
                      scene=dict(xaxis_title="Axis 1", yaxis_title="Axis 2", zaxis_title="Axis 3",
                                 bgcolor="#101c2a", aspectmode="data", dragmode="orbit"),
                      xaxis_title="Axis 1", yaxis_title="Axis 2")
    if dimensions == 2:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def build_view(data, selected, sample_size, seed, method, metric, scope, dimensions, neighbors, min_dist,
               color_field="", color_multilabel=False, match_mode="unrestricted",
               outcome_field="", outcome_type="continuous", outcome_scale="linear"):
    start = time.perf_counter()
    eligible, sample = select_documents(data, selected, int(sample_size), int(seed), match_mode)
    settings = dict(bits=sorted(selected), match_mode=match_mode, sample_size=int(sample_size), seed=int(seed), method=method,
                    metric=metric if method != "pca" else "euclidean", scope=scope,
                    dimensions=int(dimensions), neighbors=int(neighbors), min_dist=float(min_dist))
    view = {"dataset": data.signature, "sample_indices": sample.tolist(), "settings": settings,
            "eligible_count": len(eligible), "keys": [], "members": [], "counts": [], "projection": {}}
    if not len(sample):
        return empty_figure("No documents match this exact profile." if match_mode == "exact" else
                            "No documents have all the selected bits active."), view, "No matching documents."
    unique, inverse, counts = group_profiles(data.bits[sample])
    keys = profile_keys(unique)
    view.update(keys=keys, counts=counts.tolist(),
                members=[sample[np.flatnonzero(inverse == i)].tolist() for i in range(len(unique))])
    fit_codes = data.bits if scope == "global" else data.bits[sample]
    fitted, positions, info = project(fit_codes, method, metric, int(dimensions), int(seed),
                                     int(neighbors), float(min_dist))
    lookup = {key: i for i, key in enumerate(profile_keys(fitted))}
    coordinates = positions[[lookup[key] for key in keys]]
    view["projection"] = info
    view["coordinates"] = coordinates.tolist()
    revision = hashlib.sha256(json.dumps([data.signature, settings,
                             sample.tolist() if scope == "local" else None], sort_keys=True).encode()).hexdigest()[:12]
    # Global camera stays fixed when only the filter/sample changes.
    if scope == "global":
        camera_settings = {k: v for k, v in settings.items() if k not in {"bits", "sample_size", "match_mode"}}
        revision = hashlib.sha256(json.dumps([data.signature, camera_settings], sort_keys=True).encode()).hexdigest()[:12]
    colouring = None
    if outcome_field:
        if outcome_field not in data.labels:
            raise ValueError("Choose an outcome available in this dataset.")
        if outcome_type == "continuous":
            colouring = profile_outcome_summary(data, view["members"], outcome_field, outcome_scale)
        elif outcome_type == "discrete":
            summaries, categories = profile_label_summary(data, view["members"], outcome_field)
            colouring = dict(field=outcome_field, kind="discrete", multilabel=False,
                             profiles=summaries, categories=categories)
        else:
            raise ValueError("Choose Discrete or Continuous for the outcome.")
    elif color_field in data.labels:
        summaries, categories = profile_label_summary(data, view["members"], color_field, color_multilabel)
        colouring = dict(field=color_field, kind="discrete", multilabel=color_multilabel, profiles=summaries, categories=categories)
    # Colour is an annotation overlay: keep the projection, sample and camera revision unchanged.
    view["colouring"] = colouring
    view["settings"].update(color_field=color_field if color_field in data.labels else "",
                            color_multilabel=bool(color_multilabel), outcome_field=outcome_field,
                            outcome_type=outcome_type if outcome_field else None,
                            outcome_scale=outcome_scale if outcome_field and outcome_type == "continuous" else None)
    fig = make_figure(coordinates, keys, counts, int(dimensions), revision, colouring)
    if scope == "global":
        ranges = []
        for axis in range(int(dimensions)):
            low, high = float(positions[:, axis].min()), float(positions[:, axis].max())
            pad = max((high - low) * .06, .05)
            ranges.append([low - pad, high + pad])
        if int(dimensions) == 3:
            fig.update_layout(scene={f"{axis}axis": {"range": limits, "autorange": False}
                                     for axis, limits in zip("xyz", ranges)})
        else:
            fig.update_xaxes(range=ranges[0], autorange=False)
            fig.update_yaxes(range=ranges[1], autorange=False)
    caption = f"All {len(data.bit_columns)} binary coordinates · {info['weighting']} · {time.perf_counter()-start:.2f} s"
    if "variance_explained" in info:
        caption += f" · displayed PCA variance: {sum(info['variance_explained']):.1%}"
    if "negative_eigenvalue_mass_fraction" in info:
        caption += f" · negative eigenvalue mass: {info['negative_eigenvalue_mass_fraction']:.1%}"
    if info.get("note"):
        caption += " · " + info["note"]
    if colouring and colouring.get("kind") == "continuous":
        caption += (f" · colour: median {outcome_field} among sampled profile members"
                    " · fixed full-corpus scale · hover for mean and observed count")
        if outcome_scale == "log":
            caption += " · log colours: sign(x) × log(1 + |x|); summaries and colour-bar labels in original units"
        if not colouring["corpus_valid"]:
            caption += " · no finite numeric values; choose Discrete for categorical outcomes"
        elif colouring["corpus_min"] == colouring["corpus_max"]:
            caption += f" · constant corpus outcome: {colouring['corpus_min']:g} (display range padded)"
    elif colouring:
        caption += " · colour: most frequent observed label among sampled profile members; hover for composition"
    return fig, view, caption


def create_app(root: str | None = None, annotations: str | None = None):
    app = Dash(__name__, assets_folder=str(HERE / "assets"), title="ISBC Explorer", update_title="Computing…")

    @app.server.get("/DESIGN.md")
    def design_document():
        return Response((HERE / "DESIGN.md").read_text(), mimetype="text/plain")

    root = str(Path(root or HERE.parent / "runs").expanduser().resolve())
    app.layout = html.Div([
        dcc.Store(id="dataset"), dcc.Store(id="view"), dcc.Download(id="download"),
        dcc.Store(id="plot-state"), dcc.Store(id="pdf-snapshot"), dcc.Download(id="download-pdf"),
        html.Header([
            html.Div([html.Div("ISBC / EXPLORER", className="eyebrow"),
                      html.H1("Read between the bits."),
                      html.P("Explore binary profiles. Follow their documents.")]),
            html.Div([html.Span("● Local session", className="local-badge"),
                      field("Appearance", dcc.RadioItems(id="theme", options=[{"label": "Dark", "value": "dark"},
                          {"label": "White", "value": "light"}], value="dark", inline=True, persistence=True,
                          persistence_type="local", className="theme-switch")),
                      html.Div([html.Button("Export view", id="export", n_clicks=0, className="quiet"),
                                html.Button("Export PDF", id="export-pdf", n_clicks=0, className="quiet")], className="export-actions"),
                      dcc.Loading(html.Div(id="pdf-status", className="note"))], className="header-actions")
        ]),
        html.Details([html.Summary("Open an analysis"), html.Div([
            field("Runs directory or encoded CSV", dcc.Input(id="root", value=root, type="text", debounce=True)),
            html.Div([html.Button("↑ Parent", id="up", n_clicks=0),
                      dcc.Dropdown(id="folders", placeholder="Browse subfolders…"),
                      html.Button("Open folder", id="open-folder", n_clicks=0),
                      html.Button("Refresh", id="scan", n_clicks=0)], className="folder-row"),
            html.Div(id="scan-status", className="note"),
            field("Encoded documents", dcc.Dropdown(id="run", placeholder="Select a run…")),
            html.Div([
                field("Suggested annotation file", dcc.Dropdown(id="annotations", clearable=True,
                                                                   placeholder="Optional labels")),
                field("Annotation or text source override", dcc.Input(id="annotation-override", type="text",
                       placeholder="Optional CSV / Parquet path", value=annotations or "", debounce=True))], className="two-cols"),
            html.Button("Load analysis", id="load", n_clicks=0, className="primary"),
            dcc.Loading(html.Div(id="load-status", className="note")),
        ], className="source-content")], className="source-panel", open=True),
        html.Main([
            html.Aside([
                html.Div("01 / SELECT DOCUMENTS", className="eyebrow"), html.H2("Active bits"),
                html.P("Require every selected bit to be active.", className="muted"),
                field("Unselected bits", dcc.RadioItems(id="match-mode", value="unrestricted", options=[
                    {"label": " Leave unrestricted", "value": "unrestricted"},
                    {"label": " Require all to be off", "value": "exact"}], className="match-mode"),
                    "All others off matches one complete profile. With no selected bits, it matches the all-zero profile."),
                dcc.Checklist(id="bits", options=[], value=[], className="bit-grid",
                              inputClassName="bit-input", labelClassName="bit-label"),
                html.Button("Clear selection", id="clear-bits", n_clicks=0, className="quiet small"),
                html.Div(id="selection-rule", className="selection-rule"),
                html.Div([
                    field("Sample size", dcc.Input(id="size", type="number", value=1000, min=1, max=10000, step=1, debounce=True)),
                    field("Random seed", dcc.Input(id="seed", type="number", value=1234, min=0, max=4294967295, step=1, debounce=True))
                ], className="two-cols"),
                html.Hr(), html.Div("02 / DESCRIBE THE SELECTION", className="eyebrow"),
                html.H2("Label composition"),
                html.P("All matching documents, not just plotted points.", className="muted"),
                field("Annotation fields", dcc.Dropdown(id="label-fields", options=[], value=[], multi=True)),
                html.Details([html.Summary("Multi-label settings"),
                    field("Fields containing multiple labels", dcc.Dropdown(id="multi-fields", options=[], value=[], multi=True)),
                    html.Small("JSON lists or labels separated by | or ;. Each document counts once per label.")]),
                dcc.Loading(html.Div(id="label-panels", className="label-scroll")),
            ], className="sidebar"),
            html.Section([
                html.Div(id="summary", className="metrics"),
                html.Div([
                    field("Projection", dcc.Dropdown(id="method", options=[{"label": "PCA", "value": "pca"},
                          {"label": "Classical MDS", "value": "mds"}, {"label": "UMAP", "value": "umap"}], value="pca", clearable=False)),
                    field("Scope", dcc.Dropdown(id="scope", options=[{"label": "Selected sample", "value": "local"},
                          {"label": "Global context", "value": "global"}], value="local", clearable=False)),
                    field("Distance", dcc.Dropdown(id="metric", options=[{"label": "Hamming", "value": "hamming"},
                          {"label": "Jaccard", "value": "jaccard"}], value="hamming", clearable=False)),
                    field("View", dcc.RadioItems(id="dimensions", options=[2, 3], value=3, inline=True)),
                ], className="projection-controls"),
                field("Colour points by", dcc.Dropdown(id="color-field", options=[{"label": "Active-bit count", "value": ""}],
                                                     value="", clearable=False),
                      "Label colour summarises sampled documents sharing the complete profile. Ties and missing labels have separate colours."),
                html.Div([
                    field("Colour by outcome", dcc.Dropdown(id="outcome-field", options=[], value=None,
                          placeholder="None: use label / active-bit colour", clearable=True),
                          "One outcome at a time. Clear this selection to restore label colour."),
                    field("Outcome type", dcc.RadioItems(id="outcome-type", options=[
                          {"label": " Discrete", "value": "discrete"},
                          {"label": " Continuous", "value": "continuous"}], value="continuous", inline=True),
                          "Choose explicitly: counts can use a continuous scale. Profile colour uses the median; hover shows mean and valid count.")
                ], className="two-cols"),
                field("Outcome colour scale", dcc.Dropdown(id="outcome-scale", options=[
                      {"label": "Linear", "value": "linear"},
                      {"label": "Logarithmic (zero and negative values supported)", "value": "log"}],
                      value="linear", clearable=False, disabled=True),
                      "Colour only: sign(x) × log(1 + |x|). Mean, median and colour-bar labels remain in original units."),
                html.Div([
                    field("UMAP neighbours", dcc.Input(id="neighbors", type="number", value=15, min=2, max=200, step=1, debounce=True)),
                    field("UMAP min. distance", dcc.Input(id="min-dist", type="number", value=.1, min=0, max=1, step=.05, debounce=True)),
                ], id="umap-controls", className="two-cols"),
                dcc.Loading(dcc.Graph(id="map", figure=empty_figure(), className="map",
                            config={"scrollZoom": True, "displaylogo": False,
                                    "toImageButtonOptions": {"filename": "isbc-binary-profiles", "format": "png"}})),
                html.Div(id="projection-note", className="note"),
                html.P("Drag to rotate · scroll to zoom · use the plot toolbar to pan or reset. "
                       "Point size shows sampled document count; colour by active-bit count, label or outcome.", className="note"),
                html.Section([
                    html.H2("Document counts by label"),
                    html.P("Counts of documents, not profile points. Each document keeps its own label, "
                           "even when its profile is coloured by another category.", className="muted"),
                    dcc.Loading(html.Div(id="label-count-tables")),
                ], className="reader"),
                html.Section([
                    html.Div("03 / READ THE DOCUMENTS", className="eyebrow"),
                    html.H2("Behind the profile"),
                    html.P("Click a point or choose a profile. Only identical complete codes share a point.", className="muted"),
                    field("Complete profile", dcc.Dropdown(id="profile", options=[], placeholder="Choose a point…")),
                    field("Document in this sampled profile", dcc.Dropdown(id="document", options=[], placeholder="Select a document…")),
                    html.Div(id="document-body"),
                ], className="reader")
            ], className="workspace")
        ]),
        html.Details([html.Summary("Help · user guide"),
                      dcc.Markdown((HERE / "README.md").read_text(), link_target="_blank")], className="help"),
        html.Footer("ISBC Explorer · Binary codes only · All processing stays on this computer")
    ], className="app-shell", id="app-shell")

    @app.callback(Output("app-shell", "className"), Input("theme", "value"))
    def theme_class(theme):
        return "app-shell theme-light" if theme == "light" else "app-shell"

    @app.callback(Output("plot-state", "data"), Input("map", "figure"), Input("map", "relayoutData"),
                  State("plot-state", "data"))
    def plot_state(figure, relayout, previous):
        return remember_plot_state(figure, relayout if ctx.triggered_prop_ids.get("map.relayoutData") else {}, previous)

    @app.callback(Output("root", "value"), Input("up", "n_clicks"), Input("open-folder", "n_clicks"),
                  State("root", "value"), State("folders", "value"), prevent_initial_call=True)
    def navigate(_up, _open, path, folder):
        return str(Path(path).expanduser().parent) if ctx.triggered_id == "up" else (folder or no_update)

    @app.callback(Output("run", "options"), Output("run", "value"), Output("folders", "options"),
                  Output("folders", "value"), Output("scan-status", "children"),
                  Input("root", "value"), Input("scan", "n_clicks"))
    def scan(path, _click):
        try:
            p = Path(path).expanduser().resolve()
            base = p if p.is_dir() else p.parent
            folders = [{"label": f.name, "value": str(f)} for f in sorted(base.iterdir())
                       if f.is_dir() and not f.name.startswith(".")]
            runs = discover_runs(str(p))
            options = [{"label": str(f.relative_to(base)), "value": str(f)} for f in runs]
            default = next((str(f) for f in runs if f.parent.name == "agnews_qwen_all_fixed_balance"),
                           str(runs[0]) if runs else None)
            return options, default, folders, None, f"{len(runs)} encoded tables found."
        except (OSError, ValueError, TypeError) as exc:
            return [], None, [], None, str(exc)

    @app.callback(Output("annotations", "options"), Output("annotations", "value"), Input("run", "value"))
    def suggest(run):
        if not run:
            return [], None
        try:
            candidates = annotation_candidates(run)
            return [{"label": p, "value": p} for p in candidates], candidates[0] if candidates else None
        except (OSError, ValueError):
            return [], None

    @app.callback(Output("dataset", "data"), Output("bits", "options"), Output("bits", "value"),
                  Output("label-fields", "options"), Output("label-fields", "value"),
                  Output("multi-fields", "options"), Output("multi-fields", "value"), Output("load-status", "children"),
                  Input("load", "n_clicks"), Input("clear-bits", "n_clicks"),
                  State("run", "value"), State("annotations", "value"), State("annotation-override", "value"),
                  prevent_initial_call=True)
    def load(_click, _clear, run, annotations, override):
        if ctx.triggered_id == "clear-bits":
            return no_update, no_update, [], no_update, no_update, no_update, no_update, no_update
        if not run:
            return None, [], [], [], [], [], [], "Choose an encoded document table."
        try:
            annotation = (override or "").strip() or annotations or ""
            data = load_dataset(run, annotation)
            options = [{"label": f"{int(c.split('_')[1]):02d}", "value": c} for c in data.bit_columns]
            default_labels = [c for c in ["label_name", "source", "type", "label_primary", "sentiment"] if c in data.labels]
            if not default_labels:
                default_labels = data.labels[:2]
            status = f"Loaded {len(data.frame):,} documents · {len(data.bit_columns)} bits. " + " ".join(data.notes)
            return {"run": run, "annotations": annotation, "signature": data.signature}, options, [], \
                data.labels, default_labels, data.labels, [], status
        except (OSError, ValueError, KeyError) as exc:
            return None, [], [], [], [], [], [], f"Could not load analysis: {exc}"

    @app.callback(Output("metric", "disabled"), Output("umap-controls", "style"), Input("method", "value"))
    def controls(method):
        return method == "pca", {"display": "grid" if method == "umap" else "none"}

    @app.callback(Output("color-field", "options"), Output("color-field", "value"), Input("dataset", "data"))
    def color_options(ds):
        options = [{"label": "Active-bit count", "value": ""}]
        if ds:
            data = load_dataset(ds["run"], ds["annotations"])
            options += [{"label": name, "value": name} for name in data.labels]
        return options, ""

    @app.callback(Output("outcome-field", "options"), Output("outcome-field", "value"),
                  Input("dataset", "data"))
    def outcome_options(ds):
        if not ds:
            return [], None
        data = load_dataset(ds["run"], ds["annotations"])
        return [{"label": name, "value": name} for name in data.labels], None

    @app.callback(Output("color-field", "disabled"), Input("outcome-field", "value"))
    def outcome_override(outcome):
        return bool(outcome)

    @app.callback(Output("outcome-scale", "disabled"), Input("outcome-field", "value"),
                  Input("outcome-type", "value"))
    def outcome_scale_enabled(outcome, kind):
        return not outcome or kind != "continuous"

    @app.callback(Output("map", "figure"), Output("view", "data"), Output("projection-note", "children"),
                  Output("summary", "children"), Output("selection-rule", "children"),
                  Input("dataset", "data"), Input("bits", "value"), Input("size", "value"), Input("seed", "value"),
                  Input("method", "value"), Input("metric", "value"), Input("scope", "value"),
                  Input("dimensions", "value"), Input("neighbors", "value"), Input("min-dist", "value"),
                  Input("color-field", "value"), Input("multi-fields", "value"), Input("match-mode", "value"),
                  Input("theme", "value"), Input("outcome-field", "value"), Input("outcome-type", "value"),
                  Input("outcome-scale", "value"))
    def redraw(ds, selected, size, seed, method, metric, scope, dim, neighbors, min_dist, color_field, multi, match_mode, theme,
               outcome_field, outcome_type, outcome_scale):
        if not ds:
            return apply_theme(empty_figure(), theme), None, "", [], "Load an analysis to filter documents."
        try:
            if any(v is None for v in [size, seed, neighbors, min_dist]):
                raise ValueError("Fill in sample size, seed and projection parameters.")
            data = load_dataset(ds["run"], ds["annotations"])
            fig, view, note = build_view(data, selected or [], size, seed, method, metric, scope, dim, neighbors, min_dist,
                                         color_field or "", color_field in (multi or []), match_mode,
                                         outcome_field or "", outcome_type, outcome_scale)
            cards = [metric_card(f"{view['eligible_count']:,}", "matching documents"),
                     metric_card(f"{view['eligible_count']/len(data.frame):.1%}", "of corpus"),
                     metric_card(f"{len(view['sample_indices']):,}", "sampled documents"),
                     metric_card(f"{len(view['keys']):,}", "distinct sampled profiles")]
            rule = " AND ".join(f"{bit} = 1" for bit in sorted(selected or [])) or "All documents"
            if match_mode == "exact":
                rule = ((" AND ".join(f"{bit} = 1" for bit in sorted(selected)) + "; all other bits = 0")
                        if selected else "All bits = 0 (exact profile)")
            return apply_theme(fig, theme), view, note, cards, rule
        except (ValueError, OSError, KeyError, ImportError) as exc:
            return apply_theme(empty_figure("Projection unavailable — see details below."), theme), None, str(exc), [], ""

    @app.callback(Output("label-panels", "children"), Input("dataset", "data"), Input("bits", "value"),
                  Input("label-fields", "value"), Input("multi-fields", "value"), Input("match-mode", "value"))
    def labels(ds, selected, fields, multi, match_mode):
        if not ds:
            return html.P("Load an analysis to see its annotations.", className="muted")
        data = load_dataset(ds["run"], ds["annotations"])
        eligible, _ = select_documents(data, selected or [], 1, 0, match_mode)
        cards = []
        for name in fields or []:
            stats = label_distribution(data, eligible, name, name in (multi or []))
            rows = []
            for row in stats["categories"][:20]:
                rows.append(html.Div([
                    html.Div([html.Span(row["category"]), html.Strong(f"{row['percent']:.1f}%")], className="bar-title"),
                    html.Div(html.Div(style={"width": f"{row['percent']}%"}), className="bar-track"),
                    html.Small(f"n = {row['count']:,} · corpus {row['corpus_percent']:.1f}%")], className="category"))
            if len(stats["categories"]) > 20:
                rows.append(html.Small(f"Showing 20 of {len(stats['categories'])} categories. Export includes all."))
            cards.append(html.Div([html.H3(name),
                html.P(f"{stats['valid']:,} observed · {stats['missing']:,} missing / inapplicable", className="note"),
                *(rows or [html.P("No observed labels in this selection.", className="muted")]),
                html.Small("Multi-label prevalence; percentages may exceed 100% in total.") if stats["multilabel"] else None
            ], className="label-card"))
        return cards or html.P("Choose annotation fields above. Labels are optional.", className="muted")

    @app.callback(Output("label-count-tables", "children"),
                  Input("dataset", "data"), Input("bits", "value"), Input("size", "value"),
                  Input("seed", "value"), Input("match-mode", "value"), Input("label-fields", "value"),
                  Input("multi-fields", "value"), Input("color-field", "value"),
                  Input("outcome-field", "value"), Input("outcome-type", "value"))
    def label_tables(ds, selected, size, seed, match_mode, fields, multi, color_field, outcome_field, outcome_type):
        if not ds:
            return html.P("Load an analysis with annotations.", className="muted")
        if size is None or seed is None:
            return html.P("Enter a sample size and seed.", className="muted")
        data = load_dataset(ds["run"], ds["annotations"])
        try:
            eligible, sample = select_documents(data, selected or [], int(size), int(seed), match_mode)
        except ValueError as exc:
            return html.P(str(exc), className="muted")
        names = list(dict.fromkeys([*(fields or []), *([color_field] if not outcome_field and color_field in data.labels else []),
                                   *([outcome_field] if outcome_field and outcome_type == "discrete" else [])]))
        tables = []
        for name in names:
            if name not in data.labels:
                continue
            stats = label_count_summary(data, eligible, sample, name, name in (multi or []))
            rows = [html.Tr([html.Td(row["category"]), html.Td(f"{row['matching']:,}"),
                            html.Td(f"{row['matching_percent']:.2f}%" if row["matching_percent"] is not None else "—"),
                            html.Td(f"{row['sample']:,}"), html.Td(f"{row['corpus']:,}")])
                    for row in stats["categories"]]
            missing = stats["missing"]
            rows.append(html.Tr([html.Td("Missing / inapplicable"), html.Td(f"{missing['matching']:,}"),
                                 html.Td("—"), html.Td(f"{missing['sample']:,}"), html.Td(f"{missing['corpus']:,}")],
                                className="missing-row"))
            total = stats["totals"]
            footer = html.Tr([html.Th("All documents"), html.Th(f"{total['matching']:,}"), html.Th("—"),
                              html.Th(f"{total['sample']:,}"), html.Th(f"{total['corpus']:,}")])
            tables.append(html.Details([
                html.Summary(name),
                html.P(f"Matching percentages use {stats['observed']['matching']:,} documents with an observed annotation. "
                       + ("Multi-label counts can overlap." if stats["multilabel"] else ""), className="note"),
                html.Div(html.Table([
                    html.Thead(html.Tr([html.Th(c) for c in ["Label", "Matching docs", "Matching %", "In plot sample", "Whole corpus"]])),
                    html.Tbody(rows), html.Tfoot(footer)], className="count-table"), className="table-scroll")
            ], open=True, className="count-details"))
        return tables or html.P("Choose annotation fields in the sidebar or a label in Colour points by.", className="muted")

    @app.callback(Output("profile", "options"), Output("profile", "value"),
                  Input("map", "clickData"), Input("view", "data"), State("profile", "value"))
    def choose_profile(click, view, previous):
        if not view or not view["keys"]:
            return [], None
        options = [{"label": f"{key} · {count:,} sampled documents", "value": key}
                   for key, count in zip(view["keys"], view["counts"])]
        if ctx.triggered_id == "map" and click and click.get("points"):
            candidate = click["points"][0].get("customdata", [None])[0]
            if candidate in view["keys"]:
                return options, candidate
        return options, previous if previous in view["keys"] else None

    @app.callback(Output("document", "options"), Output("document", "value"),
                  Input("profile", "value"), Input("view", "data"), State("dataset", "data"))
    def documents(profile, view, ds):
        if not ds or not view or profile not in view["keys"]:
            return [], None
        data = load_dataset(ds["run"], ds["annotations"])
        members = view["members"][view["keys"].index(profile)]
        options = [{"label": f"{data.frame.iloc[i]['id']} · {data.frame.iloc[i]['text'][:90]}", "value": i} for i in members]
        return options, members[0]

    @app.callback(Output("document-body", "children"), Input("document", "value"), State("dataset", "data"))
    def document(index, ds):
        if index is None or not ds:
            return html.P("Select a profile to read its documents.", className="muted")
        data = load_dataset(ds["run"], ds["annotations"])
        row = data.frame.iloc[int(index)]
        return [html.Div(f"Document {row['id']}", className="eyebrow"),
                html.Div([html.Span(c, className="bit-tag") for c in data.bit_columns if row[c]], className="tags"),
                html.Pre(row["text"] or "Text is unavailable in this table.", className="document-text"),
                html.Div([html.Span(f"{c}: {row[c]}", className="label-tag") for c in data.labels
                          if str(row[c]).strip().lower() not in {"", "unknown", "nan", "none", "null"}], className="tags")]

    @app.callback(Output("download", "data"), Input("export", "n_clicks"), State("dataset", "data"),
                  State("view", "data"), State("label-fields", "value"), State("multi-fields", "value"),
                  prevent_initial_call=True)
    def export(_click, ds, view, fields, multi):
        if not ds or not view:
            return no_update
        data = load_dataset(ds["run"], ds["annotations"])
        eligible, sample = select_documents(data, view["settings"]["bits"], view["settings"]["sample_size"],
                                             view["settings"]["seed"], view["settings"].get("match_mode", "unrestricted"))
        payload = {"schema_version": 1, "data": ds, "settings": view["settings"],
                   "bit_columns": data.bit_columns, "projection": view["projection"], "colouring": view.get("colouring"),
                   "sample_document_ids": data.frame.iloc[sample].id.tolist(),
                   "profiles": [{"code": key, "count": count,
                                 "document_ids": data.frame.iloc[members].id.tolist(), "coordinates": coord}
                                for key, count, members, coord in zip(view["keys"], view["counts"], view["members"], view.get("coordinates", []))],
                   "label_statistics": [label_distribution(data, eligible, name, name in (multi or [])) for name in fields or []],
                   "label_document_counts": [label_count_summary(data, eligible, sample, name, name in (multi or []))
                                             for name in dict.fromkeys([*(fields or []),
                                                 *([view['colouring']['field']] if view.get('colouring') and
                                                   view['colouring'].get('kind') != 'continuous' else [])])
                                             if name in data.labels],
                   "versions": {name: importlib.metadata.version(name) for name in
                                ["dash", "plotly", "numpy", "pandas", "scipy", "umap-learn"]}}
        return dict(content=json.dumps(payload, indent=2, ensure_ascii=False), filename="isbc-exploration.json")

    app.clientside_callback(
        ClientsideFunction(namespace="dashboard", function_name="capture"),
        Output("pdf-snapshot", "data"), Input("export-pdf", "n_clicks"), prevent_initial_call=True)

    @app.callback(Output("download-pdf", "data"), Output("pdf-status", "children"),
                  Input("pdf-snapshot", "data"), prevent_initial_call=True,
                  running=[(Output("export-pdf", "disabled"), True, False)])
    def export_pdf(snapshot):
        if not snapshot:
            return no_update, ""
        if snapshot.get("error"):
            return no_update, "PDF not exported: " + snapshot["error"]
        try:
            result = build_dashboard_pdf(snapshot)
            download = dcc.send_bytes(result, "isbc-dashboard.pdf")
            return download, html.Span(["Complete dashboard ready. ",
                html.A("Download PDF", href="data:application/pdf;base64," + download["content"],
                       download="isbc-dashboard.pdf", style={"color": "var(--accent)"})])
        except Exception as exc:
            return no_update, ("PDF export could not finish. A locally installed Chrome or Chromium browser is required. "
                               f"Details: {str(exc)[:350]}")

    return app


def main():
    parser = argparse.ArgumentParser(description="Explore ISBC binary codes locally; embeddings are never used.")
    parser.add_argument("--runs", default=None, help="Runs root, model directory, or encoded CSV (also selectable in the UI)")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--annotations", default=None, help="Optional CSV/Parquet with labels or outcomes, joined by id")
    args = parser.parse_args()
    app = create_app(args.runs, args.annotations)
    print(f"Open http://127.0.0.1:{args.port} — all processing is local.")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
