"""Presentation only: themes never alter binary geometry or sampling."""
from copy import deepcopy


def apply_theme(figure, theme="dark"):
    light = theme == "light"
    bg, ink, grid = ("#ffffff", "#172b3a", "#d9e1e7") if light else ("#101c2a", "#e7eef6", "#354b60")
    figure.update_layout(template="plotly_white" if light else "plotly_dark",
                         paper_bgcolor=bg, plot_bgcolor=bg, font_color=ink,
                         scene_bgcolor=bg)
    axis = dict(backgroundcolor=bg, gridcolor=grid, zerolinecolor=grid, color=ink)
    figure.update_layout(scene={f"{a}axis": axis for a in "xyz"})
    figure.update_xaxes(gridcolor=grid, zerolinecolor=grid, color=ink)
    figure.update_yaxes(gridcolor=grid, zerolinecolor=grid, color=ink)
    return figure


def remember_plot_state(figure, relayout, previous):
    """Keep camera and 2D zoom across unrelated Plotly layout events."""
    revision = (figure or {}).get("layout", {}).get("uirevision")
    state = deepcopy(previous or {})
    if state.get("revision") != revision:
        return {"revision": revision, "layout": {}}
    for key, value in (relayout or {}).items():
        if key == "scene.camera" or key in {"xaxis.range", "yaxis.range", "xaxis.autorange", "yaxis.autorange"} or key in {
            "xaxis.range[0]", "xaxis.range[1]", "yaxis.range[0]", "yaxis.range[1]"
        }:
            state.setdefault("layout", {})[key] = value
    return state


def apply_viewport(figure, state):
    if not state or state.get("revision") != figure.layout.uirevision:
        return figure
    layout = state.get("layout", {})
    if "scene.camera" in layout:
        figure.update_layout(scene_camera=layout["scene.camera"])
    for name in ("xaxis", "yaxis"):
        axis = {}
        if name + ".range" in layout:
            axis["range"] = layout[name + ".range"]
        elif all(name + f".range[{i}]" in layout for i in (0, 1)):
            axis["range"] = [layout[name + f".range[{i}]"] for i in (0, 1)]
        if "range" in axis:
            axis["autorange"] = False
        if name + ".autorange" in layout:
            axis["autorange"] = layout[name + ".autorange"]
        figure.update_layout(**{name: axis})
    return figure
