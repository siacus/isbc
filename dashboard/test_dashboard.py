"""Run: uv run --project dashboard python -m unittest dashboard.test_dashboard -v"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from dashboard.core import (annotation_candidates, clear_caches, group_profiles, label_distribution,
                            load_dataset, profile_keys, project, select_documents, profile_label_summary,
                            label_count_summary, profile_outcome_summary)
from dashboard.app import build_view, create_app


class ExplorerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "encoded_fixture_best.csv"
        self.labels = self.root / "fixture_labels.csv"
        # IDs intentionally out of order; profiles repeat, but differ outside filter bit 0.
        pd.DataFrame({"id": ["003", "001", "004", "002", "005"],
                      "text": ["three", "one", "four", "two", "five"],
                      "bit_00": [1, 1, 0, 1, 1], "bit_01": [0, 1, 1, 0, 0],
                      "bit_02": [1, 0, 0, 1, 0], "prob_00": [.9]*5}).to_csv(self.path, index=False)
        pd.DataFrame({"id": ["004", "002", "001", "003"], "category": ["B", "A", "B", "unknown"],
                      "topics": ['["a", "b"]', '["a", "a"]', '["b"]', "unknown"]}).to_csv(self.labels, index=False)
        self.data = load_dataset(str(self.path), str(self.labels))

    def tearDown(self):
        clear_caches()
        self.tmp.cleanup()

    def test_theme_and_viewport_preserve_data(self):
        from dashboard.appearance import apply_theme, remember_plot_state, apply_viewport
        fig, view, _ = build_view(self.data, [], 100, 42, "pca", "hamming", "local", 3, 15, .1)
        before = fig.to_dict()["data"]
        revision = fig.layout.uirevision
        apply_theme(fig, "light")
        self.assertEqual(before, fig.to_dict()["data"])
        self.assertEqual(fig.layout.uirevision, revision)
        self.assertEqual(fig.layout.paper_bgcolor, "#ffffff")
        state = remember_plot_state(fig.to_dict(), {}, {})
        camera = {"eye": {"x": 2, "y": 1, "z": .5}}
        state = remember_plot_state(fig.to_dict(), {"scene.camera": camera}, state)
        state = remember_plot_state(fig.to_dict(), {"autosize": True}, state)
        apply_viewport(fig, state)
        self.assertEqual(fig.layout.scene.camera.eye.x, 2)
        fig.update_layout(uirevision="other")
        self.assertEqual(remember_plot_state(fig.to_dict(), {}, state)["layout"], {})

    def test_pdf_snapshot_retains_current_content_and_blocks_executable_markup(self):
        from dashboard.report import snapshot_document, build_dashboard_pdf
        snapshot = {"width": 1280, "css": "body{background:white}", "markup":
                    '<div><details open><summary>Source</summary>run.csv</details>'
                    '<section>Label composition: 147 Business</section>'
                    '<pre>Selected document: café &amp; climate</pre>'
                    '<img src="data:image/png;base64,AAAA" onerror="bad()">'
                    '<script>bad()</script><iframe src="https://invalid.test"></iframe>'
                    '<img src="https://invalid.test/pixel"><input value="1234" checked></div>'}
        document, width = snapshot_document(snapshot)
        self.assertEqual(width, 1280)
        self.assertIn("Selected document: café &amp; climate", document)
        self.assertIn("147 Business", document)
        self.assertIn("<details open>", document)
        self.assertIn('value="1234" checked', document)
        self.assertNotIn("bad()", document)
        self.assertNotIn("invalid.test", document)
        self.assertIn("default-src 'none'", document)
        self.assertIn('src="data:image/png;base64,AAAA"', document)
        async def render(markup, viewport_width):
            self.assertIn("Selected document", markup)
            self.assertEqual(viewport_width, 1280)
            return b"%PDF-fixture"
        with patch("dashboard.report._render_snapshot", side_effect=render), \
             patch("dashboard.core.project", side_effect=AssertionError("No projection refit")):
            self.assertEqual(build_dashboard_pdf(snapshot), b"%PDF-fixture")

    def test_join_by_id_and_preserve_leading_zeros(self):
        self.assertEqual(self.data.frame.id.tolist(), ["001", "002", "003", "004", "005"])
        self.assertEqual(self.data.frame.iloc[0]["category"], "B")
        self.assertEqual(self.data.frame.iloc[1]["text"], "two")
        self.assertNotIn("prob_00", self.data.frame)

    def test_conjunction_and_seed(self):
        eligible, sample = select_documents(self.data, ["bit_00"], 2, 42)
        self.assertEqual(eligible.tolist(), [0, 1, 2, 4])
        self.assertEqual(sample.tolist(), select_documents(self.data, ["bit_00"], 2, 42)[1].tolist())
        a = select_documents(self.data, ["bit_00", "bit_02"], 100, 7)[1]
        b = select_documents(self.data, ["bit_02", "bit_00"], 100, 7)[1]
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.tolist(), [1, 2])
        self.assertEqual(len(select_documents(self.data, [], 100, 0)[0]), 5)

    def test_filter_does_not_remove_other_coordinates(self):
        _, sample = select_documents(self.data, ["bit_00"], 100, 0)
        profiles, _, counts = group_profiles(self.data.bits[sample])
        self.assertEqual(set(profile_keys(profiles)), {"100", "101", "110"})
        self.assertEqual(sorted(counts.tolist()), [1, 1, 2])
        _, view, _ = build_view(self.data, ["bit_00"], 100, 0, "pca", "hamming", "local", 3, 15, .1)
        self.assertEqual(len(view["keys"]), 3)
        self.assertEqual(view["members"][view["keys"].index("101")], [1, 2])
        self.assertGreater(np.linalg.norm(np.array(view["coordinates"])[0] - np.array(view["coordinates"])[1]), 0)

    def test_exact_profile_requires_every_unselected_bit_off(self):
        eligible, sample = select_documents(self.data, ["bit_00"], 100, 42, "exact")
        self.assertEqual(eligible.tolist(), [4])
        self.assertEqual(sample.tolist(), [4])
        a = select_documents(self.data, ["bit_00", "bit_02"], 1, 42, "exact")[1]
        b = select_documents(self.data, ["bit_02", "bit_00"], 1, 42, "exact")[1]
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(select_documents(self.data, [], 100, 42, "exact")[0]), 0)
        self.assertEqual(len(select_documents(self.data, self.data.bit_columns, 100, 42, "exact")[0]), 0)
        _, view, _ = build_view(self.data, ["bit_00"], 100, 42, "pca", "hamming", "local", 3, 15, .1,
                                match_mode="exact")
        self.assertEqual(view["keys"], ["100"])
        self.assertEqual(view["settings"]["match_mode"], "exact")
        self.assertEqual(view["eligible_count"], 1)
        # Empty exact selection must include actual all-zero codes when present.
        self.data.bits[0, :] = 0
        self.assertEqual(select_documents(self.data, [], 100, 42, "exact")[0].tolist(), [0])

    def test_label_denominators_not_plot_sample(self):
        eligible, _ = select_documents(self.data, ["bit_00"], 1, 0)
        stats = label_distribution(self.data, eligible, "category")
        self.assertEqual(stats["valid"], 2)
        self.assertEqual(stats["missing"], 2)
        self.assertEqual({r["category"]: r["percent"] for r in stats["categories"]}, {"A": 50, "B": 50})
        self.assertEqual(stats["corpus_valid"], 3)

    def test_document_count_table_distinguishes_populations(self):
        eligible, sample = select_documents(self.data, ["bit_00"], 2, 42)
        stats = label_count_summary(self.data, eligible, sample, "category")
        by_label = {row["category"]: row for row in stats["categories"]}
        self.assertEqual(stats["totals"], {"corpus": 5, "matching": 4, "sample": 2})
        self.assertEqual(by_label["B"]["corpus"], 2)
        self.assertEqual(by_label["B"]["matching"], 1)
        self.assertEqual(by_label["B"]["matching_percent"], 50)
        self.assertEqual(stats["missing"]["matching"], 2)
        self.assertEqual(sum(row["sample"] for row in stats["categories"]) + stats["missing"]["sample"], 2)
        exact, all_exact = select_documents(self.data, ["bit_00"], 100, 42, "exact")
        stats = label_count_summary(self.data, exact, all_exact, "category")
        self.assertEqual(stats["totals"]["matching"], 1)
        self.assertTrue(all(row["matching"] == 0 for row in stats["categories"]))
        self.assertTrue(all(row["matching_percent"] is None for row in stats["categories"]))
        # Counts remain document counts even when two labels share one complete profile.
        self.data.frame.loc[2, "category"] = "B"
        same_profile, sample = select_documents(self.data, ["bit_00", "bit_02"], 100, 42, "exact")
        stats = label_count_summary(self.data, same_profile, sample, "category")
        self.assertEqual({r["category"]: r["matching"] for r in stats["categories"]}, {"A": 1, "B": 1})

    def test_multilabel_document_count_table(self):
        stats = label_count_summary(self.data, np.arange(5), np.array([0, 1]), "topics", True)
        self.assertEqual(stats["observed"]["matching"], 3)
        self.assertEqual({r["category"]: r["matching"] for r in stats["categories"]}, {"a": 2, "b": 2})
        self.assertEqual(sum(r["sample"] for r in stats["categories"]), 2)

    def test_multilabel_counts_documents_once(self):
        stats = label_distribution(self.data, np.arange(5), "topics", True)
        self.assertEqual(stats["valid"], 3)
        self.assertEqual({r["category"]: r["count"] for r in stats["categories"]}, {"a": 2, "b": 2})

    def test_label_colouring_ties_missing_and_multilabel(self):
        summaries, cats = profile_label_summary(self.data, [[0, 1], [2, 4], [1, 2]], "category")
        self.assertEqual(cats, ["A", "B"])
        self.assertEqual(summaries[0]["kind"], "tie")
        self.assertEqual(summaries[1]["kind"], "missing")
        self.assertEqual(summaries[2]["label"], "A")
        self.assertEqual(summaries[2]["missing"], 1)
        multi, _ = profile_label_summary(self.data, [[0, 1, 3]], "topics", True)
        self.assertEqual(multi[0]["kind"], "tie")
        self.assertEqual(multi[0]["counts"], {"a": 2, "b": 2})

    def test_colour_changes_neither_sample_nor_geometry(self):
        for dim in [2, 3]:
            a, va, _ = build_view(self.data, [], 100, 42, "pca", "hamming", "global", dim, 15, .1)
            b, vb, _ = build_view(self.data, [], 100, 42, "pca", "hamming", "global", dim, 15, .1,
                                   color_field="category")
            self.assertEqual(va["sample_indices"], vb["sample_indices"])
            self.assertEqual(va["coordinates"], vb["coordinates"])
            self.assertEqual(a.layout.uirevision, b.layout.uirevision)
            self.assertEqual(sum(len(trace.x) for trace in b.data), len(vb["keys"]))
            # customdata[0] must remain the complete profile for click-to-read callbacks.
            self.assertEqual({row[0] for trace in b.data for row in trace.customdata}, set(vb["keys"]))
            self.assertTrue(b.layout.showlegend)

    def test_continuous_outcome_medians_missing_and_fixed_scale(self):
        self.data.frame['engagement'] = ['-10', '2', '6', 'invalid', 'inf']
        self.data.labels.append('engagement')
        stats = profile_outcome_summary(self.data, [[1, 2], [3, 4]], 'engagement')
        self.assertEqual(stats['profiles'][0], dict(valid=2, missing=0, median=4., mean=4.))
        self.assertEqual(stats['profiles'][1]['median'], None)
        self.assertEqual(stats['corpus_missing'], 2)
        self.assertEqual((stats['scale']['cmin'], stats['scale']['cmax']), (-10, 10))
        self.assertEqual(stats['scale']['type'], 'diverging')
        for dim in [2, 3]:
            plain, va, _ = build_view(self.data, [], 100, 42, 'pca', 'hamming', 'global', dim, 15, .1)
            fig, vb, _ = build_view(self.data, [], 100, 42, 'pca', 'hamming', 'global', dim, 15, .1,
                                    outcome_field='engagement')
            self.assertEqual(va['sample_indices'], vb['sample_indices'])
            self.assertEqual(va['coordinates'], vb['coordinates'])
            self.assertEqual(plain.layout.uirevision, fig.layout.uirevision)
            self.assertEqual({r[0] for trace in fig.data for r in trace.customdata}, set(vb['keys']))
            self.assertEqual(len(fig.data), 2)  # finite and explicitly missing
            self.assertEqual(fig.data[0].marker.cmin, -10)
            self.assertEqual(fig.data[0].marker.cmax, 10)
            self.assertEqual(fig.data[1].marker.color, '#738295')
            filtered, vc, _ = build_view(self.data, ['bit_02'], 100, 42, 'pca', 'hamming', 'global', dim, 15, .1,
                                         outcome_field='engagement')
            self.assertEqual(filtered.data[0].marker.cmin, -10)
            self.assertEqual(filtered.data[0].marker.cmax, 10)
            self.assertEqual(list(filtered.data[0].marker.color), [4])
            self.assertEqual(vc['colouring']['scale'], vb['colouring']['scale'])
            json.dumps(vb, allow_nan=False)

    def test_outcome_signs_constants_and_discrete_override(self):
        self.data.labels.append('outcome')
        for values in ([0, 1, 4, 8, 10], [-10, -8, -4, -1, 0], [0]*5, [-2]*5, [2]*5):
            self.data.frame['outcome'] = values
            info = profile_outcome_summary(self.data, [[0, 1, 2, 3, 4]], 'outcome')
            self.assertEqual(info['scale']['type'], 'sequential')
            self.assertLess(info['scale']['cmin'], info['scale']['cmax'])
            self.assertLessEqual(info['scale']['cmin'], min(values))
            self.assertGreaterEqual(info['scale']['cmax'], max(values))
        self.data.frame['outcome'] = ['bad', '', 'inf', '-inf', 'unknown']
        fig, view, note = build_view(self.data, [], 100, 42, 'pca', 'hamming', 'local', 2, 15, .1,
                                     outcome_field='outcome')
        self.assertEqual(view['colouring']['corpus_valid'], 0)
        self.assertIn('no finite numeric values', note)
        self.assertEqual(fig.data[0].marker.color, '#738295')
        json.dumps(view, allow_nan=False)
        self.data.frame['outcome'] = ['0', '0', '1', '', '1']
        fig, view, _ = build_view(self.data, [], 100, 42, 'pca', 'hamming', 'local', 2, 15, .1,
                                  outcome_field='outcome', outcome_type='discrete')
        self.assertEqual(view['colouring']['categories'], ['0', '1'])
        self.assertIn('tie', [s['kind'] for s in view['colouring']['profiles']])
        self.assertTrue(fig.layout.showlegend)

    def test_outcome_callback_export_and_annotation_startup(self):
        f = pd.read_csv(self.labels, dtype=str, keep_default_na=False)
        f['engagement'] = ['0', '2', '10', '6']
        f.to_csv(self.labels, index=False)
        data = load_dataset(str(self.path), str(self.labels))
        app = create_app(str(self.root), str(self.labels))
        client = app.server.test_client()
        layout = client.get('/_dash-layout').get_data(as_text=True)
        self.assertIn('outcome-field', layout)
        self.assertIn(str(self.labels), json.dumps(json.loads(layout)))
        ds = dict(run=str(self.path), annotations=str(self.labels), signature=data.signature)
        redraw_key = next(k for k in app.callback_map if 'map.figure' in k)
        values = [ds, [], 100, 42, 'pca', 'hamming', 'global', 2, 15, .1, 'category', [],
                  'unrestricted', 'dark', 'engagement', 'continuous', 'log']
        cb = app.callback_map[redraw_key]
        response = client.post('/_dash-update-component', json=dict(
            output=redraw_key, outputs=[dict(id=x.component_id, property=x.component_property) for x in cb['output']],
            changedPropIds=['outcome-field.value'], inputs=[dict(**x, value=v) for x, v in zip(cb['inputs'], values)], state=[]))
        self.assertEqual(response.status_code, 200, response.data)
        view = response.json['response']['view']['data']
        self.assertEqual(view['colouring']['field'], 'engagement')
        self.assertEqual(view['colouring']['profiles'][view['keys'].index('101')]['median'], 4)
        response = client.post('/_dash-update-component', json={
            'output': 'download.data', 'outputs': {'id': 'download', 'property': 'data'},
            'changedPropIds': ['export.n_clicks'],
            'inputs': [{'id': 'export', 'property': 'n_clicks', 'value': 1}],
            'state': [{'id': 'dataset', 'property': 'data', 'value': ds},
                      {'id': 'view', 'property': 'data', 'value': view},
                      {'id': 'label-fields', 'property': 'value', 'value': ['category']},
                      {'id': 'multi-fields', 'property': 'value', 'value': []}]})
        self.assertEqual(response.status_code, 200, response.data)
        payload = json.loads(response.json['response']['download']['data']['content'])
        self.assertEqual(payload['settings']['outcome_field'], 'engagement')
        self.assertAlmostEqual(payload['colouring']['scale']['cmax'], np.log1p(10))
        self.assertEqual(payload['settings']['outcome_scale'], 'log')
        self.assertEqual(payload['colouring']['profiles'][view['keys'].index('101')]['median'], 4)
        self.assertEqual([s['field'] for s in payload['label_document_counts']], ['category'])

    def test_log_colour_preserves_original_summaries_and_geometry(self):
        self.data.frame['engagement'] = ['-10', '0', '100', 'invalid', '1000']
        self.data.labels.append('engagement')
        for dim in [2, 3]:
            linear, a, _ = build_view(self.data, [], 100, 42, 'pca', 'hamming', 'global', dim, 15, .1,
                                      outcome_field='engagement')
            log, b, _ = build_view(self.data, [], 100, 42, 'pca', 'hamming', 'global', dim, 15, .1,
                                   outcome_field='engagement', outcome_scale='log')
            self.assertEqual(a['colouring']['profiles'], b['colouring']['profiles'])
            self.assertEqual(a['sample_indices'], b['sample_indices'])
            self.assertEqual(a['coordinates'], b['coordinates'])
            self.assertEqual(linear.layout.uirevision, log.layout.uirevision)
            index = b['keys'].index('101')
            self.assertEqual(b['colouring']['profiles'][index]['median'], 50)
            self.assertEqual(b['colouring']['profiles'][index]['mean'], 50)
            self.assertAlmostEqual(b['colouring']['color_values'][index], np.log1p(50))
            self.assertAlmostEqual(log.data[0].marker.cmin, -np.log1p(1000))
            self.assertAlmostEqual(log.data[0].marker.cmax, np.log1p(1000))
            self.assertEqual(log.data[0].marker.colorbar.ticktext[2], '0')
            self.assertEqual(log.data[0].marker.colorbar.ticktext[-1], '1,000')
            for trace in log.data:
                self.assertNotIn('Profile %{customdata[0]}', trace.hovertemplate)
                for row in trace.customdata:
                    expected = ', '.join(f'{i:02d}: 100%' for i, v in enumerate(row[0]) if v == '1') or 'None'
                    self.assertEqual(row[4], expected)
            json.dumps(b, allow_nan=False)
        for values in ([0, 1, 4, 8, 10], [-10, -8, -4, -1, 0], [0]*5, [-2]*5, [2]*5):
            self.data.frame['engagement'] = values
            stats = profile_outcome_summary(self.data, [[0, 1, 2, 3, 4]], 'engagement', 'log')
            self.assertEqual(stats['profiles'][0]['median'], np.median(values))
            self.assertEqual(stats['scale']['type'], 'sequential')
            self.assertLess(stats['scale']['cmin'], stats['scale']['cmax'])

    def test_global_positions_stable_for_overlapping_profiles(self):
        fig_a, a, _ = build_view(self.data, [], 100, 4, "pca", "hamming", "global", 3, 15, .1)
        fig_b, b, _ = build_view(self.data, ["bit_00"], 100, 4, "pca", "hamming", "global", 3, 15, .1)
        lookup = dict(zip(a["keys"], a["coordinates"]))
        for key, coord in zip(b["keys"], b["coordinates"]):
            np.testing.assert_allclose(lookup[key], coord)
        self.assertEqual(fig_a.layout.scene.xaxis.range, fig_b.layout.scene.xaxis.range)

    def test_weighted_pca_matches_document_pca(self):
        profiles, coordinates, _ = project(self.data.bits, "pca", "hamming", 3, 0)
        expected = PCA(n_components=3).fit(self.data.bits).transform(profiles)
        # Eigenvector signs are arbitrary; pairwise distances are invariant.
        from scipy.spatial.distance import pdist
        np.testing.assert_allclose(pdist(coordinates), pdist(expected), atol=1e-7)

    def test_mds_and_tiny_samples(self):
        for metric in ["hamming", "jaccard"]:
            _, coords, info = project(self.data.bits, "mds", metric, 3, 0)
            self.assertTrue(np.isfinite(coords).all())
            self.assertIn("negative_eigenvalue_mass_fraction", info)
            _, coords, _ = project(np.zeros((5, 3), dtype=np.uint8), "mds", metric, 3, 0)
            np.testing.assert_array_equal(coords, np.zeros((1, 3)))
        _, view, _ = build_view(self.data, ["bit_00", "bit_01", "bit_02"], 100, 0,
                                "pca", "hamming", "local", 3, 15, .1)
        self.assertEqual(view["sample_indices"], [])

    def test_no_embedding_or_checkpoint_access(self):
        metadata = {"input": str(self.labels), "raw_embedding_cache": {"matrix": "/DO_NOT_OPEN.npy"},
                    "checkpoint": "/DO_NOT_OPEN.pt"}
        (self.root / "metadata.json").write_text(json.dumps(metadata))
        clear_caches()
        real_open = open
        def guarded(file, *args, **kwargs):
            self.assertNotIn(str(file).split(".")[-1], {"npy", "npz", "pt"})
            return real_open(file, *args, **kwargs)
        with patch("builtins.open", side_effect=guarded), patch("numpy.load", side_effect=AssertionError("No embedding reads")):
            data = load_dataset(str(self.path), str(self.labels))
            build_view(data, [], 5, 0, "pca", "hamming", "global", 3, 15, .1)
            self.assertEqual(annotation_candidates(str(self.path)), [str(self.labels.resolve())])

    def test_reject_invalid_bits_and_duplicate_ids(self):
        f = pd.read_csv(self.path, dtype=str)
        f.loc[0, "bit_01"] = "0.5"
        f.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "exactly 0 or 1"):
            load_dataset(str(self.path))
        f.loc[0, "bit_01"] = "1"
        f.loc[0, "id"] = f.loc[1, "id"]
        f.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "unique"):
            load_dataset(str(self.path))

    def test_app_layout_and_http(self):
        app = create_app(str(self.root))
        client = app.server.test_client()
        self.assertEqual(client.get("/").status_code, 200)
        layout = client.get("/_dash-layout")
        self.assertEqual(layout.status_code, 200)
        self.assertIn(b"Label composition", layout.data)
        self.assertEqual(client.get("/_dash-dependencies").status_code, 200)
        self.assertEqual(client.get("/DESIGN.md").status_code, 200)
        with client.get("/assets/style.css") as response:
            self.assertEqual(response.status_code, 200)

    def test_export_contains_full_codes_and_all_label_counts(self):
        app = create_app(str(self.root))
        client = app.server.test_client()
        _, view, _ = build_view(self.data, ["bit_00"], 2, 42, "pca", "hamming", "local", 3, 15, .1)
        ds = {"run": str(self.path), "annotations": str(self.labels), "signature": self.data.signature}
        response = client.post("/_dash-update-component", json={
            "output": "download.data", "outputs": {"id": "download", "property": "data"},
            "changedPropIds": ["export.n_clicks"],
            "inputs": [{"id": "export", "property": "n_clicks", "value": 1}],
            "state": [{"id": "dataset", "property": "data", "value": ds},
                      {"id": "view", "property": "data", "value": view},
                      {"id": "label-fields", "property": "value", "value": ["category"]},
                      {"id": "multi-fields", "property": "value", "value": []}]
        })
        self.assertEqual(response.status_code, 200, response.data)
        result = json.loads(response.json["response"]["download"]["data"]["content"])
        self.assertEqual(result["bit_columns"], ["bit_00", "bit_01", "bit_02"])
        self.assertEqual(len(result["sample_document_ids"]), 2)
        self.assertEqual(result["label_statistics"][0]["valid"], 2)
        self.assertEqual(result["label_statistics"][0]["missing"], 2)
        self.assertEqual(result["label_document_counts"][0]["totals"]["matching"], 4)
        self.assertEqual(result["label_document_counts"][0]["totals"]["sample"], 2)

    def test_exact_mode_applies_to_label_panels_and_export(self):
        app = create_app(str(self.root))
        client = app.server.test_client()
        ds = {"run": str(self.path), "annotations": str(self.labels), "signature": self.data.signature}
        response = client.post("/_dash-update-component", json={
            "output": "label-panels.children", "outputs": {"id": "label-panels", "property": "children"},
            "changedPropIds": ["match-mode.value"],
            "inputs": [{"id": "dataset", "property": "data", "value": ds},
                       {"id": "bits", "property": "value", "value": ["bit_00"]},
                       {"id": "label-fields", "property": "value", "value": ["category"]},
                       {"id": "multi-fields", "property": "value", "value": []},
                       {"id": "match-mode", "property": "value", "value": "exact"}], "state": []})
        self.assertEqual(response.status_code, 200, response.data)
        self.assertIn("0 observed", response.get_data(as_text=True))
        self.assertIn("1 missing", response.get_data(as_text=True))
        _, view, _ = build_view(self.data, ["bit_00"], 100, 42, "pca", "hamming", "local", 3, 15, .1,
                                match_mode="exact")
        response = client.post("/_dash-update-component", json={
            "output": "download.data", "outputs": {"id": "download", "property": "data"},
            "changedPropIds": ["export.n_clicks"],
            "inputs": [{"id": "export", "property": "n_clicks", "value": 1}],
            "state": [{"id": "dataset", "property": "data", "value": ds},
                      {"id": "view", "property": "data", "value": view},
                      {"id": "label-fields", "property": "value", "value": ["category"]},
                      {"id": "multi-fields", "property": "value", "value": []}]})
        self.assertEqual(response.status_code, 200, response.data)
        exported = json.loads(response.json["response"]["download"]["data"]["content"])
        self.assertEqual(exported["sample_document_ids"], ["005"])
        self.assertEqual(exported["settings"]["match_mode"], "exact")
        self.assertEqual(exported["label_statistics"][0]["missing"], 1)


if __name__ == "__main__":
    unittest.main()
