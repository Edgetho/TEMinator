import unittest
import csv
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from viewer_edx import SpectrumAnalysisManager
from eds_models import EDSIntegrationSettings


class ModelFitSerializationTests(unittest.TestCase):
    def test_serialize_model_fit_result_with_object_metadata(self):
        manager = SpectrumAnalysisManager.__new__(SpectrumAnalysisManager)
        item = SimpleNamespace(
            metadata=SimpleNamespace(General=SimpleNamespace(title="Fe_Ka")),
            data=np.asarray([1.0, 2.0, 3.0]),
        )
        manager.spectrum_metadata = {
            "SpecA": {
                "last_fit_result": [item],
            }
        }

        rows = SpectrumAnalysisManager._serialize_model_fit_result(manager)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["spectrum"], "SpecA")
        self.assertEqual(rows[0]["line"], "Fe_Ka")
        self.assertEqual(rows[0]["intensity"], 6.0)

    def test_serialize_model_fit_result_with_dict_metadata(self):
        manager = SpectrumAnalysisManager.__new__(SpectrumAnalysisManager)
        item = SimpleNamespace(
            metadata={"General": {"title": "Pt_La"}},
            data=np.asarray([4.0, 5.0]),
        )
        manager.spectrum_metadata = {
            "SpecB": {
                "last_fit_result": item,
            }
        }

        rows = SpectrumAnalysisManager._serialize_model_fit_result(manager)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["spectrum"], "SpecB")
        self.assertEqual(rows[0]["line"], "Pt_La")
        self.assertEqual(rows[0]["intensity"], 9.0)

    def test_serialize_model_fit_result_defaults_line_label(self):
        manager = SpectrumAnalysisManager.__new__(SpectrumAnalysisManager)
        item = SimpleNamespace(
            metadata=SimpleNamespace(),
            data=np.asarray([2.5]),
        )
        manager.spectrum_metadata = {
            "SpecC": {
                "last_fit_result": [item],
            }
        }

        rows = SpectrumAnalysisManager._serialize_model_fit_result(manager)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["line"], "line")
        self.assertEqual(rows[0]["intensity"], 2.5)

    def test_write_model_fit_rows_csv_writes_schema_and_values(self):
        manager = SpectrumAnalysisManager.__new__(SpectrumAnalysisManager)
        manager.logger = logging.getLogger("test-model-fit-csv")
        rows = [
            {"spectrum": "SpecA", "line": "Fe_Ka", "intensity": 6.0},
            {"spectrum": "SpecB", "line": "Pt_La", "intensity": 1.0 / 3.0},
        ]

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_fit.csv"
            ok = SpectrumAnalysisManager._write_model_fit_rows_csv(manager, path, rows)
            self.assertTrue(ok)
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                written_rows = list(reader)

        self.assertEqual(written_rows[0], ["spectrum", "line", "intensity"])
        self.assertEqual(written_rows[1], ["SpecA", "Fe_Ka", "6"])
        self.assertEqual(written_rows[2], ["SpecB", "Pt_La", "0.333333333333"])

    def test_model_fit_summary_groups_by_spectrum_with_timestamp(self):
        manager = SpectrumAnalysisManager.__new__(SpectrumAnalysisManager)
        manager.spectrum_metadata = {
            "SpecA": {
                "last_fit_result": [
                    SimpleNamespace(
                        metadata=SimpleNamespace(General=SimpleNamespace(title="Fe_Ka")),
                        data=np.asarray([1.0, 2.0]),
                    ),
                    SimpleNamespace(
                        metadata=SimpleNamespace(General=SimpleNamespace(title="Fe_Kb")),
                        data=np.asarray([4.0]),
                    ),
                ],
                "last_fit_timestamp": "2026-01-01T00:00:00+00:00",
            },
            "SpecB": {
                "last_fit_result": SimpleNamespace(
                    metadata={"General": {"title": "Pt_La"}},
                    data=np.asarray([5.0]),
                ),
            },
        }

        summary = SpectrumAnalysisManager._model_fit_summary_by_spectrum(manager)
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["spectrum"], "SpecA")
        self.assertEqual(summary[0]["line_count"], 2)
        self.assertEqual(summary[0]["total_intensity"], 7.0)
        self.assertEqual(summary[0]["last_fit_timestamp"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(summary[1]["spectrum"], "SpecB")
        self.assertEqual(summary[1]["line_count"], 1)
        self.assertEqual(summary[1]["total_intensity"], 5.0)
        self.assertNotIn("last_fit_timestamp", summary[1])

    def test_write_region_metadata_json_includes_model_fit_summary(self):
        manager = SpectrumAnalysisManager.__new__(SpectrumAnalysisManager)
        manager.logger = logging.getLogger("test-model-fit-json")
        manager._calibration_source = "metadata"
        manager.beam_energy_ev = 200000.0
        manager.spectrum_dispersion = 10.0
        manager.spectrum_offset = 0.0
        manager.live_time_s = 1.0
        manager.real_time_s = 1.2
        manager.integration_settings = EDSIntegrationSettings(
            integration_windows_ev=[(6400.0, 6500.0)],
            background_mode="none",
            included_lines=["Fe_Ka"],
        )
        manager.model_status_label = SimpleNamespace(text=lambda: "Fit complete")
        manager.spectra = {"SpecA": np.asarray([1.0, 2.0])}
        manager.integration_regions = []
        manager._current_quant_method = lambda: "cl"
        manager._current_quant_factor_text = lambda: "default"
        manager.spectrum_metadata = {
            "SpecA": {
                "last_fit_result": [
                    SimpleNamespace(
                        metadata=SimpleNamespace(General=SimpleNamespace(title="Fe_Ka")),
                        data=np.asarray([1.0, 2.0]),
                    )
                ],
                "last_fit_timestamp": "2026-01-02T00:00:00+00:00",
            }
        }

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "region_metadata.json"
            ok = SpectrumAnalysisManager._write_region_metadata_json(
                manager,
                path,
                scope="all",
                region_id=None,
            )
            self.assertTrue(ok)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertIn("model_fit", payload)
        self.assertIn("line_intensities", payload["model_fit"])
        self.assertIn("spectrum_summary", payload["model_fit"])
        self.assertEqual(payload["model_fit"]["line_intensities"][0]["line"], "Fe_Ka")
        self.assertEqual(payload["model_fit"]["spectrum_summary"][0]["spectrum"], "SpecA")
        self.assertEqual(payload["model_fit"]["spectrum_summary"][0]["line_count"], 1)
        self.assertEqual(
            payload["model_fit"]["spectrum_summary"][0]["last_fit_timestamp"],
            "2026-01-02T00:00:00+00:00",
        )


if __name__ == "__main__":
    unittest.main()
