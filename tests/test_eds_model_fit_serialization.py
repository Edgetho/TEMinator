import unittest
from types import SimpleNamespace

import numpy as np

from viewer_edx import SpectrumAnalysisManager


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


if __name__ == "__main__":
    unittest.main()
