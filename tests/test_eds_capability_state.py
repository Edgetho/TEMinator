import unittest

from eds_models import EDSCapabilityState


class EDSCapabilityStateTests(unittest.TestCase):
    def test_as_dict_includes_model_fit_results_flag(self):
        state = EDSCapabilityState(
            has_edx_data=True,
            has_elemental_maps=True,
            has_spectra=True,
            has_integration_regions=False,
            has_energy_calibration=True,
            has_timing_metadata=False,
            has_xray_lines=True,
            has_model_fit_results=True,
        )
        payload = state.as_dict()
        self.assertIn("has_model_fit_results", payload)
        self.assertTrue(payload["has_model_fit_results"])


if __name__ == "__main__":
    unittest.main()
