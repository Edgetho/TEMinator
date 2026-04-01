import unittest

from eds_metadata import build_eds_metadata_context


class MetadataContextTests(unittest.TestCase):
    def test_prefers_detector_values_and_begin_energy_fallback(self):
        original = {
            "Detectors": {
                "Detector-7": {
                    "DetectorType": "AnalyticalDetector",
                    "Dispersion": 5,
                    "BeginEnergy": 150,
                    "LiveTime": 10,
                    "RealTime": 11,
                }
            },
            "Sample": {"xray_lines": ["Fe_Ka"]},
        }
        ctx = build_eds_metadata_context(original_meta=original, mapped_meta={})
        self.assertEqual(ctx.dispersion_ev_per_channel, 5.0)
        self.assertEqual(ctx.offset_ev, 150.0)
        self.assertEqual(ctx.live_time_s, 10.0)
        self.assertEqual(ctx.real_time_s, 11.0)
        self.assertEqual(list(ctx.xray_lines), ["Fe_Ka"])

    def test_uses_custom_properties_when_detector_missing(self):
        original = {
            "CustomProperties": {
                "Detectors[SuperXG11].Dispersion": {"type": "double", "value": 3},
                "Detectors[SuperXG11].SpectrumBeginEnergy": {"type": "long", "value": 120},
            }
        }
        mapped = {
            "Acquisition_instrument": {"TEM": {"beam_energy": 300.0}},
            "Sample": {"xray_lines": ["Ni_Ka", "Te_La"]},
        }
        ctx = build_eds_metadata_context(original_meta=original, mapped_meta=mapped)
        self.assertEqual(ctx.beam_energy_ev, 300.0)
        self.assertEqual(ctx.dispersion_ev_per_channel, 3.0)
        self.assertEqual(ctx.offset_ev, 120.0)
        self.assertEqual(list(ctx.xray_lines), ["Ni_Ka", "Te_La"])


if __name__ == "__main__":
    unittest.main()
