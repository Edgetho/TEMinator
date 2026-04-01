import unittest

import numpy as np

from viewer_edx import SpectrumAnalysisManager


class SpectralWindowIntegrationTests(unittest.TestCase):
    def test_window_integration_without_background(self):
        energy = np.arange(0.0, 100.0, 1.0)
        spectrum = np.zeros_like(energy)
        spectrum[10:15] = 5.0
        rows = SpectrumAnalysisManager._integrate_spectrum_windows(
            energy=energy,
            spectrum=spectrum,
            windows=[(10.0, 14.0)],
            lines=["Fe_Ka"],
            background_mode="none",
        )
        self.assertEqual(rows, [("Fe", 25.0)])

    def test_window_integration_auto_background_subtracts(self):
        energy = np.arange(0.0, 200.0, 1.0)
        spectrum = np.full_like(energy, 2.0)
        spectrum[100:105] = 10.0
        rows = SpectrumAnalysisManager._integrate_spectrum_windows(
            energy=energy,
            spectrum=spectrum,
            windows=[(100.0, 104.0)],
            lines=["Ni_Ka"],
            background_mode="auto",
        )
        # Raw signal is 50, background estimate is 2 * 5 channels => 10, net 40.
        self.assertEqual(rows, [("Ni", 40.0)])

    def test_multiple_lines_accumulate_by_element(self):
        energy = np.arange(0.0, 120.0, 1.0)
        spectrum = np.zeros_like(energy)
        spectrum[20:22] = 3.0
        spectrum[40:42] = 4.0
        rows = SpectrumAnalysisManager._integrate_spectrum_windows(
            energy=energy,
            spectrum=spectrum,
            windows=[(20.0, 21.0), (40.0, 41.0)],
            lines=["Cu_Ka", "Cu_La"],
            background_mode="none",
        )
        self.assertEqual(rows, [("Cu", 14.0)])


if __name__ == "__main__":
    unittest.main()
