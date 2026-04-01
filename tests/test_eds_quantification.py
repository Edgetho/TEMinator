import unittest

from eds_models import EDSQuantResultRow
from eds_quantification import (
    EDS_CSV_HEADER,
    EDSQuantificationService,
    QuantificationRequest,
    quant_rows_to_csv_records,
)


class QuantificationServiceTests(unittest.TestCase):
    def test_cl_method_uses_factor_defaults_and_warns(self):
        service = EDSQuantificationService()
        req = QuantificationRequest(
            region_id=7,
            element_counts={"Fe": 1000.0, "Pt": 2000.0},
            method="CL",
            factor_text="Fe=2.0",
        )
        rows = list(service.quantify(req))
        self.assertEqual(len(rows), 2)
        self.assertEqual([r.element for r in rows], ["Fe", "Pt"])
        self.assertTrue(any("k-factor missing for Pt" in w for w in rows[0].warnings))

    def test_custom_method_applies_multiplicative_factors(self):
        service = EDSQuantificationService()
        req = QuantificationRequest(
            region_id=1,
            element_counts={"Ni": 10.0, "Te": 10.0},
            method="CUSTOM",
            factor_text="Ni=2, Te=1",
        )
        rows = {r.element: r for r in service.quantify(req)}
        self.assertAlmostEqual(rows["Ni"].atomic_percent or 0.0, 66.666, places=2)
        self.assertAlmostEqual(rows["Te"].atomic_percent or 0.0, 33.333, places=2)

    def test_csv_schema_formatter_is_stable(self):
        rows = [
            EDSQuantResultRow(
                region_id=3,
                element="Ta",
                counts=123.0,
                weight_percent=60.0,
                atomic_percent=40.0,
                method="CL",
                warnings=("example warning",),
            )
        ]
        records = quant_rows_to_csv_records(rows)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(tuple(record.keys()), EDS_CSV_HEADER)
        self.assertEqual(record["region_id"], "3")
        self.assertEqual(record["element"], "Ta")
        self.assertEqual(record["method"], "CL")
        self.assertIn("example warning", record["warnings"])

    def test_zeta_method_warns_for_missing_required_metadata(self):
        service = EDSQuantificationService()
        req = QuantificationRequest(
            region_id=9,
            element_counts={"Fe": 10.0, "Pt": 20.0},
            method="zeta",
            factor_text="Fe=1.0, Pt=1.0",
            beam_current_na=None,
            real_time_s=None,
        )
        rows = list(service.quantify(req))
        self.assertEqual(len(rows), 2)
        warnings = "\n".join(rows[0].warnings)
        self.assertIn("zeta method missing beam_current", warnings)
        self.assertIn("zeta method missing real_time", warnings)

    def test_cross_section_method_warns_for_probe_area(self):
        service = EDSQuantificationService()
        req = QuantificationRequest(
            region_id=11,
            element_counts={"Ni": 5.0, "Te": 5.0},
            method="cross_section",
            factor_text="Ni=1.0, Te=1.0",
            beam_current_na=0.5,
            real_time_s=1.0,
            probe_area_nm2=None,
        )
        rows = list(service.quantify(req))
        self.assertEqual(len(rows), 2)
        warnings = "\n".join(rows[0].warnings)
        self.assertIn("cross_section method missing probe_area", warnings)

    def test_cl_absorption_requires_thickness(self):
        service = EDSQuantificationService()
        req = QuantificationRequest(
            region_id=3,
            element_counts={"Ta": 100.0},
            method="CL",
            factor_text="Ta=1.0",
            absorption_correction=True,
            thickness_nm=None,
            detector_count=1,
        )
        rows = list(service.quantify(req))
        self.assertEqual(len(rows), 1)
        self.assertIn("CL absorption correction requires thickness", "\n".join(rows[0].warnings))


if __name__ == "__main__":
    unittest.main()
