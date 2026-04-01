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


if __name__ == "__main__":
    unittest.main()
