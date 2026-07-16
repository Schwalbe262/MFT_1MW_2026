import json
import sys
import tempfile
import unittest
from pathlib import Path


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

import audit_standalone_live as audit


class StandaloneLiveAuditTest(unittest.TestCase):
    def test_stage_classification_uses_furthest_marker(self):
        self.assertEqual(
            audit.classify_stdout(
                "Added design 'maxwell_matrix'\n"
                "Added design 'maxwell_cap'\n"
                "Active Design set to maxwell_matrix1\n"
                "Added design 'icepak_thermal'\n"
                "Solving design setup ThermalSetup\n"
            ),
            "thermal_solving",
        )
        self.assertEqual(
            audit.classify_stdout(
                "Added design 'maxwell_cap'\n"
                "Error in Solving Setup Setup1\n"
            ),
            "cap_solve_error",
        )

    def test_collector_cache_is_fail_closed_and_filters_ids(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "collect_cache.json"
            path.write_text(json.dumps({
                "harvested": [1, 2, True, "3"],
                "nodata": [4],
            }), encoding="utf-8")
            self.assertEqual(
                audit._load_collector_cache(path),
                {"harvested": {1, 2}, "nodata": {4}},
            )
            path.write_text("not-json", encoding="utf-8")
            self.assertEqual(audit._load_collector_cache(path), {})


if __name__ == "__main__":
    unittest.main()
