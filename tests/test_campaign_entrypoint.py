import shlex
import unittest

import run_campaign


class CampaignEntrypointContractTest(unittest.TestCase):
    def test_default_args_pin_homogenized_rx_thermal_model(self):
        args = shlex.split(run_campaign.DEFAULT_ARGS)
        settings = {
            args[index + 1]
            for index, value in enumerate(args[:-1])
            if value == "--set"
        }
        self.assertIn("n_explicit_turns=0", settings)


if __name__ == "__main__":
    unittest.main()
