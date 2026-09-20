import unittest

import torch

from phase4_helpers import tensor_dataset
from phase7_helpers import support_contract, support_dataset


class SupportEligibilityContractTests(unittest.TestCase):
    def test_manual_handoff_cases_preserve_frozen_outcomes(self):
        cases = {item["case"]: item for item in support_contract().interface_cases}
        self.assertEqual((cases["positive_before_censor"]["label"], cases["positive_before_censor"]["eligible"]), (1, True))
        self.assertEqual((cases["no_event_before_early_exit"]["label"], cases["no_event_before_early_exit"]["eligible"]), (None, False))
        self.assertEqual((cases["already_on_continuation_other_component_negative"]["label"], cases["already_on_continuation_other_component_negative"]["eligible"]), (0, True))
        for name in ("vasopressor_initiation_only", "invasive_ventilation_initiation_only", "both_initiate"):
            self.assertEqual((cases[name]["label"], cases[name]["eligible"]), (1, True))
        self.assertEqual((cases["neither_with_full_followup"]["label"], cases["neither_with_full_followup"]["eligible"]), (0, True))

    def test_overlay_consumes_labels_without_exposing_audit_metadata(self):
        dataset = support_dataset("train")
        first = dataset[0]
        second = dataset[1]
        self.assertEqual(first["targets"]["organ_support"], 1)
        self.assertTrue(first["eligibility"]["organ_support"])
        self.assertEqual(second["targets"]["organ_support"], 0)
        self.assertTrue(second["eligibility"]["organ_support"])
        self.assertNotIn("audit_metadata", first)
        self.assertNotIn("qualifying_onset_hours_after_t", first["identifiers"])

    def test_future_outcome_and_label_changes_do_not_change_model_input(self):
        original = tensor_dataset("train")[0]
        duplicate = dict(original)
        duplicate_targets = dict(original["targets"])
        duplicate_targets["organ_support"] = 0
        duplicate["targets"] = duplicate_targets
        for field in ("sequence", "padding_mask", "observation_mask"):
            self.assertTrue(torch.equal(original[field], duplicate[field]))
        self.assertEqual(original["tslo"], duplicate["tslo"])
        self.assertEqual(original["static_features"], duplicate["static_features"])
        self.assertNotEqual(original["targets"]["organ_support"], duplicate["targets"]["organ_support"])


if __name__ == "__main__":
    unittest.main()
