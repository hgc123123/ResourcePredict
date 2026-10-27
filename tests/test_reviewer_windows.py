"""Temporal window cuts must not split equal-time groups or reorder events."""
import unittest
import pandas as pd

from scripts.prepare_reviewer_windows import prefix_size, window_labels


class ReviewerWindowTests(unittest.TestCase):
    def times(self):
        return pd.Series(pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC"))

    def test_full_prefix_keeps_every_event(self):
        self.assertEqual(prefix_size(self.times(), 1.0), 20)

    def test_cut_moves_left_across_equal_submission_group(self):
        times = self.times()
        times.iloc[11:14] = times.iloc[11]
        self.assertEqual(prefix_size(times, 0.60), 11)

    def test_unsorted_events_and_unusable_group_are_rejected(self):
        with self.assertRaises(ValueError):
            prefix_size(self.times().iloc[::-1], .60)
        with self.assertRaises(ValueError):
            prefix_size(pd.Series([self.times().iloc[0]] * 20), .60)

    def test_invalid_fractions_do_not_create_windows(self):
        for fraction in (0, -0.1, 1.1):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                prefix_size(self.times(), fraction)

    def test_window_names_are_unique_before_writing(self):
        self.assertEqual(window_labels([.60, .80, 1.0]), ["prefix_0600", "prefix_0800", "prefix_1000"])
        with self.assertRaises(ValueError):
            window_labels([.6001, .6002])
