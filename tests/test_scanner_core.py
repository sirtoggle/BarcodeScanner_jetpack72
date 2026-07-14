import re
import tempfile
import unittest
from pathlib import Path

from scanner_core import (
    ConsensusTracker,
    extract_card_date,
    extract_full_name,
    extract_id,
    load_logo_words,
    path_is_on_mount,
    scale_box,
)


class MountPathTests(unittest.TestCase):
    def test_accepts_a_directory_inside_the_current_mount(self) -> None:
        self.assertTrue(path_is_on_mount("/media/user/USB/scans", ["/media/user/USB"]))

    def test_rejects_an_old_or_similarly_named_mount(self) -> None:
        self.assertFalse(path_is_on_mount("/media/user/USB-old", ["/media/user/USB"]))


class ScaleBoxTests(unittest.TestCase):
    def test_maps_detection_coordinates_back_to_original_frame(self) -> None:
        self.assertEqual(
            (200, 100, 600, 400),
            scale_box((100, 50, 300, 200), 0.5, (1080, 1920, 3)),
        )

    def test_clamps_box_to_original_frame(self) -> None:
        self.assertEqual(
            (1900, 1070, 20, 10),
            scale_box((950, 535, 100, 100), 0.5, (1080, 1920, 3)),
        )


class ExtractIdTests(unittest.TestCase):
    def test_prefers_confidence_over_unrelated_longer_number(self) -> None:
        results = [
            (None, "20260713", 0.51),
            (None, "1234567", 0.96),
        ]

        self.assertEqual("1234567", extract_id(results))

    def test_rejoins_spaces_and_hyphens_within_one_id_region(self) -> None:
        results = [(None, "123-456 789", 0.91)]

        self.assertEqual("123456789", extract_id(results))

    def test_enforces_expected_length_pattern_and_confidence(self) -> None:
        results = [
            (None, "99123456", 0.95),
            (None, "12123456", 0.39),
            (None, "12123456", 0.90),
        ]

        self.assertEqual(
            "12123456",
            extract_id(
                results,
                expected_length=8,
                pattern=re.compile(r"12\d{6}"),
                min_confidence=0.40,
            ),
        )

    def test_rejects_date_when_slashes_are_read_as_sevens(self) -> None:
        results = [
            (None, "0771372026", 0.99),
            (None, "12345678", 0.80),
        ]

        self.assertEqual("12345678", extract_id(results))

    def test_returns_none_when_only_candidate_is_a_corrupted_date(self) -> None:
        self.assertIsNone(extract_id([(None, "0771372026", 0.99)]))


class ExtractCardDateTests(unittest.TestCase):
    def test_extracts_and_normalizes_date_with_slashes(self) -> None:
        self.assertEqual(
            "07/13/2026",
            extract_card_date([(None, "07/13/2026", 0.91)]),
        )

    def test_recovers_date_when_both_slashes_are_read_as_sevens(self) -> None:
        self.assertEqual(
            "07/13/2026",
            extract_card_date([(None, "0771372026", 0.91)]),
        )

    def test_recovers_date_when_only_one_slash_is_read_as_seven(self) -> None:
        self.assertEqual(
            "07/13/2026",
            extract_card_date([(None, "07713/2026", 0.91)]),
        )

    def test_rejects_invalid_calendar_date(self) -> None:
        self.assertIsNone(extract_card_date([(None, "1374072026", 0.99)]))


class ExtractFullNameTests(unittest.TestCase):
    def test_extracts_name_and_removes_field_label(self) -> None:
        self.assertEqual(
            "JANE Q SMITH",
            extract_full_name([(None, "NAME JANE Q SMITH", 0.92)]),
        )

    def test_ignores_single_word_logo(self) -> None:
        results = [
            (None, "MEGACORP", 0.99),
            (None, "JOHN DOE", 0.82),
        ]

        self.assertEqual("JOHN DOE", extract_full_name(results))

    def test_removes_generic_organization_words(self) -> None:
        results = [
            (None, "ACME HEALTH SYSTEM", 0.99),
            (None, "JOHN DOE", 0.80),
        ]

        self.assertEqual("JOHN DOE", extract_full_name(results))

    def test_custom_logo_words_reject_entire_region(self) -> None:
        results = [
            (None, "EAST BAY HOLDINGS", 0.99),
            (None, "JANE SMITH", 0.81),
        ]

        self.assertEqual(
            "JANE SMITH",
            extract_full_name(results, logo_words=("EAST", "BAY")),
        )

    def test_joins_separate_name_boxes_on_the_same_line(self) -> None:
        results = [
            ([[10, 20], [90, 20], [90, 45], [10, 45]], "JANE", 0.91),
            ([[110, 21], [210, 21], [210, 46], [110, 46]], "SMITH", 0.89),
        ]

        self.assertEqual("JANE SMITH", extract_full_name(results))


class LoadLogoWordsTests(unittest.TestCase):
    def test_loads_phrases_and_ignores_comments_blanks_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logos.txt"
            path.write_text(
                "# Card branding\n\nWynn Rewards\nEncore Boston Harbor\nwynn rewards\n",
                encoding="utf-8",
            )

            self.assertEqual(
                ("Wynn Rewards", "Encore Boston Harbor"),
                load_logo_words(str(path)),
            )


class ConsensusTrackerTests(unittest.TestCase):
    def test_confirms_three_matching_recent_readings(self) -> None:
        tracker = ConsensusTracker(required_matches=3, window_size=5)

        self.assertEqual((None, 1), tracker.observe("123456"))
        self.assertEqual((None, 0), tracker.observe(None))
        self.assertEqual((None, 2), tracker.observe("123456"))
        self.assertEqual(("123456", 3), tracker.observe("123456"))

    def test_does_not_confirm_a_stale_candidate(self) -> None:
        tracker = ConsensusTracker(required_matches=3, window_size=5)
        tracker.observe("123456")
        tracker.observe("123456")
        tracker.observe("123456")

        self.assertEqual((None, 1), tracker.observe("999999"))

    def test_reset_discards_previous_readings(self) -> None:
        tracker = ConsensusTracker(required_matches=2, window_size=3)
        tracker.observe("123456")
        tracker.reset()

        self.assertEqual((None, 1), tracker.observe("123456"))


if __name__ == "__main__":
    unittest.main()
