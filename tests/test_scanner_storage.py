import csv
import os
import tempfile
import unittest

from scanner_storage import write_image_file, write_scan_row


class ScannerStorageTests(unittest.TestCase):
    def test_missing_output_directory_is_not_recreated(self) -> None:
        with tempfile.TemporaryDirectory() as parent_dir:
            missing_dir = os.path.join(parent_dir, "removed-usb")

            with self.assertRaises(FileNotFoundError):
                write_scan_row(missing_dir, "123456", "2026-07-13_12-00-00")

            self.assertFalse(os.path.exists(missing_dir))

    def test_csv_row_is_flushed_to_disk(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            filename = write_scan_row(
                output_dir,
                "123456",
                "2026-07-13_12-00-00",
                "07/13/2026",
                "JANE SMITH",
            )

            with open(filename, newline="", encoding="utf-8") as input_file:
                self.assertEqual(
                    [["123456", "2026-07-13_12-00-00", "07/13/2026", "JANE SMITH"]],
                    list(csv.reader(input_file)),
                )

    def test_image_encoder_failure_is_not_reported_as_success(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaises(OSError):
                write_image_file(
                    output_dir,
                    object(),
                    "2026-07-13_12-00-00",
                    lambda _filename, _image: False,
                )

            self.assertEqual([], os.listdir(output_dir))

    def test_image_is_atomically_moved_to_final_name(self) -> None:
        def successful_writer(filename: str, _image: object) -> bool:
            with open(filename, "wb") as output_file:
                output_file.write(b"jpeg")
            return True

        with tempfile.TemporaryDirectory() as output_dir:
            filename = write_image_file(
                output_dir,
                object(),
                "2026-07-13_12-00-00",
                successful_writer,
            )

            with open(filename, "rb") as input_file:
                self.assertEqual(b"jpeg", input_file.read())
            self.assertEqual(["2026-07-13_12-00-00.jpg"], os.listdir(output_dir))


if __name__ == "__main__":
    unittest.main()
