from pathlib import Path
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]


class LauncherConfigTests(unittest.TestCase):
    def test_removable_media_mount_propagates_daily_usb_changes(self) -> None:
        launcher = (PROJECT_DIR / "run_jetson.sh").read_text(encoding="utf-8")

        self.assertIn('"/media:/media:rslave"', launcher)
        self.assertIn('"/run/media:/run/media:rslave"', launcher)


if __name__ == "__main__":
    unittest.main()
