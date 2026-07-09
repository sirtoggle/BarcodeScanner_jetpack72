import unittest
import warnings
from types import SimpleNamespace

import test9


class TorchGpuDetectionTests(unittest.TestCase):
    def test_ignores_jetson_compute_capability_warning(self) -> None:
        def fake_is_available() -> bool:
            warnings.warn(
                "Found CPU0 Orin which is of compute capability (CC) 8.7",
                UserWarning,
            )
            return True

        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=fake_is_available))

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            available = test9.is_gpu_available(fake_torch)

        self.assertTrue(available)
        self.assertEqual(
            [],
            [warning for warning in captured if "compute capability" in str(warning.message)],
        )


if __name__ == "__main__":
    unittest.main()
