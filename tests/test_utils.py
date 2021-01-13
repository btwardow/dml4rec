from unittest import TestCase
from rec.utils import iso_timestamp_to_epoch

import pytest


@pytest.mark.skip(reason="Different behavior for CI/local")
class TestUtils(TestCase):
    def test_timestamp_iso_to_epoch(self):
        ts = "2016-02-06T19:54:18.000"
        epoch = iso_timestamp_to_epoch(ts)
        self.assertEqual(epoch, 1454788458)
