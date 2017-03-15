from __future__ import print_function

import unittest

import numpy as np

from DVIDSparkServices.reconutils.morpho import split_disconnected_bodies


class TestSplitDisconnectedBodies(unittest.TestCase):
    def test_splits_bodies(self):
        x = np.array([[[1, 0, 1]]], dtype=np.dtype("uint32"))
        array_result, map_result = split_disconnected_bodies(x)
        expected_array = np.array([[[1, 0, 2]]], dtype=np.dtype("uint32"))
        np.testing.assert_array_equal(expected_array, array_result)
        expected_map = {1: 1, 2: 1}
        self.assertDictEqual(expected_map, map_result)

    def test_new_bodies_start_with_the_max(self):
        x = np.array([[[4, 7, 4]]], dtype=np.dtype("uint32"))
        array_result, map_result = split_disconnected_bodies(x)
        expected_array = np.array([[[4, 7, 8]]], dtype=np.dtype("uint32"))
        np.testing.assert_array_equal(expected_array, array_result)
        expected_map = {4: 4, 8: 4}
        self.assertDictEqual(expected_map, map_result)

    def test_works_with_big_values(self):
        BIG_INT = long(2 ** 50)
        x = np.array([[[BIG_INT, 0, BIG_INT]]], dtype=np.dtype("uint64"))
        array_result, map_result = split_disconnected_bodies(x)
        expected_array = np.array([[[BIG_INT, 0, BIG_INT + 1]]], dtype=np.dtype("uint64"))
        np.testing.assert_array_equal(expected_array, array_result)
