
import logging
import math
import os
from unittest import TestCase

from nnwd.domain import monotonic_paths
from pytils.invigilator import create_suite


def tests():
    return create_suite(Tests)


class Tests(TestCase):
    def test_distance(self):
        self.assertEqual(monotonic_paths([set([0, 2])], 5, False), set([(0,), (2,)]))
        self.assertEqual(monotonic_paths([set([0]), set([0, 1])], 5, False), set([(0, 1)]))
        self.assertEqual(monotonic_paths([set([0, 2]), set([2, 3])], 5, False), set([(0, 3), (2, 3), (0, 2)]))
        self.assertEqual(monotonic_paths([set([0]), set([0])], 5, False), set())

        self.assertEqual(monotonic_paths([set([0, 2])], 5, True), set([(0,)]))
        self.assertEqual(monotonic_paths([set([0]), set([0, 1])], 5, True), set([(0, 1)]))
        self.assertEqual(monotonic_paths([set([0, 2]), set([2, 3])], 5, True), set([(0, 2)]))
        self.assertEqual(monotonic_paths([set([0]), set([0])], 5, True), set())

        self.assertEqual(monotonic_paths([set([0, 1]), set([1, 2])], 3, False), set([(0, 1), (0, 2), (1, 2)]))

