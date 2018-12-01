
import logging
import math
import os
from unittest import TestCase

from nnwd.geometry import distance, fit_point
from pytils.invigilator import create_suite


def tests():
    return create_suite(Tests)


SMALL_MAX_T = 10
BIG_MAX_T = 50


class Tests(TestCase):
    def test_distance(self):
        self.assertEqual(distance([0, 1, 2], [0, 1, 2]), 0.0)
        self.assertEqual(distance([0, 1, 2], [-1, 1, 4]), math.sqrt(1**2 + 2**2))

    def test_fit_point1_d0(self):
        reference_point = [5, 3]
        target_distance = 0
        point, t = fit_point([reference_point], [target_distance])
        self.assertTrue(math.isclose(distance(point, reference_point), 0, abs_tol=.0001), point)
        self.assertLess(t, SMALL_MAX_T)

    def test_fit_point1_d1(self):
        reference_point = [5, 3]
        target_distance = 1
        point, t = fit_point([reference_point], [target_distance])
        self.assertTrue(math.isclose(distance(point, reference_point), target_distance, abs_tol=.0001), point)
        self.assertLess(t, SMALL_MAX_T)

    def test_fit_point2_vline(self):
        reference_points = [[0, -2], [0, 4]]
        target_distances = [2, 4]
        expected = [0, 0]
        point, t = fit_point(reference_points, target_distances)
        self.assertTrue(math.isclose(distance(point, expected), 0, abs_tol=0.0001))
        self.assertLess(t, SMALL_MAX_T)

    def test_fit_point2_hline(self):
        reference_points = [[-1, 0], [5, 0]]
        target_distances = [2, 4]
        expected = [1, 0]
        point, t = fit_point(reference_points, target_distances, visualize=True)
        self.assertTrue(math.isclose(distance(point, expected), 0, abs_tol=0.0001), point)
        self.assertLess(t, SMALL_MAX_T)

    def test_fit_point3(self):
        reference_points = [[math.sqrt(8), math.sqrt(8)], [0, -4], [-4, 0]]
        target_distances = [4, 4, 4]
        expected = [0, 0]
        point, t = fit_point(reference_points, target_distances)
        self.assertTrue(math.isclose(distance(point, expected), 0, abs_tol=0.0001))
        self.assertLess(t, BIG_MAX_T)

    def test_fit_point(self):
        reference_points_1 = [[0, -2, 3], [1, 0, -6], [2, 4, 9], [3, 6, -12]]
        reference_points_2 = [[3, -2, 3], [2, 0, -6], [1, 4, 9], [0, 6, -12]]
        target_distances_1 = [5, 4, 3, 2]
        target_distances_2 = [1, 2, 3, 4]

        point_11, t_11 = fit_point(reference_points_1, target_distances_1, visualize=True)
        self.assertTrue(all([not math.isnan(p) for p in point_11]), point_11)
        self.assertLess(t_11, BIG_MAX_T)

        point_12, t_12 = fit_point(reference_points_1, target_distances_2, visualize=True)
        self.assertTrue(all([not math.isnan(p) for p in point_12]), point_12)
        self.assertLess(t_12, BIG_MAX_T)

        point_21, t_21 = fit_point(reference_points_2, target_distances_1, visualize=True)
        self.assertTrue(all([not math.isnan(p) for p in point_21]), point_21)
        self.assertLess(t_21, BIG_MAX_T)

        point_22, t_22 = fit_point(reference_points_2, target_distances_2, visualize=True)
        self.assertTrue(all([not math.isnan(p) for p in point_22]), point_22)
        self.assertLess(t_22, BIG_MAX_T)

        self.assertGreater(distance(point_11, point_12), 1)
        self.assertGreater(distance(point_11, point_21), 1)
        self.assertGreater(distance(point_11, point_22), 1)

        self.assertGreater(distance(point_12, point_21), 1)
        self.assertGreater(distance(point_12, point_22), .75)

        self.assertGreater(distance(point_21, point_22), 1)

