from pathlib import Path
from typing import Dict, List, Optional

import astropy.units as u
import numba as nb
import numpy as np
import popsynth as ps


class Cloud:
    def __init__(self, R: float = 1, zr_ratio: float = 1.0):
        """
        Creates an ellipsoidal cloud from which a
        GRB's emission would have to propagate

        :param R:
        :type R:
        :param zr_ratio:
        :type zr_ratio:
        :returns:

        """
        self._R = R
        self._R2 = R * R
        self._zr_ratio = zr_ratio
        self._Z = self._R * self._zr_ratio
        self._Z2 = self._Z * self._Z
        self._size_vec = np.array([self._R, self._R, self._Z])
        self._size_vec2 = self._size_vec ** 2

    def sample(self) -> float:
        """
        generate a random point inside the cloud
        and return its path length to the observer

        :returns:

        """
        p = self.generate_point_inside()

        u = self.get_unit_vector()

        pl = self.compute_path_length(p, u)

        return pl

    def generate_point_inside(self) -> np.ndarray:
        """
        generate a random point inside the cloud
        via rejection sampling

        :returns:

        """
        flag = True
        while flag:
            p = np.array(
                [
                    np.random.uniform(-self._R, self._R),
                    np.random.uniform(-self._R, self._R),
                    np.random.uniform(-self._Z, self._Z),
                ]
            )

            test_val = (p ** 2).dot(1.0 / self._size_vec2)

            if test_val <= 1:

                flag = False

        return p

    def get_unit_vector(self) -> np.ndarray:
        """
        generate a random unit vector

        :returns:

        """
        v = np.random.normal(size=3)
        u = v / np.linalg.norm(v)

        return u

    def compute_path_length(self, p, u) -> float:
        """
        compute the path length between two vectors

        :param p:
        :type p:
        :param u:
        :type u:
        :returns:

        """
        b = 2 * (p * u).dot(1.0 / self._size_vec2)
        a = (u ** 2).dot(1.0 / self._size_vec2)
        c = (p ** 2).dot(1.0 / self._size_vec2)

        l = (-b + np.sqrt(b * b - 4 * a * (c - 1))) / (2 * a)

        return l
