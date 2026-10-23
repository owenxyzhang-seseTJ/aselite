"""晶胞类 - 处理周期性边界条件"""

import numpy as np
from typing import Union, List


class Cell:
    """晶胞类

    高效处理晶胞参数、倒易空间、最小镜像约定
    """

    __slots__ = ('_array',)

    def __init__(self, cell: Union[np.ndarray, List, None] = None):
        """初始化晶胞

        Args:
            cell: (3, 3) 晶胞向量矩阵，或 [a, b, c] 正交晶胞边长
        """
        if cell is None:
            self._array = np.zeros((3, 3))
        elif np.array(cell).shape == (3,):
            # [a, b, c] → 对角矩阵
            self._array = np.diag(cell)
        else:
            self._array = np.asarray(cell, dtype=float).reshape(3, 3)

    @property
    def array(self) -> np.ndarray:
        """晶胞向量矩阵 (3, 3)"""
        return self._array

    @array.setter
    def array(self, cell):
        self._array = np.asarray(cell, dtype=float).reshape(3, 3)

    def volume(self) -> float:
        """晶胞体积"""
        return abs(np.linalg.det(self._array))

    def lengths(self) -> np.ndarray:
        """晶胞边长 [a, b, c]"""
        return np.linalg.norm(self._array, axis=1)

    def angles(self) -> np.ndarray:
        """晶胞角度 [alpha, beta, gamma] (度)"""
        lengths = self.lengths()
        angles = np.zeros(3)

        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3

            cos_angle = np.dot(self._array[j], self._array[k]) / (lengths[j] * lengths[k])
            cos_angle = np.clip(cos_angle, -1, 1)  # 数值稳定性
            angles[i] = np.degrees(np.arccos(cos_angle))

        return angles

    def reciprocal(self) -> 'Cell':
        """倒易晶胞"""
        volume = self.volume()
        if volume < 1e-10:
            raise ValueError("晶胞体积接近零，无法计算倒易空间")

        rec_array = 2 * np.pi * np.linalg.inv(self._array).T
        return Cell(rec_array)

    def mic(self, vectors: np.ndarray) -> np.ndarray:
        """应用最小镜像约定 (Minimum Image Convention)

        将向量折叠到最近的周期镜像

        Args:
            vectors: (..., 3) 向量数组
        """
        # 转换到分数坐标
        fractional = np.linalg.solve(self._array.T, vectors.T).T

        # 折叠到 [-0.5, 0.5)
        fractional = fractional - np.round(fractional)

        # 转回笛卡尔坐标
        return fractional @ self._array

    def scaled_positions(self, positions: np.ndarray) -> np.ndarray:
        """笛卡尔坐标 → 分数坐标"""
        return np.linalg.solve(self._array.T, positions.T).T

    def cartesian_positions(self, scaled: np.ndarray) -> np.ndarray:
        """分数坐标 → 笛卡尔坐标"""
        return scaled @ self._array

    def __repr__(self) -> str:
        lengths = self.lengths()
        angles = self.angles()
        return (f"Cell([{lengths[0]:.3f}, {lengths[1]:.3f}, {lengths[2]:.3f}], "
                f"[{angles[0]:.1f}°, {angles[1]:.1f}°, {angles[2]:.1f}°])")
