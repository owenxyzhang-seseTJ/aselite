"""几何操作工具"""

import numpy as np
from typing import Union


def rotation_matrix(angle: float, axis: Union[str, np.ndarray]) -> np.ndarray:
    """生成旋转矩阵 (Rodrigues' rotation formula)

    Args:
        angle: 旋转角度 (弧度)
        axis: 旋转轴向量或 'x'/'y'/'z'

    Returns:
        (3, 3) 旋转矩阵
    """
    if isinstance(axis, str):
        axis = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[axis]

    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)  # 归一化

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x, y, z = axis

    # Rodrigues 公式
    R = np.array([
        [cos_a + x*x*(1-cos_a),     x*y*(1-cos_a) - z*sin_a,  x*z*(1-cos_a) + y*sin_a],
        [y*x*(1-cos_a) + z*sin_a,   cos_a + y*y*(1-cos_a),    y*z*(1-cos_a) - x*sin_a],
        [z*x*(1-cos_a) - y*sin_a,   z*y*(1-cos_a) + x*sin_a,  cos_a + z*z*(1-cos_a)]
    ])

    return R


def find_mic_vectors(positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    """批量应用最小镜像约定

    Args:
        positions: (n, 3) 位置数组
        cell: (3, 3) 晶胞矩阵
        pbc: (3,) 周期性边界条件

    Returns:
        (n, 3) 折叠后的位置
    """
    if not pbc.any():
        return positions

    # 转换到分数坐标
    fractional = np.linalg.solve(cell.T, positions.T).T

    # 应用周期性
    fractional[:, pbc] = fractional[:, pbc] - np.round(fractional[:, pbc])

    # 转回笛卡尔坐标
    return fractional @ cell
