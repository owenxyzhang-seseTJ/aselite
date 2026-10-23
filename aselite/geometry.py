"""几何操作工具 - 键角、二面角、旋转等"""

import numpy as np
from typing import Union, List, Tuple


def get_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两向量夹角（弧度）

    Args:
        v1, v2: (3,) 向量

    Returns:
        夹角 [0, π]
    """
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # 避免数值误差导致超出 [-1, 1]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def get_angles_batch(v1s: np.ndarray, v2s: np.ndarray) -> np.ndarray:
    """批量计算向量对的夹角（向量化）

    Args:
        v1s, v2s: (n, 3) 向量数组

    Returns:
        (n,) 夹角数组
    """
    # 点积: (n,)
    dots = np.sum(v1s * v2s, axis=1)
    norms1 = np.linalg.norm(v1s, axis=1)
    norms2 = np.linalg.norm(v2s, axis=1)
    cos_angles = dots / (norms1 * norms2)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    return np.arccos(cos_angles)


def get_dihedral(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    """计算二面角（四个连续点 p0-p1-p2-p3 定义的扭转角）

    输入三个向量: v1 = p1-p0, v2 = p2-p1, v3 = p3-p2
    计算平面 (v1, v2) 与平面 (v2, v3) 的夹角，带符号

    Args:
        v1, v2, v3: (3,) 向量

    Returns:
        二面角 [-π, π]，右手系，沿 v2 看顺时针为正
    """
    # 法向量
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)

    # 避免零向量（三点共线）
    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)
    if norm_n1 < 1e-10 or norm_n2 < 1e-10:
        return 0.0

    n1 = n1 / norm_n1
    n2 = n2 / norm_n2

    # 夹角
    cos_angle = np.dot(n1, n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    # 确定符号：用 v2 × n1 · n2 判断
    # 如果 (v2 × n1) · n2 < 0，说明 n2 在 v2 × n1 的反方向，角度取负
    sign = np.dot(np.cross(v2, n1), n2)
    if sign < 0:
        angle = -angle

    return angle


def get_dihedrals_batch(v1s: np.ndarray, v2s: np.ndarray, v3s: np.ndarray) -> np.ndarray:
    """批量计算二面角（向量化）

    Args:
        v1s, v2s, v3s: (n, 3) 向量数组

    Returns:
        (n,) 二面角数组
    """
    n1s = np.cross(v1s, v2s)
    n2s = np.cross(v2s, v3s)

    norm_n1s = np.linalg.norm(n1s, axis=1, keepdims=True)
    norm_n2s = np.linalg.norm(n2s, axis=1, keepdims=True)

    # 避免零向量
    valid = (norm_n1s.ravel() > 1e-10) & (norm_n2s.ravel() > 1e-10)
    angles = np.zeros(len(v1s))

    n1s_norm = n1s[valid] / norm_n1s[valid]
    n2s_norm = n2s[valid] / norm_n2s[valid]

    cos_angles = np.sum(n1s_norm * n2s_norm, axis=1)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles[valid] = np.arccos(cos_angles)

    # 符号
    signs = np.sum(np.cross(v2s[valid], n1s_norm) * n2s_norm, axis=1)
    angles[valid] = np.where(signs < 0, -angles[valid], angles[valid])

    return angles


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
