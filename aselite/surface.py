"""表面与切面生成"""

import numpy as np
from typing import Tuple
from .atoms import Atoms


def surface(atoms: Atoms, indices: Tuple[int, int, int],
           layers: int, vacuum: float = 10.0,
           tol: float = 1e-10) -> Atoms:
    """生成表面结构（Miller 指数切面）

    Args:
        atoms: 晶体结构（需为正交或接近正交）
        indices: Miller 指数 (h, k, l)
        layers: 原子层数
        vacuum: 真空层厚度 (Å)
        tol: 原子层判定容差

    Returns:
        表面结构（c 轴垂直于表面）
    """
    h, k, l = indices

    if h == 0 and k == 0 and l == 0:
        raise ValueError("Miller indices cannot all be zero")

    # 归一化
    gcd = np.gcd(np.gcd(abs(h), abs(k)), abs(l))
    h, k, l = h // gcd, k // gcd, l // gcd

    # 表面法向量（倒易空间）
    cell = atoms._cell.array
    recip = np.linalg.inv(cell).T  # 倒易晶格
    normal = h * recip[0] + k * recip[1] + l * recip[2]
    normal = normal / np.linalg.norm(normal)

    # 构造新晶格：c 轴平行于法向
    # 选择两个垂直于法向的基矢
    if abs(normal[2]) < 0.9:
        v1 = np.cross(normal, [0, 0, 1])
    else:
        v1 = np.cross(normal, [1, 0, 0])
    v1 = v1 / np.linalg.norm(v1)

    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)

    # 投影原晶格基矢到新基
    # 找到能容纳原子的最小周期
    a_proj = np.array([np.dot(cell[i], v1) for i in range(3)])
    b_proj = np.array([np.dot(cell[i], v2) for i in range(3)])

    # 简单取最大投影长度（粗略估计）
    a_new = max(abs(a_proj)) * 2
    b_new = max(abs(b_proj)) * 2

    # 计算层间距
    d_hkl = 1.0 / np.linalg.norm(h * recip[0] + k * recip[1] + l * recip[2])

    # 投影所有原子到法向
    positions = atoms._positions
    proj_heights = positions @ normal

    # 按高度排序，识别层
    sorted_idx = np.argsort(proj_heights)
    sorted_heights = proj_heights[sorted_idx]

    # 折叠到一个周期（d_hkl）
    folded = sorted_heights % d_hkl
    unique_layers = []
    current_layer = [sorted_idx[0]]

    for i in range(1, len(sorted_idx)):
        if abs(folded[i] - folded[i-1]) < tol:
            current_layer.append(sorted_idx[i])
        else:
            unique_layers.append(current_layer)
            current_layer = [sorted_idx[i]]
    unique_layers.append(current_layer)

    # 选择前 layers 层
    n_layers = min(layers, len(unique_layers))
    selected_indices = []
    for i in range(n_layers):
        selected_indices.extend(unique_layers[i])

    # 提取原子
    new_symbols = [atoms._symbols[i] for i in selected_indices]
    new_positions = positions[selected_indices]
    new_masses = atoms.masses[selected_indices]

    # 平移到底部
    z_min = (new_positions @ normal).min()
    new_positions -= z_min * normal

    # 新晶格
    c_new = n_layers * d_hkl + vacuum
    new_cell = np.array([
        a_new * v1,
        b_new * v2,
        c_new * normal
    ])

    return Atoms(
        symbols=new_symbols,
        positions=new_positions,
        cell=new_cell,
        pbc=[True, True, False],
        masses=new_masses
    )
