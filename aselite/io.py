"""文件 I/O 模块 - 支持常见格式"""

import numpy as np
from pathlib import Path
from .atoms import Atoms


# 格式映射
_FORMAT_MAP = {
    '.cif': 'cif',
    '.vasp': 'vasp',
    '.poscar': 'vasp',
    '.contcar': 'vasp',
    '.xyz': 'xyz',
    '.pdb': 'pdb',
}


def read(filename: str, format: str = None, **kwargs) -> Atoms:
    """读取结构文件

    Args:
        filename: 文件路径
        format: 文件格式 (自动检测如果为 None)
    """
    path = Path(filename)

    if format is None:
        ext = path.suffix.lower()
        format = _FORMAT_MAP.get(ext)
        if format is None:
            raise ValueError(f"无法识别文件格式: {ext}")

    readers = {
        'cif': read_cif,
        'vasp': read_vasp,
        'xyz': read_xyz,
        'pdb': read_pdb,
    }

    if format not in readers:
        raise ValueError(f"不支持的格式: {format}")

    return readers[format](filename, **kwargs)


def write(filename: str, atoms: Atoms, format: str = None, **kwargs):
    """写入结构文件

    Args:
        filename: 文件路径
        atoms: Atoms 对象
        format: 文件格式 (从扩展名推断如果为 None)
    """
    path = Path(filename)

    if format is None:
        ext = path.suffix.lower()
        format = _FORMAT_MAP.get(ext)
        if format is None:
            raise ValueError(f"无法识别文件格式: {ext}")

    writers = {
        'cif': write_cif,
        'vasp': write_vasp,
        'xyz': write_xyz,
        'pdb': write_pdb,
    }

    if format not in writers:
        raise ValueError(f"不支持的格式: {format}")

    writers[format](filename, atoms, **kwargs)


# ========== CIF ==========

def read_cif(filename: str, expand_symmetry: bool = True, **kwargs) -> Atoms:
    """读取 CIF 文件（详细实现见 cif 模块）"""
    from .cif import read_cif as _read_cif
    return _read_cif(filename, expand_symmetry=expand_symmetry)


def write_cif(filename: str, atoms: Atoms):
    """写入 CIF 文件"""
    a, b, c = atoms.cell.lengths()
    alpha, beta, gamma = atoms.cell.angles()

    scaled = atoms.cell.scaled_positions(atoms.positions)

    with open(filename, 'w') as f:
        f.write("data_structure\n\n")
        f.write(f"_cell_length_a    {a:.6f}\n")
        f.write(f"_cell_length_b    {b:.6f}\n")
        f.write(f"_cell_length_c    {c:.6f}\n")
        f.write(f"_cell_angle_alpha {alpha:.3f}\n")
        f.write(f"_cell_angle_beta  {beta:.3f}\n")
        f.write(f"_cell_angle_gamma {gamma:.3f}\n\n")

        f.write("loop_\n")
        f.write("_atom_site_label\n")
        f.write("_atom_site_type_symbol\n")
        f.write("_atom_site_fract_x\n")
        f.write("_atom_site_fract_y\n")
        f.write("_atom_site_fract_z\n")

        for i, (sym, pos) in enumerate(zip(atoms.symbols, scaled)):
            f.write(f"{sym}{i+1:d}  {sym}  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}\n")


# ========== VASP POSCAR ==========

def read_vasp(filename: str) -> Atoms:
    """读取 VASP POSCAR/CONTCAR"""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # 第一行: 注释
    # 第二行: 缩放因子
    scale = float(lines[1].strip())

    # 第 3-5 行: 晶胞向量
    cell = np.array([
        [float(x) for x in lines[2].split()],
        [float(x) for x in lines[3].split()],
        [float(x) for x in lines[4].split()],
    ]) * scale

    # 第 6 行: 元素符号 (VASP 5+)
    elements = lines[5].split()

    # 第 7 行: 各元素数量
    counts = [int(x) for x in lines[6].split()]

    symbols = []
    for elem, count in zip(elements, counts):
        symbols.extend([elem] * count)

    # 第 8 行: 坐标类型
    coord_type = lines[7].strip()[0].upper()

    # 第 9 行开始: 坐标
    n_atoms = sum(counts)
    positions = np.array([
        [float(x) for x in lines[8 + i].split()[:3]]
        for i in range(n_atoms)
    ])

    if coord_type == 'D':  # Direct (分数坐标)
        positions = positions @ cell

    return Atoms(
        symbols=symbols,
        positions=positions,
        cell=cell,
        pbc=True
    )


def write_vasp(filename: str, atoms: Atoms, direct: bool = True):
    """写入 VASP POSCAR

    Args:
        direct: True=分数坐标, False=笛卡尔坐标
    """
    from collections import Counter

    formula = atoms.get_chemical_formula()
    counts = Counter(atoms.symbols)

    # 保持元素顺序
    unique_symbols = []
    seen = set()
    for s in atoms.symbols:
        if s not in seen:
            unique_symbols.append(s)
            seen.add(s)

    with open(filename, 'w') as f:
        f.write(f"{formula}\n")
        f.write("1.0\n")

        # 晶胞
        for vec in atoms.cell.array:
            f.write(f"  {vec[0]:12.6f}  {vec[1]:12.6f}  {vec[2]:12.6f}\n")

        # 元素
        f.write("  " + "  ".join(unique_symbols) + "\n")

        # 数量
        f.write("  " + "  ".join(str(counts[s]) for s in unique_symbols) + "\n")

        # 坐标类型
        f.write("Direct\n" if direct else "Cartesian\n")

        # 坐标 (按元素分组)
        for sym in unique_symbols:
            indices = [i for i, s in enumerate(atoms.symbols) if s == sym]

            for idx in indices:
                if direct:
                    pos = atoms.cell.scaled_positions(atoms.positions[idx:idx+1])[0]
                else:
                    pos = atoms.positions[idx]

                f.write(f"  {pos[0]:12.6f}  {pos[1]:12.6f}  {pos[2]:12.6f}\n")


# ========== XYZ ==========

def read_xyz(filename: str) -> Atoms:
    """读取 XYZ 文件"""
    with open(filename, 'r') as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    # 第二行是注释

    symbols = []
    positions = []

    for i in range(n_atoms):
        parts = lines[2 + i].split()
        symbols.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return Atoms(
        symbols=symbols,
        positions=np.array(positions),
        pbc=False
    )


def write_xyz(filename: str, atoms: Atoms):
    """写入 XYZ 文件"""
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{atoms.get_chemical_formula()}\n")

        for sym, pos in zip(atoms.symbols, atoms.positions):
            f.write(f"{sym:2s}  {pos[0]:12.6f}  {pos[1]:12.6f}  {pos[2]:12.6f}\n")


# ========== PDB ==========

def read_pdb(filename: str) -> Atoms:
    """读取 PDB 文件 (简化版)"""
    symbols = []
    positions = []

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                symbol = line[76:78].strip()
                if not symbol:
                    symbol = line[12:16].strip()[0]  # 从原子名提取

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                symbols.append(symbol)
                positions.append([x, y, z])

    return Atoms(
        symbols=symbols,
        positions=np.array(positions),
        pbc=False
    )


def write_pdb(filename: str, atoms: Atoms):
    """写入 PDB 文件"""
    with open(filename, 'w') as f:
        f.write("HEADER    Generated by ASE-lite\n")

        for i, (sym, pos) in enumerate(zip(atoms.symbols, atoms.positions), 1):
            f.write(
                f"ATOM  {i:5d}  {sym:<4s}MOL     1    "
                f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                f"  1.00  0.00          {sym:>2s}\n"
            )

        f.write("END\n")


# ========== 工具函数 ==========

def cell_from_parameters(a: float, b: float, c: float,
                         alpha: float, beta: float, gamma: float) -> np.ndarray:
    """从晶胞参数构建晶胞矩阵

    Args:
        a, b, c: 晶胞边长 (Å)
        alpha, beta, gamma: 晶胞角度 (度)

    Returns:
        (3, 3) 晶胞向量矩阵
    """
    alpha_r = np.radians(alpha)
    beta_r = np.radians(beta)
    gamma_r = np.radians(gamma)

    cos_alpha = np.cos(alpha_r)
    cos_beta = np.cos(beta_r)
    cos_gamma = np.cos(gamma_r)
    sin_gamma = np.sin(gamma_r)

    # a 沿 x 轴
    ax = a
    ay = 0.0
    az = 0.0

    # b 在 xy 平面
    bx = b * cos_gamma
    by = b * sin_gamma
    bz = 0.0

    # c 由其他参数确定
    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz = np.sqrt(c**2 - cx**2 - cy**2)

    return np.array([
        [ax, ay, az],
        [bx, by, bz],
        [cx, cy, cz]
    ])
