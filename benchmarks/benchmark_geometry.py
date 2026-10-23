"""几何操作性能对比：aselite vs ASE"""

import time
import numpy as np
from ase import Atoms as ASEAtoms
from aselite import Atoms as ALAtoms


def benchmark_angles(n_angles=1000):
    """键角计算性能"""
    # 使用规则网格避免原子重叠
    n = 10
    x = np.linspace(0, 9, n)
    positions = np.array([[i, j, k] for i in x for j in x for k in x[:2]])  # 200 atoms
    n_atoms = len(positions)

    # 生成有效的键角索引（确保不会选到相同原子）
    np.random.seed(42)
    angles_list = []
    for _ in range(n_angles):
        indices = np.random.choice(n_atoms, 3, replace=False)
        angles_list.append(indices)
    angles_list = np.array(angles_list)

    # ASE
    ase_atoms = ASEAtoms('C' * n_atoms, positions=positions)

    start = time.perf_counter()
    for i, j, k in angles_list:
        try:
            ase_atoms.get_angle(i, j, k)
        except ZeroDivisionError:
            pass
    ase_time = time.perf_counter() - start

    # aselite (批量)
    al_atoms = ALAtoms(['C'] * n_atoms, positions=positions)

    start = time.perf_counter()
    al_atoms.get_angles(angles_list)
    al_time = time.perf_counter() - start

    print(f"键角计算 ({n_angles} 个):")
    print(f"  ASE (循环):     {ase_time*1000:.2f} ms")
    print(f"  aselite (向量): {al_time*1000:.2f} ms")
    print(f"  加速比:         {ase_time/al_time:.1f}×")
    print()


def benchmark_dihedrals(n_dihedrals=1000):
    """二面角计算性能"""
    # 使用规则网格
    n = 10
    x = np.linspace(0, 9, n)
    positions = np.array([[i, j, k] for i in x for j in x for k in x[:2]])
    n_atoms = len(positions)

    # 生成有效的二面角索引
    np.random.seed(43)
    dihedrals_list = []
    for _ in range(n_dihedrals):
        indices = np.random.choice(n_atoms, 4, replace=False)
        dihedrals_list.append(indices)
    dihedrals_list = np.array(dihedrals_list)

    # ASE
    ase_atoms = ASEAtoms('C' * n_atoms, positions=positions)

    start = time.perf_counter()
    for i, j, k, l in dihedrals_list:
        try:
            ase_atoms.get_dihedral(i, j, k, l)
        except ZeroDivisionError:
            pass
    ase_time = time.perf_counter() - start

    # aselite (批量)
    al_atoms = ALAtoms(['C'] * n_atoms, positions=positions)

    start = time.perf_counter()
    al_atoms.get_dihedrals(dihedrals_list)
    al_time = time.perf_counter() - start

    print(f"二面角计算 ({n_dihedrals} 个):")
    print(f"  ASE (循环):     {ase_time*1000:.2f} ms")
    print(f"  aselite (向量): {al_time*1000:.2f} ms")
    print(f"  加速比:         {ase_time/al_time:.1f}×")
    print()


def benchmark_supercell():
    """超胞生成性能"""
    # FCC 结构
    positions = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    cell = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # ASE
    ase_atoms = ASEAtoms('Cu4', positions=positions, cell=cell, pbc=True)
    start = time.perf_counter()
    ase_super = ase_atoms.repeat([5, 5, 5])
    ase_time = time.perf_counter() - start

    # aselite
    al_atoms = ALAtoms(['Cu'] * 4, positions=positions, cell=cell, pbc=True)
    start = time.perf_counter()
    al_super = al_atoms.make_supercell([5, 5, 5])
    al_time = time.perf_counter() - start

    print(f"超胞生成 (5×5×5):")
    print(f"  ASE:     {ase_time*1000:.2f} ms ({len(ase_super)} atoms)")
    print(f"  aselite: {al_time*1000:.2f} ms ({len(al_super)} atoms)")
    print(f"  加速比:  {ase_time/al_time:.1f}×")
    print()


if __name__ == '__main__':
    print("=" * 60)
    print("几何操作性能对比")
    print("=" * 60)
    print()

    benchmark_angles()
    benchmark_dihedrals()
    benchmark_supercell()

    print("结论:")
    print("  批量向量化操作（get_angles/get_dihedrals）显著快于循环")
    print("  超胞生成使用 meshgrid 广播，与 ASE 相当或更快")
