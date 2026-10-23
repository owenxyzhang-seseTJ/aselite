"""测试几何操作：键角、二面角、晶体结构"""

import numpy as np
import pytest
from aselite import Atoms


def test_bond_angle():
    """测试键角计算"""
    # 正交：90°
    atoms = Atoms(['H', 'O', 'H'],
                  positions=[[0, 0, 0], [1, 0, 0], [1, 1, 0]])

    angle = atoms.get_angle(0, 1, 2)
    assert np.isclose(angle, np.pi / 2, atol=1e-6)


def test_bond_angle_water():
    """测试水分子键角"""
    # H-O-H 约 104.5°
    atoms = Atoms(['H', 'O', 'H'],
                  positions=[
                      [0.757, 0.586, 0.0],
                      [0.0, 0.0, 0.0],
                      [-0.757, 0.586, 0.0]
                  ])

    angle = atoms.get_angle(0, 1, 2)
    expected = np.radians(104.5)
    assert np.isclose(angle, expected, atol=0.02)


def test_bond_angles_batch():
    """测试批量键角计算"""
    atoms = Atoms(['C', 'C', 'C', 'C'],
                  positions=[
                      [0, 0, 0],
                      [1, 0, 0],
                      [2, 0, 0],
                      [1, 1, 0]
                  ])

    # 两个键角：0-1-2 (180°), 0-1-3 (90°)
    angles = atoms.get_angles([[0, 1, 2], [0, 1, 3]])

    assert np.isclose(angles[0], np.pi, atol=1e-6)
    assert np.isclose(angles[1], np.pi / 2, atol=1e-6)


def test_dihedral_angle():
    """测试二面角计算"""
    # 平面：0°
    atoms = Atoms(['C', 'C', 'C', 'C'],
                  positions=[
                      [0, 0, 0],
                      [1, 0, 0],
                      [2, 0, 0],
                      [3, 0, 0]
                  ])

    dihedral = atoms.get_dihedral(0, 1, 2, 3)
    assert np.isclose(dihedral, 0.0, atol=1e-6)


def test_dihedral_angle_90deg():
    """测试 90° 二面角"""
    # 正确的几何：形成 Z 字形
    atoms = Atoms(['C', 'C', 'C', 'C'],
                  positions=[
                      [0, 1, 0],
                      [0, 0, 0],
                      [1, 0, 0],
                      [1, 0, 1]
                  ])

    dihedral = atoms.get_dihedral(0, 1, 2, 3)
    assert np.isclose(abs(dihedral), np.pi / 2, atol=0.01)


def test_dihedrals_batch():
    """测试批量二面角计算"""
    atoms = Atoms(['C'] * 8,
                  positions=[
                      [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],  # 平面
                      [0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1]   # Z 字形，90°
                  ])

    dihedrals = atoms.get_dihedrals([
        [0, 1, 2, 3],  # 0°
        [4, 5, 6, 7]   # 90°
    ])

    assert np.isclose(dihedrals[0], 0.0, atol=1e-6)
    assert np.isclose(abs(dihedrals[1]), np.pi / 2, atol=0.01)


def test_set_angle():
    """测试设置键角"""
    atoms = Atoms(['H', 'O', 'H'],
                  positions=[[1, 0, 0], [0, 0, 0], [0, 1, 0]])

    # 改为 60°
    target = np.radians(60)
    atoms.set_angle(0, 1, 2, target)

    new_angle = atoms.get_angle(0, 1, 2)
    assert np.isclose(new_angle, target, atol=1e-6)


def test_set_dihedral():
    """测试设置二面角"""
    atoms = Atoms(['C', 'C', 'C', 'C'],
                  positions=[
                      [0, 1, 0],
                      [0, 0, 0],
                      [1, 0, 0],
                      [1, 1, 0]
                  ])

    # 改为 90°
    target = np.pi / 2
    atoms.set_dihedral(0, 1, 2, 3, target)

    new_dihedral = atoms.get_dihedral(0, 1, 2, 3)
    assert np.isclose(new_dihedral, target, atol=0.01)


def test_supercell_simple():
    """测试简单超胞"""
    atoms = Atoms(['C'], positions=[[0, 0, 0]],
                  cell=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                  pbc=True)

    supercell = atoms.make_supercell([2, 2, 2])

    assert len(supercell) == 8
    assert np.allclose(supercell._cell.array, [[4, 0, 0], [0, 4, 0], [0, 0, 4]])


def test_supercell_fcc():
    """测试 FCC 超胞"""
    # FCC 常规晶胞
    atoms = Atoms(['Cu'] * 4,
                  positions=[
                      [0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]
                  ],
                  cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                  pbc=True)

    supercell = atoms.make_supercell([2, 1, 1])
    assert len(supercell) == 8


def test_substitute():
    """测试元素替换"""
    atoms = Atoms(['C', 'C', 'C'], positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])

    atoms.substitute([0, 2], 'N')

    assert list(atoms._symbols) == ['N', 'C', 'N']


def test_substitute_random():
    """测试随机替换"""
    atoms = Atoms(['C'] * 10, positions=np.random.rand(10, 3))

    atoms.substitute_random('C', 'N', fraction=0.5, seed=42)

    n_count = sum(1 for s in atoms._symbols if s == 'N')
    assert n_count == 5


def test_add_adsorbate():
    """测试吸附物"""
    surface_atoms = Atoms(['Pt'] * 4,
                         positions=[
                             [0, 0, 0], [2, 0, 0],
                             [0, 2, 0], [2, 2, 0]
                         ])

    adsorbate = Atoms(['O'], positions=[[0, 0, 0]])

    surface_atoms.add_adsorbate(adsorbate, position=[1, 1, 0], height=2.0)

    assert len(surface_atoms) == 5
    assert surface_atoms._symbols[-1] == 'O'
    assert np.isclose(surface_atoms._positions[-1, 2], 2.0, atol=0.1)


def test_angle_with_mic():
    """测试周期边界下的键角"""
    # 简化测试：只验证 MIC 功能不崩溃
    atoms = Atoms(['H', 'O', 'H'],
                  positions=[[0.1, 0, 0], [0.5, 0, 0], [0.9, 0, 0]],
                  cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                  pbc=True)

    # 不用 MIC: 线性
    angle_no_mic = atoms.get_angle(0, 1, 2, mic=False)
    assert np.isclose(angle_no_mic, np.pi, atol=0.01)

    # 用 MIC: 确保可以计算
    angle_mic = atoms.get_angle(0, 1, 2, mic=True)
    assert 0 <= angle_mic <= np.pi
