"""测试套件 - 验证核心功能"""

import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aselite import Atoms, Cell


def test_atoms_creation():
    """测试 Atoms 创建"""
    print("测试 Atoms 创建...")

    atoms = Atoms(
        symbols=['C', 'C'],
        positions=[[0, 0, 0], [1.42, 0, 0]],
        cell=[10, 10, 10],
        pbc=True
    )

    assert len(atoms) == 2
    assert atoms.symbols[0] == 'C'
    assert np.allclose(atoms.positions[0], [0, 0, 0])
    print("✓ Atoms 创建成功")


def test_chemical_formula():
    """测试化学式生成"""
    print("\n测试化学式生成...")

    atoms = Atoms(
        symbols=['C', 'H', 'H', 'H', 'H'],
        positions=np.random.rand(5, 3)
    )

    formula = atoms.get_chemical_formula()
    assert formula == 'CH4'
    print(f"✓ 化学式: {formula}")


def test_distances():
    """测试距离计算"""
    print("\n测试距离计算...")

    atoms = Atoms(
        symbols=['H', 'H'],
        positions=[[0, 0, 0], [1, 0, 0]]
    )

    dist = atoms.get_distance(0, 1)
    assert np.isclose(dist, 1.0)

    # 测试向量化距离计算
    dists = atoms.get_distances(0, [1])
    assert np.isclose(dists[0], 1.0)

    print(f"✓ 距离计算: {dist:.3f} Å")


def test_cell():
    """测试晶胞操作"""
    print("\n测试晶胞操作...")

    cell = Cell([5, 5, 5])

    assert np.isclose(cell.volume(), 125.0)

    lengths = cell.lengths()
    assert np.allclose(lengths, [5, 5, 5])

    angles = cell.angles()
    assert np.allclose(angles, [90, 90, 90])

    print(f"✓ 晶胞: {cell}")


def test_mic():
    """测试最小镜像约定"""
    print("\n测试最小镜像约定...")

    cell = Cell([10, 10, 10])

    # 测试向量折叠
    vec = np.array([11.0, 0, 0])
    folded = cell.mic(vec)

    assert np.allclose(folded, [1.0, 0, 0])
    print(f"✓ MIC: {vec} → {folded}")


def test_repeat():
    """测试晶胞重复"""
    print("\n测试晶胞重复...")

    atoms = Atoms(
        symbols=['C'],
        positions=[[0, 0, 0]],
        cell=[5, 5, 5],
        pbc=True
    )

    repeated = atoms.repeat((2, 2, 2))

    assert len(repeated) == 8
    assert np.allclose(repeated.cell.lengths(), [10, 10, 10])
    print(f"✓ 重复: 1 → {len(repeated)} 原子")


def test_io_xyz():
    """测试 XYZ 文件读写"""
    print("\n测试 XYZ I/O...")

    atoms = Atoms(
        symbols=['C', 'O'],
        positions=[[0, 0, 0], [1.2, 0, 0]]
    )

    # 写入
    atoms.write('test.xyz')

    # 读取
    atoms2 = Atoms.read('test.xyz')

    assert len(atoms2) == 2
    assert atoms2.symbols[0] == 'C'
    assert np.allclose(atoms2.positions[0], [0, 0, 0])

    # 清理
    os.remove('test.xyz')

    print("✓ XYZ 读写成功")


def test_slice():
    """测试切片操作"""
    print("\n测试切片...")

    atoms = Atoms(
        symbols=['C', 'H', 'H'],
        positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    )

    sub = atoms[1:3]
    assert len(sub) == 2
    assert sub.symbols[0] == 'H'

    print(f"✓ 切片: {len(atoms)} → {len(sub)} 原子")


def test_merge():
    """测试结构合并"""
    print("\n测试结构合并...")

    atoms1 = Atoms(symbols=['C'], positions=[[0, 0, 0]])
    atoms2 = Atoms(symbols=['H'], positions=[[1, 0, 0]])

    combined = atoms1 + atoms2

    assert len(combined) == 2
    assert combined.symbols[0] == 'C'
    assert combined.symbols[1] == 'H'

    print(f"✓ 合并: {len(atoms1)} + {len(atoms2)} = {len(combined)}")


def benchmark_distance_calculation():
    """性能基准测试"""
    print("\n性能基准测试...")
    import time

    # 创建较大结构
    n = 1000
    atoms = Atoms(
        symbols=['C'] * n,
        positions=np.random.rand(n, 3) * 10
    )

    # 测试距离矩阵计算
    start = time.time()
    dist_matrix = atoms.get_all_distances()
    elapsed = time.time() - start

    print(f"✓ {n} 原子距离矩阵: {elapsed*1000:.1f} ms")
    assert dist_matrix.shape == (n, n)


def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("ASE-lite 测试套件")
    print("=" * 50)

    tests = [
        test_atoms_creation,
        test_chemical_formula,
        test_distances,
        test_cell,
        test_mic,
        test_repeat,
        test_io_xyz,
        test_slice,
        test_merge,
        benchmark_distance_calculation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} 失败: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 50)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
