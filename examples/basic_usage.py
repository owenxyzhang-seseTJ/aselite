"""使用示例 - 展示 ASE-lite 的核心功能"""

import numpy as np
from aselite import Atoms


def example_1_create_molecule():
    """示例 1: 创建分子结构"""
    print("=" * 50)
    print("示例 1: 创建 CO2 分子")
    print("=" * 50)

    co2 = Atoms(
        symbols=['C', 'O', 'O'],
        positions=[
            [0.0, 0.0, 0.0],
            [1.16, 0.0, 0.0],
            [-1.16, 0.0, 0.0]
        ]
    )

    print(f"结构: {co2}")
    print(f"化学式: {co2.get_chemical_formula()}")
    print(f"质心: {co2.get_center_of_mass()}")

    # 计算键长
    d1 = co2.get_distance(0, 1)
    d2 = co2.get_distance(0, 2)
    print(f"C-O 键长: {d1:.3f} Å, {d2:.3f} Å")

    # 保存
    co2.write('co2.xyz')
    print("✓ 已保存为 co2.xyz\n")


def example_2_crystal_structure():
    """示例 2: 创建晶体结构"""
    print("=" * 50)
    print("示例 2: 金刚石晶胞")
    print("=" * 50)

    # 金刚石晶胞 (简化版)
    a = 3.567  # 晶格常数 (Å)

    diamond = Atoms(
        symbols=['C'] * 8,
        positions=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.0],
            [0.75, 0.75, 0.25],
            [0.5, 0.0, 0.5],
            [0.75, 0.25, 0.75],
            [0.0, 0.5, 0.5],
            [0.25, 0.75, 0.75],
        ],
        cell=[a, a, a],
        pbc=True
    )

    # 转换分数坐标到笛卡尔坐标
    diamond.positions = diamond.cell.cartesian_positions(diamond.positions)

    print(f"结构: {diamond}")
    print(f"晶胞体积: {diamond.cell.volume():.3f} ų")
    print(f"密度: {len(diamond) * 12.01 / diamond.cell.volume() / 0.6022:.2f} g/cm³")

    # 重复晶胞
    supercell = diamond.repeat((2, 2, 2))
    print(f"超胞: {len(supercell)} 原子")

    # 保存
    diamond.write('diamond.vasp')
    print("✓ 已保存为 diamond.vasp\n")


def example_3_geometric_operations():
    """示例 3: 几何操作"""
    print("=" * 50)
    print("示例 3: 几何变换")
    print("=" * 50)

    # 创建水分子
    water = Atoms(
        symbols=['O', 'H', 'H'],
        positions=[
            [0.0, 0.0, 0.0],
            [0.757, 0.586, 0.0],
            [-0.757, 0.586, 0.0]
        ]
    )

    print("原始结构:")
    print(water.positions)

    # 平移
    water_shifted = water.copy()
    water_shifted.translate([5, 5, 5])
    print("\n平移后:")
    print(water_shifted.positions)

    # 旋转 (绕 z 轴 90°)
    water_rotated = water.copy()
    water_rotated.rotate(np.pi/2, 'z')
    print("\n旋转 90° 后:")
    print(water_rotated.positions)

    # 居中于盒子
    water_centered = water.copy()
    water_centered.center(vacuum=5.0)
    print(f"\n居中后晶胞: {water_centered.cell.lengths()}")
    print("✓ 几何操作完成\n")


def example_4_distance_calculations():
    """示例 4: 高效距离计算"""
    print("=" * 50)
    print("示例 4: 距离计算")
    print("=" * 50)

    # 创建随机结构
    n = 100
    atoms = Atoms(
        symbols=['C'] * n,
        positions=np.random.rand(n, 3) * 10
    )

    # 单个距离
    d = atoms.get_distance(0, 1)
    print(f"原子 0-1 距离: {d:.3f} Å")

    # 批量距离 (向量化)
    distances = atoms.get_distances(0, list(range(1, 11)))
    print("原子 0 到其他 10 个原子的距离:")
    print(f"  平均: {distances.mean():.3f} Å")
    print(f"  最小: {distances.min():.3f} Å")

    # 完整距离矩阵
    import time
    start = time.time()
    dist_matrix = atoms.get_all_distances()
    elapsed = time.time() - start

    print(f"\n距离矩阵 ({n}x{n}): {elapsed*1000:.1f} ms")
    print(f"最近邻距离: {dist_matrix[dist_matrix > 0].min():.3f} Å")
    print("✓ 距离计算完成\n")


def example_5_io_formats():
    """示例 5: 文件 I/O"""
    print("=" * 50)
    print("示例 5: 多种文件格式")
    print("=" * 50)

    # 创建测试结构
    atoms = Atoms(
        symbols=['Cu'] * 4,
        positions=[
            [0, 0, 0],
            [1.8, 1.8, 0],
            [1.8, 0, 1.8],
            [0, 1.8, 1.8]
        ],
        cell=[3.6, 3.6, 3.6],
        pbc=True
    )

    # 写入不同格式
    formats = ['xyz', 'vasp', 'cif']

    for fmt in formats:
        filename = f'cu_test.{fmt}'
        atoms.write(filename, format=fmt)
        print(f"✓ 写入 {filename}")

    # 读回验证
    for fmt in formats:
        filename = f'cu_test.{fmt}'
        atoms_read = Atoms.read(filename, format=fmt)
        print(f"✓ 读取 {filename}: {len(atoms_read)} 原子")

    print("✓ I/O 测试完成\n")


def example_6_performance_comparison():
    """示例 6: 性能对比"""
    print("=" * 50)
    print("示例 6: 性能基准")
    print("=" * 50)

    import time

    sizes = [100, 500, 1000]

    print(f"{'大小':<10} {'距离矩阵':<15} {'质心计算':<15}")
    print("-" * 40)

    for n in sizes:
        atoms = Atoms(
            symbols=['C'] * n,
            positions=np.random.rand(n, 3) * 10
        )

        # 距离矩阵
        start = time.time()
        _ = atoms.get_all_distances()
        t_dist = (time.time() - start) * 1000

        # 质心
        start = time.time()
        for _ in range(100):
            _ = atoms.get_center_of_mass()
        t_com = (time.time() - start) * 10  # 平均每次

        print(f"{n:<10} {t_dist:<15.1f} {t_com:<15.3f}")

    print("\n单位: ms")
    print("✓ 性能测试完成\n")


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("ASE-lite 使用示例")
    print("=" * 50 + "\n")

    example_1_create_molecule()
    example_2_crystal_structure()
    example_3_geometric_operations()
    example_4_distance_calculations()
    example_5_io_formats()
    example_6_performance_comparison()

    print("=" * 50)
    print("所有示例完成!")
    print("=" * 50)
