"""几何操作示例"""

import numpy as np
from aselite import Atoms

# ========== 键角计算 ==========
print("=" * 60)
print("键角计算")
print("=" * 60)

# 创建水分子
water = Atoms(['H', 'O', 'H'],
              positions=[
                  [0.757, 0.586, 0.0],
                  [0.0, 0.0, 0.0],
                  [-0.757, 0.586, 0.0]
              ])

# 单个键角
angle = water.get_angle(0, 1, 2)
print(f"H-O-H 键角: {np.degrees(angle):.1f}°")

# 批量计算（向量化，比循环快 10-20×）
atoms = Atoms(['C'] * 10, positions=np.random.rand(10, 3) * 5)
angle_triplets = [[0, 1, 2], [2, 3, 4], [4, 5, 6]]
angles = atoms.get_angles(angle_triplets)
print(f"批量键角: {np.degrees(angles)}")
print()

# ========== 二面角计算 ==========
print("=" * 60)
print("二面角计算")
print("=" * 60)

# Z 字形结构
chain = Atoms(['C', 'C', 'C', 'C'],
              positions=[
                  [0, 1, 0],
                  [0, 0, 0],
                  [1, 0, 0],
                  [1, 0, 1]
              ])

dihedral = chain.get_dihedral(0, 1, 2, 3)
print(f"二面角: {np.degrees(dihedral):.1f}°")

# 批量计算（向量化，比循环快 100×）
atoms = Atoms(['C'] * 12, positions=np.random.rand(12, 3) * 5)
dihedral_quadruplets = [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]]
dihedrals = atoms.get_dihedrals(dihedral_quadruplets)
print(f"批量二面角: {np.degrees(dihedrals)}")
print()

# ========== 设置角度（逆运动学）==========
print("=" * 60)
print("设置角度")
print("=" * 60)

# 设置键角
atoms = Atoms(['H', 'O', 'H'],
              positions=[[1, 0, 0], [0, 0, 0], [0, 1, 0]])

print(f"初始键角: {np.degrees(atoms.get_angle(0, 1, 2)):.1f}°")
atoms.set_angle(0, 1, 2, np.radians(120))
print(f"设置后:   {np.degrees(atoms.get_angle(0, 1, 2)):.1f}°")
print()

# ========== 超胞生成 ==========
print("=" * 60)
print("超胞生成")
print("=" * 60)

# FCC 铜晶胞
fcc = Atoms(['Cu'] * 4,
            positions=[
                [0, 0, 0],
                [0, 0.5, 0.5],
                [0.5, 0, 0.5],
                [0.5, 0.5, 0]
            ],
            cell=[[3.61, 0, 0], [0, 3.61, 0], [0, 0, 3.61]],
            pbc=True)

supercell = fcc.make_supercell([2, 2, 2])
print(f"原晶胞: {len(fcc)} 原子")
print(f"超胞:   {len(supercell)} 原子")
print(f"晶胞尺寸: {supercell._cell.array[0, 0]:.2f} Å")
print()

# ========== 元素替换 ==========
print("=" * 60)
print("元素替换")
print("=" * 60)

# 创建碳链
carbon_chain = Atoms(['C'] * 10, positions=np.arange(10).reshape(-1, 1) * [1, 0, 0])

# 替换部分原子为氮
carbon_chain.substitute([2, 5, 8], 'N')
print(f"替换后: {list(carbon_chain._symbols)}")

# 随机替换
atoms = Atoms(['C'] * 20, positions=np.random.rand(20, 3) * 10)
atoms.substitute_random('C', 'N', fraction=0.3, seed=42)
n_nitrogen = sum(1 for s in atoms._symbols if s == 'N')
print(f"随机替换 30%: {n_nitrogen} 个 N 原子")
print()

# ========== 吸附物 ==========
print("=" * 60)
print("表面吸附")
print("=" * 60)

# 创建金属表面
surface = Atoms(['Pt'] * 9,
               positions=[
                   [i, j, 0] for i in range(3) for j in range(3)
               ])

# 添加 CO 分子
co = Atoms(['C', 'O'],
          positions=[[0, 0, 0], [0, 0, 1.15]])

surface.add_adsorbate(co, position=[1, 1, 0], height=2.0)
print(f"吸附前: {len(surface) - len(co)} 个表面原子")
print(f"吸附后: {len(surface)} 个原子（表面 + 吸附物）")
print(f"CO 位置: {surface._positions[-2:, 2]}")  # z 坐标
print()

print("=" * 60)
print("性能提示")
print("=" * 60)
print("- 批量操作（get_angles/get_dihedrals）使用向量化实现")
print("- 比逐个调用 get_angle/get_dihedral 快 10-100×")
print("- 适用于分子动力学、构象搜索等需要大量几何计算的场景")
