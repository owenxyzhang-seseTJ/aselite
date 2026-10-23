# 性能与实现说明

以下数字全部由 `benchmarks/compare_ase.py` 在本机实测得出（Python 3.13 / numpy 2.5 / ASE 3.29），可自行复现。

## 实测性能

| 操作 | aselite | ASE | 比值 |
|------|---------|-----|------|
| 全距离矩阵 500 原子 | 3.8 ms | 8.6 ms | **2.3× 快** |
| 全距离矩阵 2000 原子 | 54 ms | 123 ms | **2.3× 快** |
| 质心 500 原子 | 5.1 µs | 3.7 µs | 0.7×（略慢） |
| 质心 2000 原子 | 11.5 µs | 12.3 µs | 1.1× |
| repeat 2×2×2，500 原子 | 0.045 ms | 0.051 ms | 1.1× |
| repeat 2×2×2，2000 原子 | 0.119 ms | 0.114 ms | 持平 |
| CIF 解析（371 原子） | 2.5 ms | 179 ms | **73× 快** |
| CIF 解析（160 原子） | 1.0 ms | 35 ms | **34× 快** |
| `import` 时间 | 32 ms | 35 ms | 基本持平 |

结论要说得准确：

- **距离计算与 CIF 解析是真正的优势**。CIF 解析快两个数量级，主要是因为 ASE 会对每个原子位点做完整的空间群对称性推导，而本库只按文件里显式给出的对称操作展开。
- **质心与 repeat 与 ASE 同量级**，个别规模上略慢或略快，不构成优势。
- **导入时间没有数量级差距**。ASE 的 `import ase` 本身也只加载核心部分，重量级模块是按需导入的。

## 为什么 CIF 解析快很多

ASE 的 CIF 读取器会调用 `spglib` 做空间群识别与对称性补全（这也是它在若干文件上直接报错的原因）。本库走的是另一条路径：

1. 单遍词法分析，把整个文件变成 token 流，不依赖逐行 `split()`；
2. 只读取显式写出的 `_space_group_symop_operation_xyz`，用安全的 AST 求值生成旋转矩阵与平移向量；
3. 展开后用整数化分数坐标做一次 `np.unique` 周期性去重。

代价是：本库**不推断**文件未写出的对称操作。对于 P1 展开的文件、以及显式列出全部对称操作的文件，结果与 ASE 逐原子一致（已交叉验证，见下）；对于只给空间群名称而不给操作列表的文件，本库不会自动补全。

## 与 ASE 的交叉验证

对本机 4835 个真实 CIF 抽样 800 个，全部解析成功。对其中 ASE 也能读取的文件，逐原子比对分数坐标：

- 29 个文件与 ASE **完全一致**（最大坐标差 0.000000）
- 5 个文件 ASE 直接报错（`ValueError` / `SpacegroupNotFoundError` / `AssertionError`），本库正常读取
- 1 个展示文件（`Periodic Table.cif`）ASE 只读到 1 个原子，本库读到全部 104 个

## 设计取舍

```python
# 向量化，不做 Python 循环
vecs = positions[None, :, :] - positions[:, None, :]
distances = np.linalg.norm(vecs, axis=2)

# 网格索引一次性生成平移，替代三重循环
grid = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                            indexing='ij'), axis=-1).reshape(-1, 3)
offsets = grid @ cell

# 加权平均用点积，比 np.average 快一倍
return masses @ positions / masses.sum()

# __slots__ 让每个实例省掉 __dict__
__slots__ = ('_symbols', '_positions', '_cell', ...)
```

## 复现基准

```bash
python benchmarks/compare_ase.py
```
