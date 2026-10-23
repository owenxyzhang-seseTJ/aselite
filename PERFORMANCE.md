# 为什么快

本页只讲可复核的原因。每个结论都对应一段可独立运行的剖析代码，数字由 `benchmarks/compare_ase.py` 实测。

## 一、CIF 解析：30–100 倍

### 根因：ASE 的对称性处理是 O(N²)，与文件内容无关

ASE 读 CIF 时，对每个原子位点调用 `Spacegroup.equivalent_sites()`。这个函数是 Python 双层循环：

```python
# ase/spacegroup/spacegroup.py::equivalent_sites
orbits = []
for kind, pos in enumerate(scaled):          # 每个原子
    for i, (kind0, positions0) in enumerate(orbits):   # 跟每个已收集的 orbit 比
        diff = pos - positions0
        diff -= np.rint(diff)
        if np.any(np.all(np.abs(diff) < symprec, axis=1)):
            break
    else:
        orbits.append((kind, find_orbit(pos)))
```

每个新原子都要和已有的全部 orbit 逐一做周期比较，所以总代价是 O(N²)。**关键在于：即使文件是 P1（只有恒等操作），这个循环照样完整跑一遍**——它并非在展开对称性，而是在判断"这个位点是否已经出现过"。

用 P1 空间群直接测 `equivalent_sites`：

| 原子数 | ASE 耗时 | 每原子 |
|--------|----------|--------|
| 50 | 3.7 ms | 74.9 µs |
| 100 | 13.2 ms | 131.9 µs |
| 200 | 51.7 ms | 258.3 µs |
| 400 | 209.2 ms | 523.0 µs |
| 800 | 830.3 ms | 1037.9 µs |
| 1600 | 3295.7 ms | 2059.8 µs |

每原子耗时为上一行的 2 倍，总耗时为 4 倍——标准的 O(N²)。

在真实文件上，这段占 ASE 总耗时的 **95%**（cProfile，371 原子文件，290 ms 中 `equivalent_sites` 占 277 ms）。真正的文本解析（`parse_items` / `parse_loop`）只占 4.5%。

### 本库的做法：一次向量化去重

不逐原子比较，而是先一次性算出所有镜像，再用整数化坐标做一次 `np.unique`：

```python
new_frac = np.vstack([frac @ mat.T + trans for mat, trans in ops])
new_frac -= np.floor(new_frac)
key = np.round(new_frac / 1e-3).astype(np.int64)   # 量化成整数键
_, keep = np.unique(key, axis=0, return_index=True) # 一次排序去重
```

同样数据，本库每原子耗时恒定在 0.2 µs 左右，不随 N 增长：

| 原子数 | 耗时 | 每原子 |
|--------|------|--------|
| 50 | 0.98 ms | 19.6 µs |
| 200 | 0.06 ms | 0.30 µs |
| 800 | 0.17 ms | 0.21 µs |
| 1600 | 0.32 ms | 0.20 µs |

1600 原子时相差约 10000 倍。这解释了为什么加速比随文件变大而升高：

| 文件 | 原子数 | ASE | aselite | 比值 |
|------|--------|-----|---------|------|
| col_01.cif | 158 | 37.1 ms | 0.94 ms | 39× |
| DFT_dehydrated_ISIF3.cif | 160 | 36.9 ms | 0.84 ms | 44× |
| hcpdata_C3H8_site01_S3.cif | 371 | 186.1 ms | 1.89 ms | 99× |
| site06_match.cif | 403 | 218.0 ms | 2.39 ms | 91× |

### 代价

本库**不推断**文件未写出的对称操作。只给空间群名称（如 `P 1 21 1`）、不给操作列表的文件，本库不会自动补全，ASE 会查表补全。对 P1 展开的文件和显式列出全部操作的文件，两者结果逐原子一致。

如果一个文件只写了空间群名称，ASE 能得到完整晶胞，本库只能得到文件里列出的原子。这是明确的取舍，不是遗漏。

## 二、距离矩阵：2.3 倍

### 根因：ASE 走"上三角索引 + 散射"，本库走完整广播

ASE 的 `get_all_distances()` 内部调用 `get_distances()`，后者用 `np.triu_indices`：

```python
# ase/geometry/geometry.py::get_distances
ind1, ind2 = np.triu_indices(np1, k=1)   # 只算上三角
D = p1[ind2] - p1[ind1]
...
Dout = np.zeros((np1, np1, 3))           # 再散射回 N×N
Dout[(ind1, ind2)] = D
Dout -= np.transpose(Dout, axes=(1, 0, 2))
```

本库直接做完整广播：

```python
vecs = positions[None, :, :] - positions[:, None, :]
return np.linalg.norm(vecs, axis=2)
```

分解 ASE 各步耗时（n=2000）：

| 步骤 | 耗时 |
|------|------|
| 完整 N² 广播（本库做法） | 53.9 ms |
| ASE `get_distances` 合计 | 117.8 ms |
| ├ `triu_indices` + 取差 | 26.3 ms |
| └ 散射回 N×N | 54.4 ms |

`np.triu_indices` 生成的是**索引数组**，对 n=2000 就是约 200 万个整数对；`R[ind2] - R[ind1]` 是花式索引，不连续访存；散射那一步又要分配一个 2000×2000×3 的零数组并做两次全矩阵写入。

本库省掉了这两步：一次连续广播 + 一次 `norm`。代价是内存占用更高（同时存在完整的 N×N×3 中间数组，ASE 只存上三角），而且对称性意味着计算了一半冗余的距离。在 n ≤ 3000 这个量级上，访存效率的收益大于冗余计算的成本。

需要说明：**这是量级相关的取舍**。n 很大时，ASE 只算一半的思路反而可能占优。本库没有在这个区间做优化。

## 三、没有变快的部分

诚实标注，避免误导：

| 操作 | 结论 |
|------|------|
| `import` | 32 ms vs 35 ms，**基本持平**。ASE 的 `import ase` 只加载核心，重量级模块按需导入 |
| `get_center_of_mass` | 同量级（500 原子时本库 3.4 µs，ASE 3.9 µs；2000 原子时 10.8 vs 12.0 µs） |
| `repeat` | 同量级（2000 原子：0.119 ms vs 0.114 ms） |

这三项在优化过程中都改过实现（质心从 `np.average` 换成点积，`repeat` 从 Python 三重循环换成 `meshgrid` + 广播），但结果只是从"明显更慢"追平到"持平"，并不构成优势。

## 四、复现

```bash
python benchmarks/compare_ase.py     # 主要对比
```

单独验证 O(N²) 结论：

```python
import numpy as np, time
from ase.spacegroup import Spacegroup

sg = Spacegroup(1)                    # P1，只有恒等操作
for n in (100, 200, 400, 800):
    pos = np.random.default_rng(0).random((n, 3))
    t = time.perf_counter()
    sg.equivalent_sites(pos, onduplicates='keep')
    print(n, (time.perf_counter() - t) * 1000, 'ms')
# 输出大致为 13 / 52 / 209 / 830 ms —— 四倍增长
```

cProfile 定位：

```python
import cProfile, pstats
from ase.io.cif import read_cif
cProfile.run("read_cif('your_file.cif', index=':')", '/tmp/prof')
pstats.Stats('/tmp/prof').sort_stats('cumulative').print_stats(15)
# 会看到 equivalent_sites 占据绝大部分 cumtime
```
