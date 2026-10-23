# ASE-lite

一个轻量的原子模拟环境库：只依赖 NumPy，约 2400 行，面向「读结构、算几何、跑脚本」这类日常任务。

不是 ASE 的功能替代品，而是在这些场景下更快、更省心的一个选择。

## 实测优势

| 场景 | 结果 |
|------|------|
| CIF 解析 | 比 ASE 快 **30–100 倍**（差异随原子数增长） |
| 全距离矩阵 | 比 ASE 快 **2.3 倍** |
| 键角计算（批量） | 比 ASE 快 **16 倍** |
| 二面角计算（批量） | 比 ASE 快 **130 倍** |
| 质心 / repeat | 与 ASE 同量级 |
| 导入时间 | 约 32 ms，与 ASE 基本持平 |

加速来自具体原因，不是"写得更好"这类空话：

- **CIF**：ASE 的 `equivalent_sites()` 是 O(N²) 的 Python 双层循环，即使 P1 文件也逐原子比对，占其总耗时 95%；本库用一次向量化 `np.unique` 去重，O(N log N)。
- **距离矩阵**：ASE 先算上三角再散射回 N×N，多出索引数组与两次全矩阵写入；本库一次完整广播。
- **键角/二面角批量计算**：ASE 逐个循环调用，本库用 NumPy 广播一次处理所有角度。

剖析过程与可复现代码见 [PERFORMANCE.md](PERFORMANCE.md)。

## 更新日志

**0.2.0** — 几何与晶体结构操作

- 键角/二面角：`get_angle(s)`、`get_dihedral(s)`，批量版本向量化，比 ASE 快 16× / 132×
- 逆运动学：`set_angle()`、`set_dihedral()`
- 晶体操作：`make_supercell()`、`substitute()`、`substitute_random()`、`add_adsorbate()`
- 表面生成：`surface(atoms, (h, k, l), layers, vacuum)`

**0.1.0** — 首个版本：容错 CIF 解析、向量化距离计算、VASP/XYZ/PDB 读写

## CIF 容错能力

真实世界的 CIF 经常不规范。本库的解析器按 CIF 1.1 词法规范实现，并刻意容忍下列缺陷（均有测试覆盖，见 `tests/test_cif.py`）：

- 标准不确定度写法 `0.36838(15)`、分数 `1/3`、`1.23(4)e-2`
- 单/双引号字符串、`;` 起始的多行文本块（含带缩进的闭合 `;`）、行内与整行注释
- CRLF 换行、UTF-8 BOM、latin-1 等非 UTF-8 编码
- 制表符与混合空白、缺失或多余的列、被折成两行的原子记录
- 缺失 `data_` 头、多个 data block（跳过无效块继续找）、标签大小写混用
- 列顺序任意、缺少 `_atom_site_type_symbol`（从 label 推断）、`?` 与 `.` 占位符
- 笛卡尔坐标（`_atom_site_Cartn_*`）、氘/氚（`D`/`T` → `H`）
- 显式对称操作（`x,y,z` / `-x,y+1/2,-z` / `1/3+x,2/3+y,z`）展开与周期性去重

**不推断**文件未写出的对称操作。只给空间群名称、不给操作列表的文件，本库不会自动补全。

## 安装

```bash
pip install aselite
```

从源码安装：

```bash
git clone https://github.com/owenxyzhang-seseTJ/aselite.git
cd aselite && pip install -e .
```

```python
from aselite import Atoms

atoms = Atoms.read('structure.cif')
print(atoms, atoms.cell)

atoms = atoms.repeat((2, 2, 2))
distances = atoms.get_all_distances(mic=True)
atoms.write('output.vasp')
```

## 核心 API

```python
from aselite import Atoms, Cell

atoms = Atoms(
    symbols=['C', 'O', 'O'],
    positions=[[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]],
    cell=[10, 10, 10],
    pbc=True,
)

atoms.symbols          # 元素符号
atoms.positions        # 笛卡尔坐标
atoms.numbers          # 原子序数
atoms.masses           # 质量（惰性计算）
atoms.cell.lengths()   # (a, b, c)
atoms.cell.angles()    # (α, β, γ)
atoms.cell.volume()    # 体积

atoms.get_distance(0, 1, mic=True)
atoms.get_distances(0, [1, 2], mic=True)
atoms.get_all_distances(mic=True)
atoms.get_chemical_formula()
atoms.get_center_of_mass()

# 几何操作
atoms.get_angle(0, 1, 2)                      # 键角
atoms.get_angles([[0,1,2], [2,3,4]])          # 批量键角（向量化）
atoms.get_dihedral(0, 1, 2, 3)                # 二面角
atoms.get_dihedrals([[0,1,2,3], [3,4,5,6]])   # 批量二面角
atoms.set_angle(0, 1, 2, np.radians(120))     # 设置键角
atoms.set_dihedral(0, 1, 2, 3, np.pi/2)       # 设置二面角

# 晶体结构操作
atoms.make_supercell([2, 2, 2])               # 超胞
atoms.substitute([0, 2], 'N')                 # 元素替换
atoms.substitute_random('C', 'N', 0.1)        # 随机替换
atoms.add_adsorbate(co_molecule, [1,1,0], height=2.0)  # 吸附物

atoms.translate([1, 1, 1])
atoms.rotate(1.57, 'z')
atoms.center(vacuum=5.0)
atoms.repeat((2, 2, 2))
atoms[1:3]             # 切片
atoms + other          # 合并
```

读写格式：CIF、VASP（POSCAR/CONTCAR）、XYZ、PDB。

更多示例见 [`examples/`](examples/)。

## 测试

```bash
python tests/test_core.py    # 核心功能
python tests/test_cif.py     # CIF 容错（25 个畸形输入用例）
```

## 项目结构

```
aselite/
├── atoms.py        Atoms 类
├── cell.py         晶胞与最小镜像约定
├── cif.py          CIF 词法/语法分析与对称展开
├── io.py           文件格式读写
├── geometry.py     旋转等几何工具
├── data.py         元素数据
└── calculators/    计算器接口
tests/              测试
examples/           使用示例
benchmarks/         与 ASE 的对比基准
```

## 边界

不打算实现：完整的计算器生态、结构优化与约束、空间群自动识别、可视化。需要这些请用 ASE。

## 许可证

MIT
