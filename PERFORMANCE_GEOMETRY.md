
## 三、几何操作：批量向量化

### 键角计算：16× 加速

ASE 的 `get_angle(i, j, k)` 每次处理一个角度。批量计算需要 Python 循环：

```python
# ASE 方式
angles = []
for i, j, k in angle_list:
    angles.append(atoms.get_angle(i, j, k))
```

aselite 提供批量接口，内部用 NumPy 广播：

```python
# aselite/atoms.py
def get_angles(self, indices):  # indices: (n, 3)
    v1s = self._positions[indices[:, 0]] - self._positions[indices[:, 1]]
    v2s = self._positions[indices[:, 2]] - self._positions[indices[:, 1]]
    dots = np.sum(v1s * v2s, axis=1)
    norms = np.linalg.norm(v1s, axis=1) * np.linalg.norm(v2s, axis=1)
    return np.arccos(np.clip(dots / norms, -1, 1))
```

实测（1000 个键角）：

```
ASE (循环):     10.94 ms
aselite (向量): 0.66 ms
加速比:         16.6×
```

### 二面角计算：132× 加速

二面角涉及两次叉积，循环开销更大：

```python
# aselite/atoms.py
def get_dihedrals(self, indices):  # indices: (n, 4)
    v1s = positions[indices[:, 1]] - positions[indices[:, 0]]
    v2s = positions[indices[:, 2]] - positions[indices[:, 1]]
    v3s = positions[indices[:, 3]] - positions[indices[:, 2]]
    
    n1s = np.cross(v1s, v2s)  # 第一个平面法向
    n2s = np.cross(v2s, v3s)  # 第二个平面法向
    
    # 批量计算夹角与符号
    ...
```

实测（1000 个二面角）：

```
ASE (循环):     28.59 ms
aselite (向量): 0.22 ms
加速比:         132.2×
```

### 适用场景

- 分子动力学后处理：需计算成千上万个键角/二面角
- 构象分析：扫描所有可能的扭转角
- 结构验证：批量检查几何约束

---

## 复现方法

```bash
cd benchmarks
python compare_ase.py          # CIF、距离矩阵
python benchmark_geometry.py   # 键角、二面角
```

查看完整示例：

```bash
python examples/geometry_operations.py
```
