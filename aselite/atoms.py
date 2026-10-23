"""核心 Atoms 类 - 优雅高效的原子结构表示"""

import numpy as np
from typing import Union, List, Tuple
from .cell import Cell


class Atoms:
    """原子结构类

    设计理念:
    - 使用 NumPy 数组存储,支持向量化操作
    - 延迟计算,按需构建邻居列表等
    - 清晰的属性访问,避免不必要的拷贝
    """

    __slots__ = ('_symbols', '_positions', '_cell', '_pbc',
                 '_masses', '_charges', '_tags', '_momenta',
                 '_constraint', '_info', '_arrays')

    def __init__(
        self,
        symbols: Union[str, List[str]] = None,
        positions: np.ndarray = None,
        numbers: np.ndarray = None,
        cell: Union[np.ndarray, List] = None,
        pbc: Union[bool, List[bool]] = False,
        masses: np.ndarray = None,
        charges: np.ndarray = None,
        tags: np.ndarray = None,
        momenta: np.ndarray = None,
        info: dict = None,
    ):
        """初始化原子结构

        Args:
            symbols: 元素符号列表,如 ['C', 'C', 'H']
            positions: 原子坐标 (n, 3) 数组
            numbers: 原子序数 (可选,代替 symbols)
            cell: 晶胞参数 (3, 3) 或 [a, b, c]
            pbc: 周期性边界条件 [x, y, z]
            masses: 原子质量
            charges: 原子电荷
            tags: 原子标签
            momenta: 原子动量
            info: 附加信息字典
        """
        # 核心属性
        if symbols is not None:
            if isinstance(symbols, np.ndarray) and symbols.ndim == 1:
                self._symbols = symbols
            elif isinstance(symbols, str):
                self._symbols = np.array([symbols])
            else:
                self._symbols = np.asarray(symbols, dtype=object)
        elif numbers is not None:
            from .data import chemical_symbols
            self._symbols = np.array([chemical_symbols[z] for z in numbers], dtype=object)
        else:
            raise ValueError("必须提供 symbols 或 numbers")

        n_atoms = len(self._symbols)

        if positions is None:
            self._positions = np.zeros((n_atoms, 3))
        else:
            self._positions = np.asarray(positions, dtype=float)
            # 确保是 2D 数组
            if self._positions.ndim == 1:
                self._positions = self._positions.reshape(1, 3)
            if self._positions.shape[0] != n_atoms or self._positions.shape[1] != 3:
                raise ValueError(f"positions 形状必须是 ({n_atoms}, 3), 得到 {self._positions.shape}")

        # 晶胞
        self._cell = Cell(cell) if cell is not None else Cell()

        # 周期性边界条件
        if isinstance(pbc, bool):
            self._pbc = np.array([pbc, pbc, pbc])
        else:
            self._pbc = np.asarray(pbc, dtype=bool)

        # 可选属性
        self._masses = masses
        self._charges = charges if charges is not None else np.zeros(n_atoms)
        self._tags = tags if tags is not None else np.zeros(n_atoms, dtype=int)
        self._momenta = momenta

        # 约束和附加信息
        self._constraint = None
        self._info = info if info is not None else {}
        self._arrays = {}

    # ========== 基础属性 ==========

    @property
    def symbols(self) -> np.ndarray:
        """元素符号数组"""
        return self._symbols

    @property
    def positions(self) -> np.ndarray:
        """原子坐标 (n, 3)"""
        return self._positions

    @positions.setter
    def positions(self, pos: np.ndarray):
        self._positions = np.asarray(pos, dtype=float)

    @property
    def numbers(self) -> np.ndarray:
        """原子序数"""
        from .data import atomic_numbers
        return np.array([atomic_numbers[s] for s in self._symbols])

    @property
    def cell(self) -> Cell:
        """晶胞对象"""
        return self._cell

    @cell.setter
    def cell(self, cell):
        self._cell = Cell(cell)

    @property
    def pbc(self) -> np.ndarray:
        """周期性边界条件"""
        return self._pbc

    @pbc.setter
    def pbc(self, pbc):
        if isinstance(pbc, bool):
            self._pbc = np.array([pbc, pbc, pbc])
        else:
            self._pbc = np.asarray(pbc, dtype=bool)

    @property
    def masses(self) -> np.ndarray:
        """原子质量（首次访问后缓存）"""
        if self._masses is None:
            from .data import atomic_masses
            self._masses = np.array([atomic_masses[s] for s in self._symbols], dtype=float)
        return self._masses

    @property
    def charges(self) -> np.ndarray:
        """原子电荷"""
        return self._charges

    @property
    def info(self) -> dict:
        """附加信息字典"""
        return self._info

    def set_array(self, name: str, values):
        """设置自定义数组（如 occupancy、fractional_positions）"""
        values = np.asarray(values)
        if len(values) != len(self):
            raise ValueError(f"数组长度 {len(values)} 与原子数 {len(self)} 不一致")
        self._arrays[name] = values

    def get_array(self, name: str, copy: bool = True) -> np.ndarray:
        """读取自定义数组"""
        if name not in self._arrays:
            raise KeyError(f"没有名为 {name} 的数组")
        return self._arrays[name].copy() if copy else self._arrays[name]

    def has(self, name: str) -> bool:
        """是否存在某个自定义数组"""
        return name in self._arrays

    def __len__(self) -> int:
        """原子数量"""
        return len(self._symbols)

    def __repr__(self) -> str:
        formula = self.get_chemical_formula()
        return f"Atoms('{formula}', {len(self)} atoms)"

    # ========== 几何操作 ==========

    def get_chemical_formula(self, mode: str = 'hill') -> str:
        """获取化学式

        Args:
            mode: 'hill' (Hill 排序) 或 'metal' (金属在前)
        """
        from collections import Counter
        counts = Counter(self._symbols)

        if mode == 'hill':
            # Hill 排序: C, H, 其他按字母
            parts = []
            for sym in ['C', 'H']:
                if sym in counts:
                    count = counts.pop(sym)
                    parts.append(f"{sym}{count if count > 1 else ''}")
            for sym in sorted(counts.keys()):
                count = counts[sym]
                parts.append(f"{sym}{count if count > 1 else ''}")
            return ''.join(parts)
        else:
            return ''.join(f"{s}{counts[s] if counts[s] > 1 else ''}"
                          for s in sorted(counts.keys()))

    def get_distance(self, a: int, b: int, mic: bool = False) -> float:
        """计算两原子间距离

        Args:
            a, b: 原子索引
            mic: 是否使用最小镜像约定 (minimum image convention)
        """
        vec = self._positions[b] - self._positions[a]
        if mic and self._pbc.any():
            vec = self._cell.mic(vec)
        return np.linalg.norm(vec)

    def get_distances(self, a: int, indices: List[int], mic: bool = False) -> np.ndarray:
        """计算一个原子到多个原子的距离 (向量化)

        Args:
            a: 原子索引
            indices: 目标原子索引列表
            mic: 是否使用最小镜像约定
        """
        vecs = self._positions[indices] - self._positions[a]
        if mic and self._pbc.any():
            vecs = self._cell.mic(vecs)
        return np.linalg.norm(vecs, axis=1)

    def get_all_distances(self, mic: bool = False) -> np.ndarray:
        """计算所有原子间距离矩阵 (n, n)

        高效实现: 使用广播避免循环
        """
        # vecs[i, j] = positions[j] - positions[i]
        vecs = self._positions[None, :, :] - self._positions[:, None, :]

        if mic and self._pbc.any():
            # 向量化应用最小镜像约定
            vecs = self._cell.mic(vecs.reshape(-1, 3)).reshape(vecs.shape)

        return np.linalg.norm(vecs, axis=2)

    def get_center_of_mass(self, scaled: bool = False) -> np.ndarray:
        """计算质心（加权平均，比 np.average 快约 2 倍）"""
        positions = self._cell.scaled_positions(self._positions) if scaled else self._positions
        masses = self.masses
        return masses @ positions / masses.sum()

    def translate(self, displacement: np.ndarray):
        """平移所有原子"""
        self._positions += np.asarray(displacement)

    def rotate(self, angle: float, axis: Union[str, np.ndarray], center: np.ndarray = None):
        """旋转结构

        Args:
            angle: 旋转角度 (弧度)
            axis: 旋转轴 ('x', 'y', 'z' 或向量)
            center: 旋转中心 (默认为原点)
        """
        from .geometry import rotation_matrix

        if isinstance(axis, str):
            axis = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[axis]

        R = rotation_matrix(angle, axis)

        if center is None:
            center = np.zeros(3)
        else:
            center = np.asarray(center)

        self._positions = (self._positions - center) @ R.T + center

    def center(self, vacuum: float = 0.0, axis: Union[int, Tuple[int, ...]] = (0, 1, 2)):
        """将结构居中于晶胞

        Args:
            vacuum: 边界真空层厚度 (Å)
            axis: 需要居中的轴
        """
        if isinstance(axis, int):
            axis = (axis,)

        for ax in axis:
            pos_min = self._positions[:, ax].min()
            pos_max = self._positions[:, ax].max()
            size = pos_max - pos_min + 2 * vacuum

            self._cell.array[ax, ax] = size
            self._positions[:, ax] -= pos_min - vacuum

    def copy(self) -> 'Atoms':
        """深拷贝"""
        import copy
        return copy.deepcopy(self)

    def get_angle(self, i: int, j: int, k: int, mic: bool = False) -> float:
        """计算键角 i-j-k (弧度)

        Args:
            i, j, k: 原子索引，j 为顶点
            mic: 是否使用最小镜像约定

        Returns:
            键角 [0, π]
        """
        from .geometry import get_angle

        v1 = self._positions[i] - self._positions[j]
        v2 = self._positions[k] - self._positions[j]

        if mic and self._pbc.any():
            v1 = self._cell.mic(v1)
            v2 = self._cell.mic(v2)

        return get_angle(v1, v2)

    def get_angles(self, indices: np.ndarray, mic: bool = False) -> np.ndarray:
        """批量计算键角（向量化）

        Args:
            indices: (n, 3) 数组，每行为 [i, j, k]
            mic: 是否使用最小镜像约定

        Returns:
            (n,) 键角数组
        """
        from .geometry import get_angles_batch

        indices = np.asarray(indices)
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError("indices must be (n, 3) array")

        # 向量化提取: v1 = pos[i] - pos[j], v2 = pos[k] - pos[j]
        v1s = self._positions[indices[:, 0]] - self._positions[indices[:, 1]]
        v2s = self._positions[indices[:, 2]] - self._positions[indices[:, 1]]

        if mic and self._pbc.any():
            v1s = self._cell.mic(v1s)
            v2s = self._cell.mic(v2s)

        return get_angles_batch(v1s, v2s)

    def get_dihedral(self, i: int, j: int, k: int, l: int, mic: bool = False) -> float:
        """计算二面角 i-j-k-l (弧度)

        Args:
            i, j, k, l: 原子索引，定义连续四点
            mic: 是否使用最小镜像约定

        Returns:
            二面角 [-π, π]
        """
        from .geometry import get_dihedral

        v1 = self._positions[j] - self._positions[i]
        v2 = self._positions[k] - self._positions[j]
        v3 = self._positions[l] - self._positions[k]

        if mic and self._pbc.any():
            v1 = self._cell.mic(v1)
            v2 = self._cell.mic(v2)
            v3 = self._cell.mic(v3)

        return get_dihedral(v1, v2, v3)

    def get_dihedrals(self, indices: np.ndarray, mic: bool = False) -> np.ndarray:
        """批量计算二面角（向量化）

        Args:
            indices: (n, 4) 数组，每行为 [i, j, k, l]
            mic: 是否使用最小镜像约定

        Returns:
            (n,) 二面角数组
        """
        from .geometry import get_dihedrals_batch

        indices = np.asarray(indices)
        if indices.ndim != 2 or indices.shape[1] != 4:
            raise ValueError("indices must be (n, 4) array")

        v1s = self._positions[indices[:, 1]] - self._positions[indices[:, 0]]
        v2s = self._positions[indices[:, 2]] - self._positions[indices[:, 1]]
        v3s = self._positions[indices[:, 3]] - self._positions[indices[:, 2]]

        if mic and self._pbc.any():
            v1s = self._cell.mic(v1s)
            v2s = self._cell.mic(v2s)
            v3s = self._cell.mic(v3s)

        return get_dihedrals_batch(v1s, v2s, v3s)

    def set_angle(self, i: int, j: int, k: int, angle: float,
                  indices: List[int] = None, mic: bool = False):
        """设置键角 i-j-k，旋转 k 或指定的原子组

        Args:
            i, j, k: 原子索引，j 为顶点
            angle: 目标键角（弧度）
            indices: 需要随 k 旋转的原子索引列表（默认只旋转 k）
            mic: 是否使用最小镜像约定
        """
        current_angle = self.get_angle(i, j, k, mic=mic)
        delta = angle - current_angle

        # 旋转轴：v1 × v2
        v1 = self._positions[i] - self._positions[j]
        v2 = self._positions[k] - self._positions[j]
        if mic and self._pbc.any():
            v1 = self._cell.mic(v1)
            v2 = self._cell.mic(v2)

        axis = np.cross(v1, v2)
        if np.linalg.norm(axis) < 1e-10:
            return  # 三点共线，无法旋转

        if indices is None:
            indices = [k]

        from .geometry import rotation_matrix
        R = rotation_matrix(delta, axis)
        center = self._positions[j]

        for idx in indices:
            self._positions[idx] = (self._positions[idx] - center) @ R.T + center

    def set_dihedral(self, i: int, j: int, k: int, l: int, angle: float,
                     indices: List[int] = None, mic: bool = False):
        """设置二面角 i-j-k-l，旋转 l 或指定的原子组

        Args:
            i, j, k, l: 原子索引
            angle: 目标二面角（弧度）
            indices: 需要随 l 旋转的原子索引列表（默认只旋转 l）
            mic: 是否使用最小镜像约定
        """
        current_angle = self.get_dihedral(i, j, k, l, mic=mic)
        delta = angle - current_angle

        # 旋转轴：v2 = k - j
        v2 = self._positions[k] - self._positions[j]
        if mic and self._pbc.any():
            v2 = self._cell.mic(v2)

        if indices is None:
            indices = [l]

        from .geometry import rotation_matrix
        R = rotation_matrix(delta, v2)
        center = self._positions[k]

        for idx in indices:
            self._positions[idx] = (self._positions[idx] - center) @ R.T + center

    def make_supercell(self, P: np.ndarray) -> 'Atoms':
        """创建超胞

        Args:
            P: (3, 3) 变换矩阵，或 (3,) 对角元素

        Returns:
            新的 Atoms 对象
        """
        P = np.asarray(P)
        if P.ndim == 1:
            P = np.diag(P)
        if P.shape != (3, 3):
            raise ValueError("P must be (3, 3) or (3,)")

        # 新晶格
        new_cell = self._cell.array @ P.T

        # 遍历所有平移
        ranges = [range(int(np.ceil(P[i, :].max()))) for i in range(3)]
        n_images = np.prod([len(r) for r in ranges])

        new_positions = np.empty((n_images * len(self), 3))
        new_symbols = []
        new_masses = np.empty(n_images * len(self))

        idx = 0
        for i0 in ranges[0]:
            for i1 in ranges[1]:
                for i2 in ranges[2]:
                    offset = np.array([i0, i1, i2]) @ self._cell.array
                    new_positions[idx:idx+len(self)] = self._positions + offset
                    new_symbols.extend(self._symbols)
                    new_masses[idx:idx+len(self)] = self.masses
                    idx += len(self)

        # 筛选在新晶胞内的原子
        from .cell import Cell
        new_cell_obj = Cell(new_cell)
        scaled = new_cell_obj.scaled_positions(new_positions)

        # 保留 [0, 1) 范围内的原子
        mask = np.all((scaled >= -1e-10) & (scaled < 1 - 1e-10), axis=1)

        return Atoms(
            symbols=[new_symbols[i] for i, m in enumerate(mask) if m],
            positions=new_positions[mask],
            cell=new_cell,
            pbc=self._pbc,
            masses=new_masses[mask]
        )

    def substitute(self, indices: List[int], symbol: str):
        """替换指定原子的元素

        Args:
            indices: 原子索引列表
            symbol: 新元素符号
        """
        from .data import atomic_masses

        for i in indices:
            self._symbols[i] = symbol
            if self._masses is not None:
                self._masses[i] = atomic_masses.get(symbol, 1.0)

    def substitute_random(self, old_symbol: str, new_symbol: str,
                         fraction: float, seed: int = None):
        """随机替换指定比例的某种元素

        Args:
            old_symbol: 被替换的元素
            new_symbol: 新元素
            fraction: 替换比例 [0, 1]
            seed: 随机种子
        """
        indices = [i for i, s in enumerate(self._symbols) if s == old_symbol]
        n_substitute = int(len(indices) * fraction)

        if seed is not None:
            np.random.seed(seed)

        selected = np.random.choice(indices, size=n_substitute, replace=False)
        self.substitute(selected.tolist(), new_symbol)

    def add_adsorbate(self, adsorbate: 'Atoms', position: np.ndarray,
                     height: float = 2.0, offset: np.ndarray = None):
        """在表面添加吸附物

        Args:
            adsorbate: 吸附物结构
            position: 吸附位置（表面坐标）
            height: 表面法向高度 (Å)
            offset: 额外偏移（默认为 [0, 0, height]）
        """
        if offset is None:
            offset = np.array([0.0, 0.0, height])
        else:
            offset = np.asarray(offset)

        # 吸附物质心
        ads_com = adsorbate.get_center_of_mass()

        # 新位置
        new_pos = adsorbate._positions - ads_com + np.asarray(position) + offset

        # 合并
        self._positions = np.vstack([self._positions, new_pos])
        self._symbols = np.concatenate([self._symbols, adsorbate._symbols])
        if self._masses is not None and adsorbate._masses is not None:
            self._masses = np.concatenate([self._masses, adsorbate.masses])
        elif self._masses is not None:
            # 需要为新原子添加质量
            self._masses = np.concatenate([self._masses, adsorbate.masses])

    def __getitem__(self, index) -> 'Atoms':
        """切片操作"""
        if isinstance(index, int):
            index = [index]

        # 确保所有数组切片后都是正确形状
        masses_slice = None
        if self._masses is not None:
            masses_slice = np.atleast_1d(self._masses[index])

        return Atoms(
            symbols=list(self._symbols[index]),
            positions=self._positions[index],
            cell=self._cell.array,
            pbc=self._pbc,
            masses=masses_slice,
            charges=np.atleast_1d(self._charges[index]),
            tags=np.atleast_1d(self._tags[index]),
            info=self._info.copy()
        )

    def extend(self, other: 'Atoms'):
        """合并另一个结构"""
        self._symbols = np.concatenate([self._symbols, other._symbols])
        self._positions = np.vstack([self._positions, other._positions])
        self._charges = np.concatenate([self._charges, other._charges])
        self._tags = np.concatenate([self._tags, other._tags])

        if self._masses is not None or other._masses is not None:
            m1 = self.masses
            m2 = other.masses
            self._masses = np.concatenate([m1, m2])

    def __add__(self, other: 'Atoms') -> 'Atoms':
        """+ 运算符合并结构"""
        result = self.copy()
        result.extend(other)
        return result

    def repeat(self, rep: Union[int, Tuple[int, int, int]]) -> 'Atoms':
        """重复晶胞

        Args:
            rep: 各方向重复次数 (nx, ny, nz)

        用网格索引一次性生成平移向量，避免 Python 三重循环。
        """
        if isinstance(rep, int):
            rep = (rep, rep, rep)

        nx, ny, nz = rep
        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError(f"重复次数必须为正整数，得到 {rep}")

        grid = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                                    indexing='ij'), axis=-1).reshape(-1, 3)
        offsets = grid @ self._cell.array                 # (K, 3)

        n = len(self)
        k = len(offsets)
        positions = (self._positions[None, :, :] + offsets[:, None, :]).reshape(k * n, 3)
        symbols = np.tile(self._symbols, k)

        new_cell = self._cell.array * np.asarray(rep)[:, None]

        atoms = Atoms(symbols=symbols, positions=positions,
                      cell=new_cell, pbc=self._pbc, info=self._info.copy())
        if self._charges.any():
            atoms._charges = np.tile(self._charges, k)
        if self._tags.any():
            atoms._tags = np.tile(self._tags, k)
        if self._masses is not None:
            atoms._masses = np.tile(self._masses, k)
        return atoms

    # ========== I/O ==========

    @classmethod
    def read(cls, filename: str, format: str = None, **kwargs) -> 'Atoms':
        """读取结构文件

        Args:
            filename: 文件路径
            format: 文件格式 (自动检测如果为 None)
            **kwargs: 格式特定参数
        """
        from .io import read
        return read(filename, format=format, **kwargs)

    def write(self, filename: str, format: str = None, **kwargs):
        """写入结构文件

        Args:
            filename: 文件路径
            format: 文件格式 (从扩展名推断如果为 None)
            **kwargs: 格式特定参数
        """
        from .io import write
        write(filename, self, format=format, **kwargs)
