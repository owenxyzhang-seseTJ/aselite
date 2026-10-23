"""计算器基类 - 统一的计算器接口"""

from abc import ABC, abstractmethod
import numpy as np
from .atoms import Atoms


class Calculator(ABC):
    """计算器抽象基类

    所有计算器(VASP, Gaussian 等)的统一接口
    """

    def __init__(self, **kwargs):
        """初始化计算器

        Args:
            **kwargs: 计算器特定参数
        """
        self.parameters = kwargs
        self.results = {}

    @abstractmethod
    def calculate(self, atoms: Atoms, properties: list = None):
        """执行计算

        Args:
            atoms: Atoms 对象
            properties: 需要计算的性质列表 ['energy', 'forces', 'stress']
        """
        pass

    def get_potential_energy(self, atoms: Atoms) -> float:
        """获取势能"""
        if 'energy' not in self.results:
            self.calculate(atoms, properties=['energy'])
        return self.results['energy']

    def get_forces(self, atoms: Atoms) -> np.ndarray:
        """获取原子受力 (n, 3)"""
        if 'forces' not in self.results:
            self.calculate(atoms, properties=['forces'])
        return self.results['forces']

    def get_stress(self, atoms: Atoms) -> np.ndarray:
        """获取应力张量 (6,) Voigt 记号"""
        if 'stress' not in self.results:
            self.calculate(atoms, properties=['stress'])
        return self.results['stress']

    def __repr__(self) -> str:
        params = ', '.join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.__class__.__name__}({params})"


class SinglePointCalculator(Calculator):
    """单点计算器 - 直接设置结果,不执行真实计算"""

    def __init__(self, energy: float = None, forces: np.ndarray = None,
                 stress: np.ndarray = None):
        super().__init__()
        if energy is not None:
            self.results['energy'] = energy
        if forces is not None:
            self.results['forces'] = forces
        if stress is not None:
            self.results['stress'] = stress

    def calculate(self, atoms: Atoms, properties: list = None):
        """单点计算器不执行计算"""
        pass
