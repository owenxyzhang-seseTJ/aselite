"""VASP 计算器"""

import os
import subprocess
from pathlib import Path
from .base import Calculator
from ..atoms import Atoms


class VASP(Calculator):
    """VASP 计算器

    简化的 VASP 接口,只关注核心功能
    """

    def __init__(
        self,
        xc: str = 'PBE',
        encut: float = 400,
        kpts: tuple = (1, 1, 1),
        ismear: int = 0,
        sigma: float = 0.1,
        directory: str = '.',
        command: str = None,
        **kwargs
    ):
        """初始化 VASP 计算器

        Args:
            xc: 交换关联泛函
            encut: 平面波截断能 (eV)
            kpts: k点网格
            ismear: 展宽方法
            sigma: 展宽宽度
            directory: 工作目录
            command: VASP 命令
            **kwargs: 其他 INCAR 参数
        """
        params = {
            'xc': xc,
            'encut': encut,
            'kpts': kpts,
            'ismear': ismear,
            'sigma': sigma,
        }
        params.update(kwargs)
        super().__init__(**params)

        self.directory = Path(directory)
        self.command = command or os.environ.get('VASP_COMMAND', 'vasp_std')

    def calculate(self, atoms: Atoms, properties: list = None):
        """执行 VASP 计算

        Args:
            atoms: Atoms 对象
            properties: 需要计算的性质
        """
        self.directory.mkdir(parents=True, exist_ok=True)

        # 写入 POSCAR
        atoms.write(self.directory / 'POSCAR', format='vasp')

        # 写入 INCAR
        self._write_incar()

        # 写入 KPOINTS
        self._write_kpoints()

        # 如果需要真实计算
        if properties:
            self._run_vasp()
            self._read_results()

    def _write_incar(self):
        """生成 INCAR"""
        with open(self.directory / 'INCAR', 'w') as f:
            f.write("SYSTEM = ASE-lite calculation\n")
            f.write(f"ENCUT = {self.parameters['encut']}\n")
            f.write(f"ISMEAR = {self.parameters['ismear']}\n")
            f.write(f"SIGMA = {self.parameters['sigma']}\n")

            # 其他参数
            for key, value in self.parameters.items():
                if key not in ['xc', 'encut', 'kpts', 'ismear', 'sigma']:
                    f.write(f"{key.upper()} = {value}\n")

    def _write_kpoints(self):
        """生成 KPOINTS"""
        kpts = self.parameters['kpts']
        with open(self.directory / 'KPOINTS', 'w') as f:
            f.write("Automatic mesh\n")
            f.write("0\n")
            f.write("Gamma\n")
            f.write(f"{kpts[0]} {kpts[1]} {kpts[2]}\n")
            f.write("0 0 0\n")

    def _run_vasp(self):
        """运行 VASP"""
        subprocess.run(
            self.command,
            cwd=self.directory,
            shell=True,
            check=True
        )

    def _read_results(self):
        """读取 OUTCAR"""
        outcar = self.directory / 'OUTCAR'

        if not outcar.exists():
            raise FileNotFoundError("OUTCAR 未找到")

        with open(outcar, 'r') as f:
            lines = f.readlines()

        # 解析能量
        for line in lines:
            if 'energy(sigma->0)' in line:
                self.results['energy'] = float(line.split()[-1])

        # 解析受力 (简化版)
        # ... 实际实现需要更复杂的解析逻辑
