"""与 ASE 的性能和结果对比基准。

需要先安装 ASE：pip install ase
运行：python benchmarks/compare_ase.py
"""

import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from aselite import Atoms
from aselite.cif import read_cif

try:
    from ase import Atoms as AseAtoms
    from ase.io.cif import read_cif as ase_read_cif
except ImportError:
    print('未安装 ASE，跳过对比（pip install ase）')
    sys.exit(0)


def best_of(fn, repeat, number):
    """取 repeat 轮中每轮 number 次调用的最快平均耗时（毫秒）。"""
    best = float('inf')
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(number):
            fn()
        best = min(best, (time.perf_counter() - start) / number * 1000)
    return best


def compare_atoms():
    print(f"{'操作':26s} {'aselite':>11s} {'ASE':>11s} {'加速比':>8s}")
    print('-' * 60)

    for n in (500, 2000):
        rng = np.random.default_rng(0)
        pos = rng.random((n, 3)) * 20
        ours = Atoms(symbols=['C'] * n, positions=pos, cell=[20, 20, 20], pbc=True)
        ref = AseAtoms(symbols=['C'] * n, positions=pos, cell=[20, 20, 20], pbc=True)

        rows = [
            ('全距离矩阵', lambda: ours.get_all_distances(),
             lambda: ref.get_all_distances(), 5),
            ('质心', lambda: ours.get_center_of_mass(),
             lambda: ref.get_center_of_mass(), 300),
            ('repeat 2x2x2', lambda: ours.repeat((2, 2, 2)),
             lambda: ref.repeat((2, 2, 2)), 100),
        ]
        for label, f_ours, f_ref, number in rows:
            t1 = best_of(f_ours, 3, number)
            t2 = best_of(f_ref, 3, number)
            unit = 'us' if t1 < 1 else 'ms'
            scale = 1000 if unit == 'us' else 1
            print(f"n={n} {label:18s} {t1*scale:9.2f}{unit} {t2*scale:9.2f}{unit} {t2/t1:7.2f}x")


def compare_cif(cif_dir):
    print('\nCIF 解析：')
    files = ['hcpdata_C3H8_site01_S3.cif', 'DFT_dehydrated_ISIF3.cif', 'col_01.cif']
    for name in files:
        path = os.path.join(cif_dir, name)
        if not os.path.exists(path):
            continue
        t1 = best_of(lambda: read_cif(path), 3, 10)
        try:
            t2 = best_of(lambda: ase_read_cif(path, index=':'), 3, 10)
            print(f"  {name[:32]:34s} {t1:7.2f}ms vs {t2:7.2f}ms  {t2/t1:6.1f}x")
        except Exception as exc:
            print(f"  {name[:32]:34s} {t1:7.2f}ms vs ASE 读取失败 ({type(exc).__name__})")


def compare_import():
    print('\n导入时间：')
    code = "import time;t=time.perf_counter();import {m};print((time.perf_counter()-t)*1000)"
    for module in ('ase', 'aselite'):
        runs = [float(subprocess.run([sys.executable, '-c', code.format(m=module)],
                                     capture_output=True, text=True).stdout)
                for _ in range(5)]
        print(f'  import {module:9s}: {min(runs):6.1f} ms')


if __name__ == '__main__':
    compare_atoms()
    compare_cif(os.path.expanduser('~'))
    compare_import()
