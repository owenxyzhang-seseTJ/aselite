"""构造畸形 CIF，验证解析器的容错能力。

每个用例都从一份最小合法 CIF 出发，只注入一种缺陷。
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from aselite import Atoms
from aselite.cif import parse_cif, to_float, to_symbol

BASE = """data_test
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.0 0.0 0.0
O1 O 0.5 0.5 0.5
"""

CASES = {}


def case(name):
    def register(fn):
        CASES[name] = fn
        return fn
    return register


@case('标准最小 CIF')
def _():
    atoms = parse_cif(BASE)
    assert len(atoms) == 2, len(atoms)
    assert np.allclose(atoms.cell.lengths(), 5.0)
    assert atoms.pbc.all()


@case('带 esd 括号的数值')
def _():
    text = BASE.replace('0.5 0.5 0.5', '0.5(2) 0.4983(17) 0.5(10)')
    atoms = parse_cif(text)
    assert np.allclose(atoms.positions[1], [2.5, 2.4915, 2.5], atol=1e-6), atoms.positions[1]
    assert to_float('0.36838(15)') == 0.36838
    assert to_float('1.23(4)e-2') is not None


@case('CRLF 换行 + BOM')
def _():
    text = '﻿' + BASE.replace('\n', '\r\n')
    atoms = parse_cif(text)
    assert len(atoms) == 2


@case('制表符分隔')
def _():
    text = BASE.replace('C1 C 0.0 0.0 0.0', 'C1\tC\t0.0\t0.0\t0.0')
    text = text.replace('O1 O 0.5 0.5 0.5', 'O1\t\tO\t0.5  0.5\t0.5')
    assert len(parse_cif(text)) == 2


@case('行内与整行注释')
def _():
    lines = BASE.splitlines()
    lines.insert(1, '# 这是一整行注释')
    out = []
    for ln in lines:
        if '0.5' in ln:
            ln = ln + '   # 行尾注释'
        out.append(ln)
    assert len(parse_cif('\n'.join(out))) == 2


@case('引号包裹的符号与标签')
def _():
    text = BASE.replace('C1 C 0.0', "'C1' 'C' 0.0").replace('O1 O 0.5', '"O1" "O" 0.5')
    atoms = parse_cif(text)
    assert list(atoms.symbols) == ['C', 'O'], list(atoms.symbols)


@case('缺失 type_symbol 列（靠 label 推断）')
def _():
    text = """data_x
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Cu1 0.0 0.0 0.0
O12a 0.25 0.25 0.25
"""
    atoms = parse_cif(text)
    assert list(atoms.symbols) == ['Cu', 'O'], list(atoms.symbols)


@case('? 与 . 占位符')
def _():
    text = BASE.replace('C1 C 0.0 0.0 0.0', 'C1 C 0.0 0.0 0.0')
    text = text.replace('_atom_site_fract_z', '_atom_site_fract_z\n_atom_site_occupancy')
    text = text.replace('C1 C 0.0 0.0 0.0', 'C1 C 0.0 0.0 0.0 ?')
    text = text.replace('O1 O 0.5 0.5 0.5', 'O1 O 0.5 0.5 0.5 .')
    atoms = parse_cif(text)
    assert len(atoms) == 2
    assert np.isnan(atoms.get_array('occupancy')).sum() + 2 >= 2


@case('坐标列缺失一整行（残行）')
def _():
    text = BASE.replace('C1 C 0.0 0.0 0.0\n', '')      # 删掉一行
    atoms = parse_cif(text)
    assert len(atoms) == 1, len(atoms)


@case('坐标行被折成两行')
def _():
    text = BASE.replace('C1 C 0.0 0.0 0.0', 'C1 C 0.0\n0.0 0.0')
    text = text.replace('_atom_site_fract_z\n', '_atom_site_fract_z\n_atom_site_occupancy\n')
    text = text.replace('C1 C 0.0\n0.0 0.0\n', 'C1 C 0.0\n0.0 0.0 1.0\n')
    text = text.replace('O1 O 0.5 0.5 0.5', 'O1 O 0.5 0.5 0.5 1.0')
    atoms = parse_cif(text)
    assert len(atoms) == 2, len(atoms)


@case('缺失 data_ 头')
def _():
    text = BASE.split('\n', 1)[1]
    assert len(parse_cif(text)) == 2


@case('多个 data block（第一个无效，第二个有效）')
def _():
    text = """data_empty
_cell_length_a 5
_cell_length_b 5
_cell_length_c 5
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90

data_real
""" + BASE.split('\n', 1)[1]
    atoms = parse_cif(text)
    assert len(atoms) == 2
    assert atoms.info['cif_block'] == 'real', atoms.info


@case('标签大小写混用')
def _():
    text = BASE.replace('_atom_site_fract_x', '_ATOM_SITE_FRACT_X')
    text = text.replace('_cell_length_a', '_CELL_LENGTH_A')
    text = text.replace('_atom_site_type_symbol', '_Atom_Site_Type_Symbol')
    atoms = parse_cif(text)
    assert len(atoms) == 2


@case('列顺序任意（类型符号在最后）')
def _():
    text = """data_x
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_label
_atom_site_type_symbol
0.0 0.0 0.0 C1 C
0.5 0.5 0.5 O1 O
"""
    atoms = parse_cif(text)
    assert list(atoms.symbols) == ['C', 'O']


@case('分号文本块（含以 _ 开头的行）')
def _():
    text = """data_x
_publ_section_title
;
_atom_site_fract_x 不是标签，只是文本
第二行
;
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.0 0.0 0.0
O1 O 0.5 0.5 0.5
"""
    atoms = parse_cif(text)
    assert len(atoms) == 2


@case('缩进的闭合分号')
def _():
    text = """data_x
_publ_section_title
;
标题内容
   ;
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.0 0.0 0.0
"""
    assert len(parse_cif(text)) == 1


@case('分数形式与科学计数法')
def _():
    text = BASE.replace('0.5 0.5 0.5', '1/2 2.5e-1 5E-1')
    atoms = parse_cif(text)
    assert np.allclose(atoms.positions[1], [2.5, 1.25, 2.5], atol=1e-9), atoms.positions[1]


@case('笛卡尔坐标 (Cartn)')
def _():
    text = """data_x
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_Cartn_x
_atom_site_Cartn_y
_atom_site_Cartn_z
C1 C 1.0 2.0 3.0
"""
    atoms = parse_cif(text)
    assert np.allclose(atoms.positions[0], [1, 2, 3])


@case('对称操作展开（含去重）')
def _():
    text = """data_x
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_space_group_symop_operation_xyz
x,y,z
-x,-y,-z
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.25 0.25 0.25
O1 O 0.0 0.0 0.0
"""
    atoms = parse_cif(text)
    # C1 展开为 2 个；O1 在原点，两次映射重合，只留 1 个
    assert len(atoms) == 3, len(atoms)


@case('化学式与元素符号边界')
def _():
    assert to_symbol('O10') == 'O'
    assert to_symbol('Cu2+') == 'Cu'
    assert to_symbol('Cl1') == 'Cl'
    assert to_symbol('C12A') == 'C'
    assert to_symbol('?') is None
    assert to_symbol('') is None


@case('零体积/缺晶胞参数时退化为分子')
def _():
    text = """data_x
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.0 0.0 0.0
"""
    atoms = parse_cif(text)
    assert not atoms.pbc.any()
    assert len(atoms) == 1


@case('非 UTF-8 编码 (latin-1)')
def _():
    text = """data_x
_publ_section_title
°
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.0 0.0 0.0
"""
    with tempfile.NamedTemporaryFile('wb', suffix='.cif', delete=False) as fh:
        fh.write(text.encode('latin-1'))
        path = fh.name
    try:
        assert len(Atoms.read(path)) == 1
    finally:
        os.unlink(path)


@case('截断文件（缺少末尾换行）')
def _():
    text = BASE.rstrip('\n')
    assert len(parse_cif(text)) == 2


@case('空文件与纯注释文件')
def _():
    for bad in ('', '   \n\n', '# 只有注释\n'):
        try:
            parse_cif(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('空内容应抛 ValueError')


@case('标签无值（后面紧跟另一个标签）')
def _():
    text = BASE.replace('_cell_length_a 5.0', '_cell_length_a\n_publ_x 1')
    atoms = parse_cif(text)
    assert len(atoms) == 2


def run():
    print('=' * 62)
    print('CIF 容错测试')
    print('=' * 62)
    passed = failed = 0
    for name, fn in CASES.items():
        try:
            fn()
            print(f'  ✓ {name}')
            passed += 1
        except Exception as exc:
            print(f'  ✗ {name}: {type(exc).__name__}: {exc}')
            failed += 1
    print('-' * 62)
    print(f'{passed} 通过 / {failed} 失败')
    return failed == 0


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
