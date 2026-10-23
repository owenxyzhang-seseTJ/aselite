"""CIF 解析器

按 CIF 1.1 词法规范实现，并刻意容忍真实世界中的不规范文件：

- 多行文本块（`;` 起始）、单/双引号字符串、行内注释
- 标准不确定度写法：0.36838(15)
- 占位符 ? 和 .、缺失列、多余列、制表符与混合空白
- CRLF / BOM / 非 UTF-8 编码、缺失 data_ 头、多个 data block
- 标签大小写不一致、列顺序任意、atom_site_type_symbol 缺失（从 label 推断）
- 分数坐标与笛卡尔坐标（_atom_site_Cartn_x）
- 显式对称操作展开（_space_group_symop_operation_xyz / _symmetry_equiv_pos_as_xyz）
"""

from __future__ import annotations

import ast
import re
from typing import Iterator, Optional

import numpy as np

from .atoms import Atoms

_WS = ' \t\r\v\f\n'
_KEYWORDS = ('loop_', 'stop_')


# --------------------------------------------------------------------------
# 词法分析
# --------------------------------------------------------------------------

def tokenize(text: str) -> Iterator[tuple]:
    """产出 (token, 行号, 是否裸 token)。

    裸 token 才可能是标签或关键字；引号串与文本块永远是值。
    不抛异常：不完整的引号或文本块读到文件尾即止。
    """
    n = len(text)
    i = 0
    line = 1
    line_start = True
    while i < n:
        ch = text[i]

        if ch in ' \t\r\v\f':
            i += 1
            continue
        if ch == '\n':
            line += 1
            line_start = True
            i += 1
            continue
        if ch == '#':                          # 注释到行尾
            j = text.find('\n', i)
            i = n if j < 0 else j
            continue

        if ch == ';' and line_start:           # 文本块：; 位于行首
            start_line = line
            i += 1
            if i < n and text[i] == '\n':
                line += 1
                i += 1
            started = i
            end = n
            while i < n:
                j = text.find('\n', i)
                if j < 0:
                    break
                nxt = j + 1
                probe = nxt                    # 闭合 ; 允许带缩进（真实文件常见）
                while probe < n and text[probe] in ' \t':
                    probe += 1
                if probe < n and text[probe] == ';':
                    end = j
                    i = probe + 1
                    if i < n and text[i] == '\n':
                        i += 1
                        line += 1
                    line += 1
                    break
                line += 1
                i = nxt
            yield text[started:end].replace('\r\n', '\n').rstrip('\r\n'), start_line, False
            line_start = False
            continue

        if ch in '"\'':
            quote = ch
            start = i + 1
            i += 1
            while i < n:
                if text[i] == quote and (i + 1 >= n or text[i + 1] in _WS):
                    break
                if text[i] == '\n':
                    line += 1
                i += 1
            yield text[start:i], line, False
            i += 1
            line_start = False
            continue

        start = i                                  # 裸 token
        while i < n and text[i] not in _WS:
            i += 1
        yield text[start:i], line, True
        line_start = False


def _is_keyword(tok: str, bare: bool = True) -> bool:
    if not bare:
        return False
    low = tok.lower()
    return tok.startswith('_') or low in _KEYWORDS or low.startswith(('data_', 'save_'))


# --------------------------------------------------------------------------
# 语法分析
# --------------------------------------------------------------------------

class _Block:
    __slots__ = ('name', 'tags', 'loops')

    def __init__(self, name: str = ''):
        self.name = name
        self.tags: dict = {}        # 小写标签 -> 值
        self.loops: list = []       # [(labels, values)]

    def tag(self, *names) -> Optional[str]:
        for name in names:
            if name in self.tags:
                return self.tags[name]
        return None

    def loop_with(self, *tag_names):
        """返回包含任一指定标签的 loop，优先匹配标签最多的一个。"""
        best, best_hits = None, 0
        for labels, values in self.loops:
            hits = sum(1 for name in tag_names if name in labels)
            if hits > best_hits:
                best, best_hits = (labels, values), hits
        return best


def parse_blocks(text: str) -> list:
    """把 CIF 文本解析为 data block 列表。"""
    tokens = list(tokenize(text))
    blocks: list = []
    block = _Block()
    i, n = 0, len(tokens)

    while i < n:
        tok, _, bare = tokens[i]
        low = tok.lower()

        if bare and low.startswith('data_'):
            if block.tags or block.loops or block.name:
                blocks.append(block)
            block = _Block(tok[5:])
            i += 1

        elif bare and low == 'loop_':
            i += 1
            labels = []
            while i < n and tokens[i][2] and tokens[i][0].startswith('_'):
                labels.append(tokens[i][0].lower())
                i += 1
            values = []
            while i < n and not _is_keyword(tokens[i][0], tokens[i][2]):
                values.append(tokens[i][0])
                i += 1
            if labels:
                block.loops.append((labels, values))

        elif bare and tok.startswith('_'):
            i += 1
            if i < n and not _is_keyword(tokens[i][0], tokens[i][2]):
                block.tags[low] = tokens[i][0]
                i += 1
            else:
                block.tags[low] = ''

        else:
            i += 1

    if block.tags or block.loops or block.name:
        blocks.append(block)
    return blocks


# --------------------------------------------------------------------------
# 值转换
# --------------------------------------------------------------------------

_MISSING = {'', '?', '.'}
_NUM = re.compile(r'^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$')
_FRACTION = re.compile(r'^([+-]?\d+)\s*/\s*([+-]?\d+)$')


def to_float(value: Optional[str]) -> Optional[float]:
    """把 CIF 数值转为 float。

    支持 0.1234(5)、1/3、科学计数法；? 与 . 返回 None。
    """
    if value is None:
        return None
    s = value.strip()
    if s in _MISSING:
        return None
    s = re.sub(r'\((\d+)\)', '', s).strip()      # 去掉标准不确定度（含 1.23(4)e-2 形式）
    if not s or s in _MISSING:
        return None
    m = _FRACTION.match(s)
    if m:
        den = float(m.group(2))
        return float(m.group(1)) / den if den else None
    if _NUM.match(s):
        try:
            return float(s)
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


_SYMBOL = re.compile(r'([A-Za-z][a-z]?)')

# 同位素写法与常见别名
_ALIASES = {'D': 'H', 'T': 'H', 'DEUTERIUM': 'H', 'TRITIUM': 'H'}


def to_symbol(label: Optional[str]) -> Optional[str]:
    """从原子标签推断元素符号：O10 / Cu2+ / C12A / D → O / Cu / C / H"""
    if not label:
        return None
    s = label.strip().strip('\'"')
    if not s or s in _MISSING:
        return None

    from .data import atomic_numbers

    head = s.split('(')[0]
    if head.upper() in _ALIASES:                 # D / T / Deuterium
        return _ALIASES[head.upper()]

    m = _SYMBOL.match(s)
    if not m:
        return None
    sym = m.group(1)
    sym = sym[0].upper() + sym[1:].lower()       # 规范化大小写：CU → Cu
    if sym in atomic_numbers:
        return sym
    if sym.upper() in _ALIASES:
        return _ALIASES[sym.upper()]
    if sym[0] in atomic_numbers:
        return sym[0]
    return None


# --------------------------------------------------------------------------
# 表格行切分
# --------------------------------------------------------------------------

def _rows(labels: list, values: list) -> list:
    """把一维值序列切成行。

    若每行值数都等于列数（最常见的“一行一个原子”），按行切；
    否则按列数顺序切分，这样被折行书写的原子记录仍能正确还原。
    """
    ncol = len(labels)
    if ncol == 0:
        return []
    if len(values) % ncol == 0:
        return [values[k:k + ncol] for k in range(0, len(values), ncol)]
    # 行数与列数不整齐：退回顺序切分，末尾残行保留（由调用方校验丢弃）
    return [values[k:k + ncol] for k in range(0, len(values), ncol)]


# --------------------------------------------------------------------------
# 对称操作
# --------------------------------------------------------------------------

_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
            ast.Div, ast.USub, ast.UAdd, ast.Constant, ast.Name, ast.Load)


def _eval_component(expr: str, x: float, y: float, z: float):
    """安全求值对称操作的一个分量，如 '-x+1/2'。"""
    try:
        tree = ast.parse(expr.strip().lower(), mode='eval')
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            return None
        if isinstance(node, ast.Name) and node.id not in ('x', 'y', 'z'):
            return None
    env = {'x': x, 'y': y, 'z': z}
    try:
        return float(eval(compile(tree, '<symop>', 'eval'), {'__builtins__': {}}, env))
    except Exception:
        return None


def parse_symop(expr: str):
    """'x,y,z' 或 '-x+1/2,y,-z' → (3x3 矩阵, 平移向量)；失败返回 None。"""
    if not expr:
        return None
    parts = [p for p in expr.strip().strip('\'"').replace(' ', '').split(',')]
    if len(parts) != 3:
        return None
    rot = np.zeros((3, 3))
    trans = np.zeros(3)
    for row, comp in enumerate(parts):
        base = _eval_component(comp, 0.0, 0.0, 0.0)
        if base is None:
            return None
        trans[row] = base
        for col, var in enumerate('xyz'):
            unit = _eval_component(comp, 1.0 if var == 'x' else 0.0,
                                   1.0 if var == 'y' else 0.0,
                                   1.0 if var == 'z' else 0.0)
            if unit is None:
                return None
            rot[row, col] = unit - base
    return rot, trans


def _expand_symmetry(frac: np.ndarray, ops: list, symbols,
                     tol: float = 1e-3):
    """按对称操作展开分数坐标并去重，保持原始顺序。"""
    new_frac, new_symbols = [], []
    for op in ops:
        mat, trans = op
        new_frac.append(frac @ mat.T + trans)
        new_symbols.append(symbols)
    new_frac = np.vstack(new_frac)
    new_symbols = np.concatenate(new_symbols)

    new_frac -= np.floor(new_frac)                  # 折回 [0, 1)
    key = np.round(new_frac / tol).astype(np.int64)  # 周期性去重
    _, keep = np.unique(key, axis=0, return_index=True)
    keep.sort()                                      # unique 会重排序，恢复原始次序
    return new_frac[keep], new_symbols[keep]


# --------------------------------------------------------------------------
# 原子位点提取
# --------------------------------------------------------------------------

_CELL_TAGS = {
    'a': ('_cell_length_a',),
    'b': ('_cell_length_b',),
    'c': ('_cell_length_c',),
    'alpha': ('_cell_angle_alpha',),
    'beta': ('_cell_angle_beta',),
    'gamma': ('_cell_angle_gamma',),
}

_XYZ = ('_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z')
_CART = ('_atom_site_cartn_x', '_atom_site_cartn_y', '_atom_site_cartn_z')


def _cell_params(block: _Block):
    out = {}
    for key, names in _CELL_TAGS.items():
        value = block.tag(*names)
        out[key] = to_float(value)
    if None in out.values():
        return None
    return out


def _cell_matrix(params: dict) -> np.ndarray:
    a, b, c = params['a'], params['b'], params['c']
    al, be, ga = (np.radians(params[k]) for k in ('alpha', 'beta', 'gamma'))
    sin_ga = np.sin(ga)
    if abs(sin_ga) < 1e-12:
        return np.diag([a, b, c])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / sin_ga
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.array([[a, 0.0, 0.0],
                     [b * np.cos(ga), b * sin_ga, 0.0],
                     [cx, cy, cz]])


def _atom_loop(block: _Block):
    """找出原子位点 loop：必须含坐标三列。"""
    for labels, values in block.loops:
        if all(t in labels for t in _XYZ) or all(t in labels for t in _CART):
            return labels, values
    return None


def _read_sites(block: _Block, max_atoms: int = 2_000_000):
    found = _atom_loop(block)
    if not found:
        return None
    labels, values = found
    idx = {name: k for k, name in enumerate(labels)}

    has_cart = all(t in idx for t in _CART)
    xyz_keys = _CART if has_cart else _XYZ
    cols = [idx[t] for t in xyz_keys]

    sym_col = idx.get('_atom_site_type_symbol')
    lab_col = idx.get('_atom_site_label')
    occ_col = idx.get('_atom_site_occupancy')

    coords, symbols, occupancies = [], [], []
    for row in _rows(labels, values):
        if len(row) < len(labels):          # 残行：缺列则跳过
            remaining = {c: to_float(row[c]) for c in cols if c < len(row)}
            if len(remaining) < 3:
                continue
        vals = [to_float(row[c]) if c < len(row) else None for c in cols]
        if any(v is None for v in vals):
            continue

        raw_sym = row[sym_col] if sym_col is not None and sym_col < len(row) else None
        sym = to_symbol(raw_sym) or to_symbol(
            row[lab_col] if lab_col is not None and lab_col < len(row) else None)
        if sym is None:
            continue

        coords.append(vals)
        symbols.append(sym)
        if occ_col is not None and occ_col < len(row):
            occupancy = to_float(row[occ_col])
            occupancies.append(1.0 if occupancy is None else occupancy)

        if len(coords) >= max_atoms:
            break

    if not coords:
        return None
    return (np.asarray(coords, dtype=float), symbols,
            np.asarray(occupancies) if occupancies else None, has_cart)


def _symops(block: _Block) -> list:
    ops = []
    for labels, values in block.loops:
        for tag in ('_space_group_symop_operation_xyz',
                    '_symmetry_equiv_pos_as_xyz'):
            if tag in labels:
                col = labels.index(tag)
                for row in _rows(labels, values):
                    if col < len(row):
                        op = parse_symop(row[col])
                        if op is not None:
                            ops.append(op)
                break
    if not ops:
        value = block.tag('_symmetry_equiv_pos_as_xyz')
        if value:
            op = parse_symop(value)
            if op is not None:
                ops.append(op)
    return ops


# --------------------------------------------------------------------------
# 公开入口
# --------------------------------------------------------------------------

def parse_cif(text: str, expand_symmetry: bool = True) -> Atoms:
    """解析 CIF 文本，返回第一个可用 block 对应的 Atoms。"""
    if not text.strip():
        raise ValueError('CIF 内容为空')

    blocks = parse_blocks(text)
    if not blocks:
        raise ValueError('CIF 中没有可识别的 data block')

    last_error = None
    for block in blocks:
        try:
            sites = _read_sites(block)
            if sites is None:
                continue
            coords, symbols, occupancies, has_cart = sites

            params = _cell_params(block)
            cell = _cell_matrix(params) if params else np.zeros((3, 3))
            pbc = params is not None

            if has_cart:
                positions = coords
                frac = None
            elif params is not None:
                frac = coords
                positions = frac @ cell
            else:
                frac = coords
                positions = coords

            if expand_symmetry and frac is not None:
                ops = _symops(block)
                if len(ops) > 1:
                    frac, symbols = _expand_symmetry(frac, ops, np.asarray(symbols))
                    symbols = list(symbols)
                    positions = frac @ cell
                    occupancies = None      # 展开后位点不再与原始占位一一对应
                # 单一恒等操作或无操作：保持文件中的原始分数坐标，不做折叠

            atoms = Atoms(symbols=symbols, positions=positions,
                          cell=cell if params else None, pbc=pbc)
            if occupancies is not None and len(occupancies) == len(atoms):
                atoms.set_array('occupancy', occupancies)

            atoms.info['cif_block'] = block.name
            if block.tag('_symmetry_space_group_name_H-M',
                         '_space_group_name_H-M_alt'):
                atoms.info['space_group'] = block.tag(
                    '_symmetry_space_group_name_H-M', '_space_group_name_H-M_alt')
            if frac is not None:
                atoms.set_array('fractional_positions', frac)
            return atoms
        except Exception as exc:                 # 单个 block 失败不影响其他
            last_error = exc

    raise ValueError(f'CIF 中未找到有效的原子坐标: {last_error}')


def read_cif(filename: str, expand_symmetry: bool = True, **kwargs) -> Atoms:
    """读取 CIF 文件。自动处理 BOM、CRLF 与非 UTF-8 编码。"""
    with open(filename, 'rb') as handle:
        raw = handle.read()
    for encoding in ('utf-8-sig', 'utf-8', 'latin-1'):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:                                        # 全部失败时替换非法字节
        text = raw.decode('utf-8', errors='replace')
    return parse_cif(text, expand_symmetry=expand_symmetry)
