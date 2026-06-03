"""
绘制 SITM 攻击 ILLcipher-64 扩散图

左侧：前向计算（P → 匹配点 SubCell 输入），从上往下
右侧：后向计算（C → 匹配点 SubCell 输出），从下往上
中间：S-box 匹配层（粉红 = 可筛选）

颜色：
  棕/蓝/绿/红（按 16-bit 字）= 已知 bit
  黑 = 未知/被污染 bit
  黄 = 本轮轮密钥注入新增的不确定 bit
  粉红 S-box = 匹配点S盒
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import math

from illcipher import PT1_64, PT2_64, S1, S2
from sitm_functions import compute_influence_matrix, known_state_bits

# ============ 常量 ============
MATRIX_SHIFTS = [[0,4,6,7,13],[0,2,5,7,14],[0,1,6,7,10],[0,4,6,7,13]]
INV_MATRIX_SHIFTS = [[0,2,5,7,8],[0,5,6,8,15],[0,4,11,12,13],[0,2,5,7,8]]

INV_PT2_64 = [0]*64
for _i in range(64): INV_PT2_64[PT2_64[_i]] = _i
INV_PT1_64 = [0]*64
for _i in range(64): INV_PT1_64[PT1_64[_i]] = _i

RK_SHIFT = {1:3,2:5,3:7,4:10,5:12,6:15,7:17,8:20,9:23,10:25}

# ============ 辅助 ============
def compute_sieving_probability(sbox, n_bits, input_positions, output_positions):
    """
    Compute the sieving probability π_{I,J} for a given S-box.

    Given subsets I (input bit positions) and J (output bit positions),
    π_{I,J} is the fraction of (u, v) pairs in F_2^|I| × F_2^|J| that
    correspond to some valid transition through the S-box.

    Args:
        sbox: lookup table (list)
        n_bits: total input/output bits of the S-box
        input_positions: list of input bit indices we observe (0 = MSB)
        output_positions: list of output bit indices we observe (0 = MSB)

    Returns:
        (π, valid_pairs): sieving probability and set of valid (u,v) pairs
    """
    m = len(input_positions)
    p = len(output_positions)

    valid_pairs = set()

    for x in range(1 << n_bits):
        y = sbox[x]
        # Extract observed input bits
        u = 0
        for i, pos in enumerate(input_positions):
            bit = (x >> (n_bits - 1 - pos)) & 1
            u = (u << 1) | bit
        # Extract observed output bits
        v = 0
        for i, pos in enumerate(output_positions):
            bit = (y >> (n_bits - 1 - pos)) & 1
            v = (v << 1) | bit
        valid_pairs.add((u, v))

    total_pairs = (1 << m) * (1 << p)
    pi = len(valid_pairs) / total_pairs

    return pi, valid_pairs

def _md16(ib, sh):
    return {(b+s)%16 for b in ib for s in sh}

def _sbe(bits):
    r = set()
    for b in bits:
        if b < 60:
            base = (b//5)*5
            for j in range(5): r.add(base+j)
        else:
            for j in range(60,64): r.add(j)
    return r

def _me(bits, sl):
    r = set()
    for br in range(4):
        bs = br*16
        bb = {b-bs for b in bits if bs<=b<bs+16}
        if bb:
            for o in _md16(bb, sl[br]): r.add(bs+o)
    return r

def _ime(bits, inv_sl):
    r = set()
    for br in range(4):
        bs = br*16
        bb = {b-bs for b in bits if bs<=b<bs+16}
        if bb:
            for o in _md16(bb, inv_sl[br]): r.add(bs+o)
    return r

def _pm(bits, t):
    return {t[b] for b in bits}

def _k1_affected(b):
    return {0,1,2,3} if b in {0,1,2,3} else {b}

def _word_color(bit):
    w = bit // 16
    return ['#CD853F','#4169E1','#228B22','#DC143C'][w]

# ============ 构建状态序列 ============
def build_forward_states(match_round, fwd_unk_k1):
    """返回 [(label, unk_set, row_type, new_bits_or_None), ...]"""
    rows = []
    unk = set()
    rows.append(('P', set(unk), 'pt', None))
    # Shell R1, R2
    for r in [1,2]:
        unk = _sbe(unk)
        rows.append(('sbox', set(unk), 'op', None))
        unk = _me(unk, MATRIX_SHIFTS)
        rows.append(('matrix', set(unk), 'op', None))
        rows.append((f'\u2295sk{r}', set(unk), 'key', None))
        unk = _pm(unk, PT1_64)
        rows.append(('perm1', set(unk), 'op', None))
        rows.append((str(r), set(unk), 'round', None))
    # Core R3..match_round
    for r in range(3, match_round+1):
        # AddRoundKey K1
        inj = set()
        for b in fwd_unk_k1:
            for ab in _k1_affected(b):
                inj.add((ab - RK_SHIFT[r]) % 64)
        new = inj - unk
        unk = unk | inj
        rows.append((f'\u2295sk{r}', set(unk), 'key', set(new)))
        # DiffMatrix
        unk = _me(unk, MATRIX_SHIFTS)
        if r < match_round:
            rows.append(('matrix', set(unk), 'op', None))
            unk = _sbe(unk)
            rows.append(('sbox', set(unk), 'op', None))
            unk = _pm(unk, PT2_64)
            rows.append(('perm2', set(unk), 'op', None))
            rows.append((str(r), set(unk), 'round', None))
        else:
            rows.append(('matrix', set(unk), 'op', None))
            rows.append((f"{r}'", set(unk), 'round', None))
    return rows

def build_backward_states(match_round, bwd_unk_k1):
    """返回 [(label, unk_set, row_type, new_bits_or_None), ...] 从 C 端开始"""
    rows = []
    unk = set()
    rows.append(('C', set(unk), 'ct', None))
    # Shell R10, R9 (inverse)
    for r in [10, 9]:
        unk = _pm(unk, INV_PT1_64)
        rows.append(('perm1\u207B\u00B9', set(unk), 'op', None))
        rows.append((f'\u2295sk{r}', set(unk), 'key', None))
        unk = _ime(unk, INV_MATRIX_SHIFTS)
        rows.append(('matrix\u207B\u00B9', set(unk), 'op', None))
        unk = _sbe(unk)
        rows.append(('sbox\u207B\u00B9', set(unk), 'op', None))
        rows.append((f"{r}'", set(unk), 'round', None))
    # Core R8..match_round+1
    for r in range(8, match_round, -1):
        unk = _pm(unk, INV_PT2_64)
        rows.append(('perm2\u207B\u00B9', set(unk), 'op', None))
        unk = _sbe(unk)
        rows.append(('sbox\u207B\u00B9', set(unk), 'op', None))
        unk = _ime(unk, INV_MATRIX_SHIFTS)
        rows.append(('matrix\u207B\u00B9', set(unk), 'op', None))
        # AddRoundKey
        inj = set()
        for b in bwd_unk_k1:
            for ab in _k1_affected(b):
                inj.add((ab - RK_SHIFT[r]) % 64)
        new = inj - unk
        unk = unk | inj
        rows.append((f'\u2295sk{r}', set(unk), 'key', set(new)))
        rows.append((f"{r}'", set(unk), 'round', None))
    # 最后 match 轮的 InvPerm -> SubCell output
    unk = _pm(unk, INV_PT2_64)
    rows.append(('perm2\u207B\u00B9', set(unk), 'op', None))
    rows.append((f"{match_round}'", set(unk), 'round', None))
    return rows

# ============ 主绘图函数 ============
def draw_sitm_attack(match_round, fwd_only, bwd_only, filename):
    fwd_only = set(fwd_only)
    bwd_only = set(bwd_only)
    shared = set(range(64)) - fwd_only - bwd_only

    fwd_unk_k1 = bwd_only
    bwd_unk_k1 = fwd_only

    fwd_rows = build_forward_states(match_round, fwd_unk_k1)
    bwd_rows = build_backward_states(match_round, bwd_unk_k1)

    # 精确影响矩阵
    print(f'  Computing influence matrices for R{match_round}...')
    fwd_inf = compute_influence_matrix(match_round, "forward", num_samples=500)
    bwd_inf = compute_influence_matrix(match_round, "backward", num_samples=500)
    exact_fwd_known = known_state_bits(fwd_unk_k1, fwd_inf)
    exact_bwd_known = known_state_bits(bwd_unk_k1, bwd_inf)
    exact_fwd_unk = set(range(64)) - exact_fwd_known
    exact_bwd_unk = set(range(64)) - exact_bwd_known
    print(f'  fwd_known={len(exact_fwd_known)}/64, bwd_known={len(exact_bwd_known)}/64')

    # ---- 布局参数 ----
    CW = 0.52
    CH = 0.58
    NBITS = 64
    LABEL_PAD = 2.8
    TOP_PAD = 2.6
    BOTTOM_PAD = 1.8
    MATCH_GAP = 1.0
    SIDE_GAP = 8.5

    n_fwd = len(fwd_rows)
    n_bwd = len(bwd_rows)
    total_h = n_fwd + n_bwd + 1

    fig_w = NBITS * CW * 0.36 + SIDE_GAP * 0.18 + 3.5
    fig_h = (TOP_PAD + total_h * CH + BOTTOM_PAD + 2.4) * 0.42 + 1.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_aspect('equal')
    ax.axis('off')

    X0 = 0

    def bit_x(x0, bit):
        """bit 0 在左, bit 63 在右"""
        return x0 + bit * CW

    def draw_state_row(x0, y, unk, new_bits=None):
        for b in range(NBITS):
            x = bit_x(x0, b)
            if new_bits and b in new_bits:
                fc = '#FFFF00'
            elif b in unk:
                fc = '#000000'
            else:
                fc = _word_color(b)
            rect = patches.Rectangle((x, y), CW, CH,
                    lw=0.28, edgecolor='#555555', facecolor=fc)
            ax.add_patch(rect)

    def draw_sbox_match(x0, y, in_unk, out_unk):
        for si in range(13):
            sz = 5 if si < 12 else 4
            base = si*5 if si < 12 else 60
            sb = list(range(base, base+sz))
            in_k = sorted([bb-base for bb in sb if bb not in in_unk])
            out_k = sorted([bb-base for bb in sb if bb not in out_unk])
            if in_k and out_k:
                sbox = S2 if si < 12 else S1
                pi, _ = compute_sieving_probability(sbox, sz, in_k, out_k)
                is_sieve = pi < 1.0
            else:
                is_sieve = False
            fc = '#FF69B4' if is_sieve else '#FFFFFF'
            x_start = bit_x(x0, base)
            rect = patches.Rectangle((x_start, y), sz*CW, CH,
                    lw=1.0, edgecolor='black', facecolor=fc)
            ax.add_patch(rect)
            tc = 'black' if is_sieve else '#AAAAAA'
            ax.text(x_start + sz*CW/2, y + CH/2, 'S',
                    ha='center', va='center', fontsize=9, fontweight='bold', color=tc)

    def draw_key_numbers(x0, y, unk_k1, rnd):
        """在轮密钥行上标注注入的 bit 位置号"""
        inj = set()
        for b in unk_k1:
            for ab in _k1_affected(b):
                inj.add((ab - RK_SHIFT[rnd]) % 64)
        positions = sorted(inj)
        if not positions:
            return
        # 选代表性位置标注
        labeled = []
        for p in positions:
            if not labeled or p - labeled[-1] >= 3:
                labeled.append(p)
        for p in labeled:
            x = bit_x(x0, p) + CW/2
            ax.text(x, y + CH + 0.08, str(p),
                    ha='center', va='bottom', fontsize=5.8, color='#333333',
                    fontweight='bold')

    def draw_bit_scale(x0, y, above=True):
        for b in range(0, NBITS, 16):
            tx = bit_x(x0, b) + CW/2
            ty = y + 0.35 if above else y - 0.35
            va = 'bottom' if above else 'top'
            ax.text(tx, ty, str(b),
                    ha='center', va=va, fontsize=7.5, color='gray',
                    fontweight='bold')

    # ---- 绘制前向（上方，从上往下）----
    y_top = TOP_PAD + total_h * CH + MATCH_GAP + BOTTOM_PAD
    draw_bit_scale(X0, y_top, above=True)

    fwd_ys = []
    for i, (label, unk, rtype, new) in enumerate(fwd_rows):
        y = y_top - (i + 1) * CH
        fwd_ys.append(y)
        draw_state_row(X0, y, unk, new)

        if rtype == 'key' and new:
            rnd_num = int(''.join(c for c in label if c.isdigit()))
            if 3 <= rnd_num <= 8:
                draw_key_numbers(X0, y, fwd_unk_k1, rnd_num)

        lx = X0 - LABEL_PAD
        if rtype in ('pt', 'ct'):
            ax.text(lx, y + CH/2, label, ha='right', va='center',
                    fontsize=11, fontweight='bold')
        elif rtype == 'key':
            ax.text(lx, y + CH/2, label, ha='right', va='center',
                    fontsize=8.5, color='#B22222', fontstyle='italic')
            ax.annotate('', xy=(X0 - 0.10, y + CH/2),
                        xytext=(X0 - 1.15, y + CH/2),
                        arrowprops=dict(arrowstyle='->', color='#B22222', lw=1.0))
        elif rtype == 'round':
            ax.text(lx, y + CH/2, label, ha='right', va='center',
                    fontsize=9.2, fontweight='bold')
        else:
            ax.text(lx, y + CH/2, label, ha='right', va='center',
                    fontsize=8.2, color='#555555')

    # 前向大箭头
    ax.annotate('', xy=(X0 - 5.0, fwd_ys[-1] + CH/2),
                xytext=(X0 - 5.0, fwd_ys[0] + CH/2),
                arrowprops=dict(arrowstyle='-|>', color='red', lw=2.5))
    ax.text(X0 - 6.0, (fwd_ys[0] + fwd_ys[-1]) / 2 + CH/2, 'Forward',
            ha='center', va='center', rotation=90,
            fontsize=10, color='red', fontweight='bold')

    # ---- S-box 匹配层 ----
    y_match = fwd_ys[-1] - MATCH_GAP - CH
    draw_sbox_match(X0, y_match, exact_fwd_unk, exact_bwd_unk)
    ax.text(X0 + NBITS*CW/2, y_match - 0.52,
            'S-box match point', ha='center', fontsize=10,
            fontweight='bold', color='#FF1493')

    # ---- 绘制后向（下方，翻转显示，使匹配轮靠上）----
    y_bwd_top = y_match - MATCH_GAP
    bwd_ys = []
    bwd_rows_display = list(reversed(bwd_rows))
    for i, (label, unk, rtype, new) in enumerate(bwd_rows_display):
        y = y_bwd_top - (i + 1) * CH
        bwd_ys.append(y)
        draw_state_row(X0, y, unk, new)

        if rtype == 'key' and new:
            rnd_num = int(''.join(c for c in label if c.isdigit()))
            if 3 <= rnd_num <= 8:
                draw_key_numbers(X0, y, bwd_unk_k1, rnd_num)

        rx = X0 + NBITS*CW + LABEL_PAD
        if rtype in ('pt', 'ct'):
            ax.text(rx, y + CH/2, label, ha='left', va='center',
                    fontsize=11, fontweight='bold')
        elif rtype == 'key':
            ax.text(rx, y + CH/2, label, ha='left', va='center',
                    fontsize=8.5, color='#B22222', fontstyle='italic')
            ax.annotate('', xy=(X0 + NBITS*CW + 0.10, y + CH/2),
                        xytext=(X0 + NBITS*CW + 1.15, y + CH/2),
                        arrowprops=dict(arrowstyle='->', color='#B22222', lw=1.0))
        elif rtype == 'round':
            ax.text(rx, y + CH/2, label, ha='left', va='center',
                    fontsize=9.2, fontweight='bold')
        else:
            ax.text(rx, y + CH/2, label, ha='left', va='center',
                    fontsize=8.2, color='#555555')

    # 比特标尺（每 16 bit 标一个）
    y_btm = bwd_ys[-1]
    draw_bit_scale(X0, y_btm, above=False)

    # 后向大箭头
    ax.annotate('', xy=(X0 + NBITS*CW + 5.0, bwd_ys[0] + CH/2),
                xytext=(X0 + NBITS*CW + 5.0, bwd_ys[-1] + CH/2),
                arrowprops=dict(arrowstyle='-|>', color='blue', lw=2.5))
    ax.text(X0 + NBITS*CW + 6.0, (bwd_ys[0] + bwd_ys[-1]) / 2 + CH/2, 'Backward',
            ha='center', va='center', rotation=270,
            fontsize=10, color='blue', fontweight='bold')

    # ---- 信息面板 ----
    f = len(fwd_only)
    b = len(bwd_only)
    s = len(shared)
    kappa = 64 + s
    if f > 0 and b > 0:
        log_main = kappa + max(f,b) + math.log2(1 + 2**(min(f,b)-max(f,b)))
    elif f > 0:
        log_main = kappa + f
    elif b > 0:
        log_main = kappa + b
    else:
        log_main = kappa

    total_log_pi = 0
    sieve_strs = []
    for si in range(13):
        sz = 5 if si < 12 else 4
        base = si*5 if si < 12 else 60
        sb = list(range(base, base+sz))
        in_k = sorted([bb-base for bb in sb if bb not in exact_fwd_unk])
        out_k = sorted([bb-base for bb in sb if bb not in exact_bwd_unk])
        if in_k and out_k:
            sbox = S2 if si < 12 else S1
            pi_v, _ = compute_sieving_probability(sbox, sz, in_k, out_k)
            if pi_v < 1.0:
                lp = math.log2(pi_v)
                total_log_pi += lp
                sieve_strs.append(f'S{si}: \u03C0=2^{lp:.1f}')

    log_verify = 128 + total_log_pi
    log_total = max(log_main, log_verify)
    advantage = 128 - log_total

    px = X0 + NBITS*CW + SIDE_GAP
    py = y_top - 0.2
    lines = [
        f'Match: R{match_round} SubCell',
        f'f = {f},  b = {b},  s = {s},  \u03BA = {kappa}',
        '',
        f'fwd_only ({f}): {sorted(fwd_only)}  (guess fwd only)',
        f'bwd_only ({b}): {sorted(bwd_only)}  (guess bwd only)',
        '',
        f'fwd unk K1 = bwd_only({b}bit) -> state known: {len(exact_fwd_known)}/64',
        f'bwd unk K1 = fwd_only({f}bit) -> state known: {len(exact_bwd_known)}/64',
        f'\u03C0 = 2^{total_log_pi:.1f}',
        '',
        f'main   = 2^{log_main:.1f}',
        f'verify = 2^{log_verify:.1f}',
        f'total  = 2^{log_total:.1f}',
        f'advantage = 2^{advantage:.1f}',
    ]
    if sieve_strs:
        lines += ['', 'Sieving S-boxes:'] + [f'  {ss}' for ss in sieve_strs]

    # for j, line in enumerate(lines):
    #     ax.text(px, py - j*0.6, line, ha='left', va='top',
    #             fontsize=4.5, fontfamily='monospace', color='#222222')

    # 图例
    lx = px
    ly = py - 0.8
    items = [
        ('#CD853F',  'Known (word 0, bit 0-15)'),
        ('#4169E1',  'Known (word 1, bit 16-31)'),
        ('#228B22',  'Known (word 2, bit 32-47)'),
        ('#DC143C',  'Known (word 3, bit 48-63)'),
        ('#000000',  'Unknown'),
        ('#FFFF00',  'Newly uncertain (key XOR)'),
        ('#FF69B4',  'Sieving S-box'),
    ]
    ax.text(lx, ly + 0.1, 'Legend:', fontsize=8, fontweight='bold')
    for k, (c, t) in enumerate(items):
        yy = ly - (k+1)*0.72
        rect = patches.Rectangle((lx, yy), 1.0, 0.46,
                lw=0.5, edgecolor='black', facecolor=c)
        ax.add_patch(rect)
        ax.text(lx+1.4, yy+0.23, t, ha='left', va='center', fontsize=5.6)

    # 标题
    title = f'Fig. SITM attack on 10-round ILLcipher-64 (match at R{match_round})'
    ax.text(X0 + NBITS*CW / 2, y_top + 1.3, title,
            ha='center', va='bottom', fontsize=13, fontweight='bold')

    # 范围
    all_ys = fwd_ys + bwd_ys + [y_match]
    ax.set_xlim(X0 - 8.5, px + 12.5)
    ax.set_ylim(min(all_ys) - 1.8, y_top + 2.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {filename}')


# ============ main ============
if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
    os.makedirs(out_dir, exist_ok=True)

    # R5 最优攻击（全局最优）
    print('Generating R5 diagram...')
    draw_sitm_attack(
        match_round=5,
        fwd_only={4, 5, 11, 12, 13, 14},
        bwd_only={0, 1, 2, 3, 6, 56, 57, 58, 59, 60, 61, 62, 63},
        filename=os.path.join(out_dir, 'sitm_attack_R5.png'),
    )

    print('Done.')
