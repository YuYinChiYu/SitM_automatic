"""
SITM 攻击 ILLcipher-64 (10 轮, 64-bit 分组, 128-bit 密钥)

密钥划分（与 sitm_analysis.py 的 R5 约束一致；K0 全部为共享）：
  K0 (高 64 位, chunks 0-3): 全部为共享密钥，无前向/后向独有划分
  K1 (低 64 位, chunks 4-7):
    FWD_ONLY = {4,5,11,12,13,14}                     → 前向独有 (在 Lf 中遍历), 6 bits
    BWD_ONLY = {0,1,2,3,6,56,57,58,59,60,61,62,63}  → 后向独有 (在 Lb 中遍历), 13 bits
    其余为共享

向量 u（前向，R5 SubCell 输入，共 6 bits）：
  - S-box 6 (0-based idx=5): 输入 bits [2,3,4] → 3 bits
  - S-box 7 (idx=6):         输入 bit  [3]    → 1 bit
  - S-box 9 (idx=8):         输入 bits [2,3]  → 2 bits

向量 v（后向，SubCell 输出，共 15 bits）：
  - S-box 6/7/9 的全部 5 个输出 bits，各 5 bits。
"""

import itertools
import random
import time


from illcipher import (
    encrypt, decrypt, key_schedule,
    add_round_key, add_round_con,
    diffmatrix, diffmatrix_inv,
    subcell, subcell_inv,
    permucell1, permucell1_inv,
    permucell2, permucell2_inv,
    int_to_bits,
    S1, S2,
)
from sitm_functions import (
    compute_influence_matrix,
    known_bits_per_sbox,
    known_state_bits,
)

NR = 10
MATCH_ROUND = 5
CONSTRAINT_SAMPLES = 500

# ---------------------------------------------------------------------------
# 密钥划分（仅针对 K1，MSB-first bit numbering，与 sitm_analysis.py 同语义）
# K0 (高 64 位) 全部为共享密钥，不做划分。
# K1:
#   FWD_ONLY: 仅前向猜测，后向未知；不能落在 bwd_full 中。
#   BWD_ONLY: 仅后向猜测，前向未知；不能落在 fwd_full 中。
# ---------------------------------------------------------------------------
FWD_ONLY = {4, 5, 11, 12, 13, 14}
BWD_ONLY = {0, 1, 2, 3, 6, 56, 57, 58, 59, 60, 61, 62, 63}

# K1 在全局 128-bit 主密钥中的起始位置（MSB-first）
K1_GLOBAL_OFFSET = 64


def _get_partition(chunk_idx):
    """
    返回指定 chunk 的 (FWD_ONLY_set, BWD_ONLY_set)。
    K0 chunks (0-3): 空划分（全部为共享）。
    K1 chunks (4-7): 返回 +64 偏移后的划分。
    """
    if chunk_idx < 4:
        return set(), set()
    return (
        {b + K1_GLOBAL_OFFSET for b in FWD_ONLY},
        {b + K1_GLOBAL_OFFSET for b in BWD_ONLY},
    )


def derive_analysis_constraints(match_round=MATCH_ROUND, num_samples=CONSTRAINT_SAMPLES):
    """按 sitm_analysis.py 的规则推导 R_match 的 K1 约束。"""
    fwd_infl = compute_influence_matrix(match_round, 'forward', num_samples)
    bwd_infl = compute_influence_matrix(match_round, 'backward', num_samples)

    fwd_full = {b for b in range(64) if fwd_infl[b].sum() == 64}
    bwd_full = {b for b in range(64) if bwd_infl[b].sum() == 64}
    return {
        'fwd_full': fwd_full,
        'bwd_full': bwd_full,
        'both_full': fwd_full & bwd_full,
        'fwd_candidates': set(range(64)) - bwd_full,
        'bwd_candidates': set(range(64)) - fwd_full,
        'fwd_infl': fwd_infl,
        'bwd_infl': bwd_infl,
    }


def validate_partition_constraints(match_round=MATCH_ROUND, num_samples=CONSTRAINT_SAMPLES):
    """
    校验当前 Verify.py 的 K1 划分是否满足 sitm_analysis.py 的约束：
      fwd_full 中的 bit 不能做 BWD_ONLY；
      bwd_full 中的 bit 不能做 FWD_ONLY；
      both_full 必须 shared。
    同时校验 SBOX_INFO 与该划分导出的已知匹配位一致。
    """
    constraints = derive_analysis_constraints(match_round, num_samples)
    violations = []

    bad_fwd_only = FWD_ONLY & constraints['bwd_full']
    if bad_fwd_only:
        violations.append(f"FWD_ONLY contains bwd_full bits: {sorted(bad_fwd_only)}")

    bad_bwd_only = BWD_ONLY & constraints['fwd_full']
    if bad_bwd_only:
        violations.append(f"BWD_ONLY contains fwd_full bits: {sorted(bad_bwd_only)}")

    bad_shared = (FWD_ONLY | BWD_ONLY) & constraints['both_full']
    if bad_shared:
        violations.append(f"both_full bits must be shared: {sorted(bad_shared)}")

    fwd_known = known_state_bits(BWD_ONLY, constraints['fwd_infl'])
    bwd_known = known_state_bits(FWD_ONLY, constraints['bwd_infl'])
    fwd_per = known_bits_per_sbox(fwd_known)
    bwd_per = known_bits_per_sbox(bwd_known)
    derived_info = tuple(
        (i, tuple(fwd_per[i]), tuple(bwd_per[i]))
        for i in range(13)
        if fwd_per[i] and bwd_per[i]
    )
    configured_info = tuple(
        (i, tuple(inp), tuple(out))
        for i, inp, out in SBOX_INFO
    )
    if configured_info != derived_info:
        violations.append(
            f"SBOX_INFO mismatch: configured={configured_info}, derived={derived_info}"
        )

    if violations:
        raise ValueError("Verify.py constraints differ from sitm_analysis.py: "
                         + "; ".join(violations))

    return constraints, derived_info


# ---------------------------------------------------------------------------
# Chunk helpers（128-bit master key 分成 8 个 16-bit chunk）
# ---------------------------------------------------------------------------

def chunk_type(chunk_idx):
    """根据 chunk 内是否包含前向/后向未知比特，返回分类。"""
    fwd_only, bwd_only = _get_partition(chunk_idx)
    bits_in_chunk = set(range(16 * chunk_idx, 16 * chunk_idx + 16))
    has_fwd = bool(bits_in_chunk & fwd_only)
    has_bwd = bool(bits_in_chunk & bwd_only)
    if has_fwd and has_bwd:
        return '混合'
    elif has_fwd:
        return '前向'
    elif has_bwd:
        return '后向'
    return '共享'


def chunk_bit_breakdown(chunk_idx):
    """返回该 chunk 的 (前向独有比特数, 后向独有比特数, 共享比特数)。"""
    fwd_only, bwd_only = _get_partition(chunk_idx)
    fwd = bwd = shared = 0
    for b in range(16 * chunk_idx, 16 * chunk_idx + 16):
        if b in fwd_only:
            fwd += 1
        elif b in bwd_only:
            bwd += 1
        else:
            shared += 1
    return fwd, bwd, shared


def set_chunk(chunk_idx, value16):
    """构造一个 128-bit master key：仅第 chunk_idx 个 16-bit 块设为 value16，其余置 0。"""
    return value16 << (16 * (7 - chunk_idx))


# ---------------------------------------------------------------------------
# 前向 / 后向部分计算到第 5 轮 SubCell 匹配点
# ---------------------------------------------------------------------------

def forward_to_match(plaintext, master_key, rks=None):
    """明文 P → 第 5 轮 SubCell 的输入（64-bit 状态）。"""
    k0 = (master_key >> 64) & ((1 << 64) - 1)
    if rks is None:
        rks = key_schedule(master_key)
    t = plaintext ^ k0
    for i in range(1, 5):                     # 完整轮 1~4
        ri = i - 1
        if i < 3 or i > NR - 2:               # Shell round
            t = subcell(t);  t = diffmatrix(t)
            t = add_round_con(t, ri);  t = add_round_key(t, rks[ri])
            t = permucell1(t)
        else:                                 # Core round
            t = add_round_key(t, rks[ri]);  t = add_round_con(t, ri)
            t = diffmatrix(t);  t = subcell(t);  t = permucell2(t)
    # 轮 5 部分：AddRoundKey → AddRoundCon → DiffMatrix（SubCell 输入）
    ri = 4
    t = add_round_key(t, rks[ri])
    t = add_round_con(t, ri)
    t = diffmatrix(t)
    return t


def backward_to_match(ciphertext, master_key, rks=None):
    """密文 C → 第 5 轮 SubCell 的输出（64-bit 状态）。"""
    if rks is None:
        rks = key_schedule(master_key)
    t = ciphertext
    for i in range(NR, 5, -1):                # 逆向完整轮 10~6
        ri = i - 1
        if i < 3 or i > NR - 2:
            t = permucell1_inv(t);  t = add_round_key(t, rks[ri])
            t = add_round_con(t, ri);  t = diffmatrix_inv(t);  t = subcell_inv(t)
        else:
            t = permucell2_inv(t);  t = subcell_inv(t);  t = diffmatrix_inv(t)
            t = add_round_con(t, ri);  t = add_round_key(t, rks[ri])
    # 轮 5：仅撤销 PermuCell2 → 得到 SubCell 输出
    t = permucell2_inv(t)
    return t


# ---------------------------------------------------------------------------
# 从匹配点状态提取向量 u（前向，6 bits）与向量 v（后向，15 bits）
# ---------------------------------------------------------------------------

def extract_u(state):
    """从 SubCell 输入状态中提取向量 u。
    u = (u6, u7, u9)，其中位集合由 SBOX_INFO 给出。
    """
    bits = int_to_bits(state, 64)
    return tuple(
        tuple(bits[sbox_start(sbox_idx) + p] for p in in_pos)
        for sbox_idx, in_pos, _ in SBOX_INFO
    )


def extract_v(state):
    """从 SubCell 输出状态中提取向量 v。
    v = (v6, v7, v9)，其中位集合由 SBOX_INFO 给出。
    """
    bits = int_to_bits(state, 64)
    return tuple(
        tuple(bits[sbox_start(sbox_idx) + p] for p in out_pos)
        for sbox_idx, _, out_pos in SBOX_INFO
    )


# ---------------------------------------------------------------------------
# 即时匹配（Instant Matching）表构建与算法
# 对 3 个匹配 S-box，预建反向表 T_j[v_j] = { 所有与 v_j 相容的 u_j }
# ---------------------------------------------------------------------------

# (sbox_index_0based, known_input_positions, known_output_positions)
SBOX_INFO = [
    (5, (2, 3, 4),    (0, 1, 2, 3, 4)),
    (6, (3,),         (0, 1, 2, 3, 4)),
    (8, (2, 3),       (0, 1, 2, 3, 4)),
]


def sbox_width(sbox_idx):
    return 4 if sbox_idx == 12 else 5


def sbox_start(sbox_idx):
    return 60 if sbox_idx == 12 else sbox_idx * 5


def sbox_lut(sbox_idx):
    return S1 if sbox_idx == 12 else S2


def _build_instant_match_tables():
    """
    对每个匹配 S-box 构建反向表 T_j：
      T_j[v_j] = { u_j | 存在完整 S-box 输入 x，使得 x 的已知输入位 = u_j
                           且 S(x) 的已知输出位 = v_j }
    返回 3 个 dict 组成的列表，索引 0/1/2 分别对应 S-box 6/7/9。
    """
    tables = []
    for sbox_idx, in_pos, out_pos in SBOX_INFO:
        reverse = {}
        width = sbox_width(sbox_idx)
        lut = sbox_lut(sbox_idx)
        for x in range(1 << width):
            y = lut[x]
            xb = int_to_bits(x, width)
            yb = int_to_bits(y, width)
            u = tuple(xb[p] for p in in_pos)
            v = tuple(yb[p] for p in out_pos)
            if v not in reverse:
                reverse[v] = set()
            reverse[v].add(u)
        tables.append(reverse)
    return tables


INSTANT_MATCH_TABLES = _build_instant_match_tables()


def instant_match(Lf, Lb):
    """
    标准即时匹配算法

    参数:
      Lf : dict，键 = u_tuple，值 = list of fwd_guess
      Lb : list of (v_tuple, bwd_guess)

    返回:
      candidates : set of (fwd_guess, bwd_guess)
    """
    tables = INSTANT_MATCH_TABLES
    candidates = set()
    for v_tuple, bwd_guess in Lb:
        compatible_u_parts = [
            table.get(v_part, set())
            for table, v_part in zip(tables, v_tuple)
        ]
        if any(not part for part in compatible_u_parts):
            continue
        for u_tuple in itertools.product(*compatible_u_parts):
            if u_tuple in Lf:
                for fwd_guess in Lf[u_tuple]:
                    candidates.add((fwd_guess, bwd_guess))
    return candidates


# ---------------------------------------------------------------------------
# 单 chunk 攻击（严格按照 SITM 攻击模型）
# ---------------------------------------------------------------------------

def attack_chunk(chunk_idx, true_chunk_value, num_pairs, seed=0):
    """
    对单个 16-bit chunk 执行 Sieve-in-the-Middle 攻击。
    """
    # ---- 1. 构建真值主密钥并生成明密文对 ----
    master_key_true = set_chunk(chunk_idx, true_chunk_value)

    rng = random.Random(seed * 1000 + chunk_idx)
    pairs = []
    used_pt = set()
    while len(pairs) < num_pairs:
        pt = rng.getrandbits(64)
        if pt in used_pt:
            continue
        used_pt.add(pt)
        pairs.append((pt, encrypt(pt, master_key_true)))

    pt0, ct0 = pairs[0]

    # ---- 2. 分析 chunk 内的 bit 分布 ----
    fwd_count, bwd_count, shared_count = chunk_bit_breakdown(chunk_idx)

    fwd_only_set, bwd_only_set = _get_partition(chunk_idx)
    chunk_start = 16 * chunk_idx
    fwd_bits = []     # 该 chunk 内属于 fwd_only_set 的局部 bit 位置（前向独有）
    bwd_bits = []     # 该 chunk 内属于 bwd_only_set 的局部 bit 位置（后向独有）
    shared_bits = []  # 共享 bit 的局部位置
    for local_pos in range(16):
        global_pos = chunk_start + local_pos
        if global_pos in fwd_only_set:
            fwd_bits.append(local_pos)
        elif global_pos in bwd_only_set:
            bwd_bits.append(local_pos)
        else:
            shared_bits.append(local_pos)

    # 辅助：将 shared/fwd/bwd 三部分组合回 16-bit chunk_val
    def compose_chunk(shared_val, fwd_val, bwd_val):
        val = 0
        for i, pos in enumerate(shared_bits):
            bit = (shared_val >> (shared_count - 1 - i)) & 1
            val |= (bit << (15 - pos))
        for i, pos in enumerate(fwd_bits):
            bit = (fwd_val >> (fwd_count - 1 - i)) & 1
            val |= (bit << (15 - pos))
        for i, pos in enumerate(bwd_bits):
            bit = (bwd_val >> (bwd_count - 1 - i)) & 1
            val |= (bit << (15 - pos))
        return val

    # key_schedule 缓存
    _rks_cache = {}
    def get_rks(mk):
        if mk not in _rks_cache:
            _rks_cache[mk] = key_schedule(mk)
        return _rks_cache[mk]

    # 快速验证（仅需 1 对明密文）
    def verify_fast(mk):
        return encrypt(pt0, mk) == ct0

    survivors = []
    t0 = time.time()

    # ======== 退化情况 1：全共享 chunk（前向=后向=0）========
    if fwd_count == 0 and bwd_count == 0:
        for guess in range(1 << 16):
            mk = set_chunk(chunk_idx, guess)
            if verify_fast(mk):
                survivors.append(guess)
        elapsed = time.time() - t0
        return survivors, elapsed

    # ======== 退化情况 2：仅有后向独有（前向=0，后向>0）========
    if fwd_count == 0 and bwd_count > 0:
        for shared_guess in range(1 << shared_count):
            for bwd_guess in range(1 << bwd_count):
                chunk_val = compose_chunk(shared_guess, 0, bwd_guess)
                mk = set_chunk(chunk_idx, chunk_val)
                if verify_fast(mk):
                    survivors.append(chunk_val)
        elapsed = time.time() - t0
        return survivors, elapsed

    # ======== 退化情况 3：仅有前向独有（后向=0，前向>0）========
    if bwd_count == 0 and fwd_count > 0:
        for shared_guess in range(1 << shared_count):
            for fwd_guess in range(1 << fwd_count):
                chunk_val = compose_chunk(shared_guess, fwd_guess, 0)
                mk = set_chunk(chunk_idx, chunk_val)
                if verify_fast(mk):
                    survivors.append(chunk_val)
        elapsed = time.time() - t0
        return survivors, elapsed

    # ======== 标准 SITM 流程（混合型 chunk：前向>0 且 后向>0）========
    for shared_guess in range(1 << shared_count):
        # ---- 构建 L_f（前向列表）----
        # Bug 2 Fix: 只用共享密钥 + 前向独有密钥，后向独有部分置 0
        Lf = {}  # u_tuple → list of fwd_guess
        for fwd_guess in range(1 << fwd_count):
            chunk_val = compose_chunk(shared_guess, fwd_guess, 0)
            mk = set_chunk(chunk_idx, chunk_val)
            rks = get_rks(mk)
            f_state0 = forward_to_match(pt0, mk, rks)
            u = extract_u(f_state0)
            Lf.setdefault(u, []).append(fwd_guess)

        # ---- 构建 L_b（后向列表）----
        # Bug 2 Fix: 只用共享密钥 + 后向独有密钥，前向独有部分置 0
        Lb = []  # list of (v_tuple, bwd_guess)
        for bwd_guess in range(1 << bwd_count):
            chunk_val = compose_chunk(shared_guess, 0, bwd_guess)
            mk = set_chunk(chunk_idx, chunk_val)
            rks = get_rks(mk)
            b_state0 = backward_to_match(ct0, mk, rks)
            v = extract_v(b_state0)
            Lb.append((v, bwd_guess))

        # ---- 即时匹配合并（Instant Matching）----
        candidate_set = instant_match(Lf, Lb)

        # ---- 快速验证所有候选 ----
        verified_from_match = []
        for fwd_guess, bwd_guess in candidate_set:
            chunk_val = compose_chunk(shared_guess, fwd_guess, bwd_guess)
            mk = set_chunk(chunk_idx, chunk_val)
            if verify_fast(mk):
                verified_from_match.append(chunk_val)

        survivors.extend(verified_from_match)

    elapsed = time.time() - t0
    return survivors, elapsed


# ---------------------------------------------------------------------------
# 自测：验证加解密及匹配点一致性
# ---------------------------------------------------------------------------

def self_test(num_trials=20, seed=12345):
    """
    往返自测：
      1) decrypt(encrypt(P, K), K) == P
      2) SubCell(forward_to_match(P,K)) == backward_to_match(encrypt(P,K),K)
    """
    rng = random.Random(seed)
    print("=" * 60)
    print("算法正确性往返自测")
    print("-" * 60)
    fail = 0
    for t in range(num_trials):
        mk = rng.getrandbits(128)
        pt = rng.getrandbits(64)
        ct = encrypt(pt, mk)
        ok_rt = (decrypt(ct, mk) == pt)
        f_in = forward_to_match(pt, mk)
        b_out = backward_to_match(ct, mk)
        ok_meet = (subcell(f_in) == b_out)
        ok_all = ok_rt and ok_meet
        if not ok_all:
            fail += 1
            print(f"  第 {t + 1:>2} 次：往返={ok_rt} 中点一致={ok_meet}")
    print(f"自测通过 {num_trials - fail}/{num_trials} 次")
    print("=" * 60)
    return fail == 0


# ---------------------------------------------------------------------------
# 完整验证：攻击全部 8 个 chunk
# ---------------------------------------------------------------------------

def main():
    if not self_test():
        print("自测失败，终止攻击。")
        return

    constraints, match_info = validate_partition_constraints()
    print("R5 约束检查通过：Verify.py 与 sitm_analysis.py 一致")

    rng = random.Random(20260419)
    true_vals = [rng.getrandbits(16) for _ in range(8)]

    print("ILLcipher-64 Sieve-in-the-Middle 验证 (Nr=10, n=64, k=128)")
    print("=" * 90)
    print(f"{'分块':>4} {'类型':>6} {'前向bit':>8} {'后向bit':>8} {'共享bit':>8} "
          f"{'真值':>8} {'候选数':>8} {'命中':>5} {'耗时(s)':>8}  恢复出的候选")
    print("-" * 90)

    all_ok = True
    for j in range(8):
        ctype = chunk_type(j)
        fwd_b, bwd_b, sh_b = chunk_bit_breakdown(j)
        survivors, elapsed = attack_chunk(j, true_vals[j], num_pairs=1)
        hit = true_vals[j] in survivors
        all_ok = all_ok and hit

        shown = ", ".join(f"{v:#06x}" for v in survivors[:8])
        if len(survivors) > 8:
            shown += f", ...(+{len(survivors) - 8})"
        print(f"{j:>4} {ctype:>7} {fwd_b:>7} {bwd_b:>8} {sh_b:>8} "
              f"{true_vals[j]:>#14x} {len(survivors):>5} "
              f"{('是' if hit else '否'):>7} {elapsed:>8.1f}     {shown}",
              flush=True)

    print("=" * 110)
    print("全部分块均成功恢复" if all_ok else "失败：存在未恢复的真值密钥")


if __name__ == "__main__":
    main()
