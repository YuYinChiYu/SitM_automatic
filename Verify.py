"""
SITM 攻击 ILLcipher-64 (10 轮, 64-bit 分组, 128-bit 密钥)

攻击模型：
  1. 外层遍历共享密钥空间（当前 chunk 内的 shared bits）。
  2. 对每种 shared 猜测，遍历前向独有密钥 K_f，计算前向匹配点向量 u，构建列表 L_f。
  3. 同时遍历后向独有密钥 K_b，计算后向匹配点向量 v，构建列表 L_b。
  4. 通过即时匹配（Instant Matching）算法合并 L_f 与 L_b，得到候选 (K_f, K_b)。
  5. 用多对明密文对候选进行 sbox_match 验证， survivor 即为恢复出的密钥。

密钥划分（基于影响矩阵分析，bit 0 = MSB）：
  FWD_UNK = {0,1,2,3,6,56,57,58,59,60,61,62,63}  → 前向未知比特，即后向独有 K_b，13 bits
  BWD_UNK = {4,5,11,12,13,14}                     → 后向未知比特，即前向独有 K_f，6 bits

向量 u（前向，SubCell 输入，共 6 bits）：
  - S-box 5 (0-based idx=4): 输入 bits [2,3,4] → 3 bits
  - S-box 6 (idx=5):          输入 bit  [3]    → 1 bit
  - S-box 8 (idx=7):          输入 bits [2,3]  → 2 bits

向量 v（后向，SubCell 输出，共 15 bits）：
  - S-box 5/6/8 的全部 5 个输出 bits，各 5 bits。
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
    int_to_bits, bits_to_int,
    S2,
)

NR = 10

# ---------------------------------------------------------------------------
# 密钥划分（MSB-first bit numbering）
# ---------------------------------------------------------------------------
FWD_UNK = {0, 1, 2, 3, 6, 56, 57, 58, 59, 60, 61, 62, 63}
BWD_UNK = {4, 5, 11, 12, 13, 14}

# ---------------------------------------------------------------------------
# Chunk helpers（128-bit master key 分成 8 个 16-bit chunk）
# ---------------------------------------------------------------------------

def chunk_type(chunk_idx):
    """根据 chunk 内是否包含前向/后向未知比特，返回分类。"""
    bits_in_chunk = set(range(16 * chunk_idx, 16 * chunk_idx + 16))
    if bits_in_chunk & BWD_UNK:
        return '前向'
    if bits_in_chunk & FWD_UNK:
        return '后向'
    return '共享'


def chunk_bit_breakdown(chunk_idx):
    """返回该 chunk 的 (前向独有比特数, 后向独有比特数, 共享比特数)。"""
    fwd = bwd = shared = 0
    for b in range(16 * chunk_idx, 16 * chunk_idx + 16):
        if b in BWD_UNK:
            fwd += 1          # BWD_UNK 中的 bit 是前向独有的
        elif b in FWD_UNK:
            bwd += 1          # FWD_UNK 中的 bit 是后向独有的
        else:
            shared += 1       # 其余为共享
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
    u = (u5, u6, u8)，其中
      u5 = S-box 5 (idx=4) 的输入 bits [2,3,4] → 3 bits
      u6 = S-box 6 (idx=5) 的输入 bit  [3]    → 1 bit
      u8 = S-box 8 (idx=7) 的输入 bits [2,3]  → 2 bits
    """
    bits = int_to_bits(state, 64)
    u5 = tuple(bits[4 * 5 + p] for p in (2, 3, 4))
    u6 = tuple(bits[5 * 5 + p] for p in (3,))
    u8 = tuple(bits[7 * 5 + p] for p in (2, 3))
    return (u5, u6, u8)


def extract_v(state):
    """从 SubCell 输出状态中提取向量 v。
    v = (v5, v6, v8)，其中每个均为对应 S-box 的全部 5 个输出 bits。
    """
    bits = int_to_bits(state, 64)
    v5 = tuple(bits[4 * 5 + p] for p in (0, 1, 2, 3, 4))
    v6 = tuple(bits[5 * 5 + p] for p in (0, 1, 2, 3, 4))
    v8 = tuple(bits[7 * 5 + p] for p in (0, 1, 2, 3, 4))
    return (v5, v6, v8)


# ---------------------------------------------------------------------------
# 即时匹配（Instant Matching）表构建与算法
# 对 3 个匹配 S-box，预建反向表 T_j[v_j] = { 所有与 v_j 相容的 u_j }
# ---------------------------------------------------------------------------

# (sbox_index_0based, known_input_positions, known_output_positions)
SBOX_INFO = [
    (4, (2, 3, 4),    (0, 1, 2, 3, 4)),
    (5, (3,),         (0, 1, 2, 3, 4)),
    (7, (2, 3),       (0, 1, 2, 3, 4)),
]


def _build_instant_match_tables():
    """
    对每个匹配 S-box 构建反向表 T_j：
      T_j[v_j] = { u_j | 存在完整 S-box 输入 x，使得 x 的已知输入位 = u_j
                           且 S(x) 的已知输出位 = v_j }
    返回 3 个 dict 组成的列表，索引 0/1/2 分别对应 S-box 5/6/8。
    """
    tables = []
    for sbox_idx, in_pos, out_pos in SBOX_INFO:
        reverse = {}
        for x in range(1 << 5):
            y = S2[x]
            xb = int_to_bits(x, 5)
            yb = int_to_bits(y, 5)
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
    标准即时匹配算法（对应论文图 2）。

    参数:
      Lf : dict，键 = u_tuple，值 = list of (fwd_guess, f_state0)
      Lb : list of (v_tuple, bwd_guess, b_state0)

    返回:
      candidates : set of (fwd_guess, bwd_guess)
    """
    T5, T6, T8 = INSTANT_MATCH_TABLES
    candidates = set()
    for v_tuple, bwd_guess, _ in Lb:
        v5, v6, v8 = v_tuple
        U5 = T5.get(v5, set())
        U6 = T6.get(v6, set())
        U8 = T8.get(v8, set())
        if not U5 or not U6 or not U8:
            continue
        for u5 in U5:
            for u6 in U6:
                for u8 in U8:
                    u_tuple = (u5, u6, u8)
                    if u_tuple in Lf:
                        for fwd_guess, _ in Lf[u_tuple]:
                            candidates.add((fwd_guess, bwd_guess))
    return candidates


# ---------------------------------------------------------------------------
# S-box 匹配（用于多对明密文验证阶段）
# ---------------------------------------------------------------------------

def _build_feasible_table():
    """构建 feasible[in_known] = set of out_known，用于 sbox_match 验证。"""
    table = []
    for sbox_idx, in_pos, out_pos in SBOX_INFO:
        unk_in = [b for b in range(5) if b not in in_pos]
        feasible = {}
        for in_known in itertools.product((0, 1), repeat=len(in_pos)):
            outs = set()
            for free in itertools.product((0, 1), repeat=len(unk_in)):
                full_in = [0] * 5
                for idx, b in enumerate(in_pos):
                    full_in[b] = in_known[idx]
                for idx, b in enumerate(unk_in):
                    full_in[b] = free[idx]
                in_val = bits_to_int(full_in)
                out_val = S2[in_val]
                ob = int_to_bits(out_val, 5)
                outs.add(tuple(ob[b] for b in out_pos))
            feasible[in_known] = outs
        table.append((sbox_idx, in_pos, out_pos, feasible))
    return table


FEASIBLE = _build_feasible_table()


def sbox_match(fwd_state, bwd_state):
    """检查 3 个匹配 S-box 上前向输入 u 与后向输出 v 是否满足 S-box 约束。"""
    fb = int_to_bits(fwd_state, 64)
    bb = int_to_bits(bwd_state, 64)
    for sbox_idx, in_pos, out_pos, feasible in FEASIBLE:
        in_known = tuple(fb[sbox_idx * 5 + p] for p in in_pos)
        out_known = tuple(bb[sbox_idx * 5 + p] for p in out_pos)
        if out_known not in feasible[in_known]:
            return False
    return True


# ---------------------------------------------------------------------------
# 单 chunk 攻击（严格按照 SITM 攻击模型）
# ---------------------------------------------------------------------------

def attack_chunk(chunk_idx, true_chunk_value, num_pairs=5, seed=0):
    """
    对单个 16-bit chunk 执行 Sieve-in-the-Middle 攻击。

    流程（对应论文攻击模型）：
      1. 将该 chunk 的 16 bits 按 {前向独有, 后向独有, 共享} 分割。
      2. 外层枚举共享 bits。
      3. 内层 A：遍历前向独有 bits，对每个猜测计算前向匹配点状态，
         提取向量 u，构建字典 L_f（u → [(fwd_guess, f_state0), ...]）。
      4. 内层 B：遍历后向独有 bits，对每个猜测计算后向匹配点状态，
         提取向量 v，构建列表 L_b（[(v, bwd_guess, b_state0), ...]）。
      5. 即时匹配（Instant Matching）合并 L_f 与 L_b，得到候选 (fwd, bwd)。
      6. 用多对明密文对候选做 sbox_match 验证， survivors 即为恢复出的密钥。

    优化：
      - key_schedule 结果缓存，避免重复计算。
      - L_f / L_b 构建时同时缓存第一对明密文的匹配点状态，
        验证阶段对第一对直接复用，减少约 1/num_pairs 的计算量。
      - 退化情况（只有前向/后向/共享一种类型）退化为直接枚举，
        避免无意义的空 L_f / L_b 构建。
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

    chunk_start = 16 * chunk_idx
    fwd_bits = []     # 该 chunk 内属于 BWD_UNK 的局部 bit 位置（前向独有）
    bwd_bits = []     # 该 chunk 内属于 FWD_UNK 的局部 bit 位置（后向独有）
    shared_bits = []  # 共享 bit 的局部位置
    for local_pos in range(16):
        global_pos = chunk_start + local_pos
        if global_pos in BWD_UNK:
            fwd_bits.append(local_pos)
        elif global_pos in FWD_UNK:
            bwd_bits.append(local_pos)
        else:
            shared_bits.append(local_pos)

    # 从真值 chunk_value 中提取前向/后向/共享三部分的值
    def extract_part(chunk_val, positions, width):
        if not positions:
            return 0
        return sum(((chunk_val >> (15 - pos)) & 1) << (width - 1 - i)
                   for i, pos in enumerate(positions))

    true_fwd = extract_part(true_chunk_value, fwd_bits, fwd_count)
    true_bwd = extract_part(true_chunk_value, bwd_bits, bwd_count)
    true_shared = extract_part(true_chunk_value, shared_bits, shared_count)

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

    # key_schedule 缓存（避免同一 mk 的重复密钥调度）
    _rks_cache = {}
    def get_rks(mk):
        if mk not in _rks_cache:
            _rks_cache[mk] = key_schedule(mk)
        return _rks_cache[mk]

    survivors = []
    t0 = time.time()

    # ======== 退化情况 1：全共享 chunk（前向=后向=0）========
    if fwd_count == 0 and bwd_count == 0:
        for guess in range(1 << 16):
            mk = set_chunk(chunk_idx, guess)
            rks = get_rks(mk)
            ok = True
            for pt, ct in pairs:
                f = forward_to_match(pt, mk, rks)
                b = backward_to_match(ct, mk, rks)
                if not sbox_match(f, b):
                    ok = False
                    break
            if ok:
                survivors.append(guess)
        elapsed = time.time() - t0
        return survivors, elapsed

    # ======== 退化情况 2：仅有后向独有（前向=0，后向>0）========
    # 此时 L_f 只有一个占位条目，即时匹配无任何筛选效果。
    # 为避免无意义的空 L_f 构建与冗余即时匹配，直接枚举所有 (shared, bwd) 组合。
    if fwd_count == 0 and bwd_count > 0:
        for shared_guess in range(1 << shared_count):
            for bwd_guess in range(1 << bwd_count):
                chunk_val = compose_chunk(shared_guess, 0, bwd_guess)
                mk = set_chunk(chunk_idx, chunk_val)
                rks = get_rks(mk)
                ok = True
                for pt, ct in pairs:
                    f = forward_to_match(pt, mk, rks)
                    b = backward_to_match(ct, mk, rks)
                    if not sbox_match(f, b):
                        ok = False
                        break
                if ok:
                    survivors.append(chunk_val)
        elapsed = time.time() - t0
        return survivors, elapsed

    # ======== 退化情况 3：仅有前向独有（后向=0，前向>0）========
    if bwd_count == 0 and fwd_count > 0:
        for shared_guess in range(1 << shared_count):
            for fwd_guess in range(1 << fwd_count):
                chunk_val = compose_chunk(shared_guess, fwd_guess, 0)
                mk = set_chunk(chunk_idx, chunk_val)
                rks = get_rks(mk)
                ok = True
                for pt, ct in pairs:
                    f = forward_to_match(pt, mk, rks)
                    b = backward_to_match(ct, mk, rks)
                    if not sbox_match(f, b):
                        ok = False
                        break
                if ok:
                    survivors.append(chunk_val)
        elapsed = time.time() - t0
        return survivors, elapsed

    # ======== 标准 SITM 流程（混合型 chunk：前向>0 且 后向>0）========
    for shared_guess in range(1 << shared_count):
        # ---- 构建 L_f（前向列表，同时缓存第一对的匹配点状态 f_state0）----
        Lf = {}  # u_tuple → list of (fwd_guess, f_state0)
        for fwd_guess in range(1 << fwd_count):
            chunk_val = compose_chunk(shared_guess, fwd_guess, true_bwd)
            mk = set_chunk(chunk_idx, chunk_val)
            rks = get_rks(mk)
            f_state0 = forward_to_match(pt0, mk, rks)
            u = extract_u(f_state0)
            Lf.setdefault(u, []).append((fwd_guess, f_state0))

        # ---- 构建 L_b（后向列表，同时缓存第一对的匹配点状态 b_state0）----
        Lb = []  # list of (v_tuple, bwd_guess, b_state0)
        for bwd_guess in range(1 << bwd_count):
            chunk_val = compose_chunk(shared_guess, true_fwd, bwd_guess)
            mk = set_chunk(chunk_idx, chunk_val)
            rks = get_rks(mk)
            b_state0 = backward_to_match(ct0, mk, rks)
            v = extract_v(b_state0)
            Lb.append((v, bwd_guess, b_state0))

        # ---- 即时匹配合并（Instant Matching，对应论文图 2）----
        candidate_set = instant_match(Lf, Lb)

        # ---- 多对明密文验证（第一对复用预计算状态）----
        for fwd_guess, bwd_guess in candidate_set:
            chunk_val = compose_chunk(shared_guess, fwd_guess, bwd_guess)
            mk = set_chunk(chunk_idx, chunk_val)
            rks = get_rks(mk)
            ok = True

            # 第一对 (pt0, ct0)：复用 L_f / L_b 中已缓存的 f_state0 / b_state0
            f_state0 = None
            for gf, fs in Lf.get(extract_u(forward_to_match(pt0, mk, rks)), []):
                if gf == fwd_guess:
                    f_state0 = fs
                    break
            if f_state0 is None:
                f_state0 = forward_to_match(pt0, mk, rks)

            b_state0 = None
            for _, gb, bs in Lb:
                if gb == bwd_guess:
                    b_state0 = bs
                    break
            if b_state0 is None:
                b_state0 = backward_to_match(ct0, mk, rks)

            if not sbox_match(f_state0, b_state0):
                ok = False
            else:
                # 其余对：重新计算 forward / backward
                for pt, ct in pairs[1:]:
                    f = forward_to_match(pt, mk, rks)
                    b = backward_to_match(ct, mk, rks)
                    if not sbox_match(f, b):
                        ok = False
                        break

            if ok:
                survivors.append(chunk_val)

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
      3) sbox_match 在真值密钥下必须通过
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
        ok_match = sbox_match(f_in, b_out)
        ok_all = ok_rt and ok_meet and ok_match
        if not ok_all:
            fail += 1
            print(f"  第 {t + 1:>2} 次：往返={ok_rt} 中点一致={ok_meet} S盒匹配={ok_match}")
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

    rng = random.Random(20260419)
    true_vals = [rng.getrandbits(16) for _ in range(8)]

    print("ILLcipher-64 Sieve-in-the-Middle 验证 (Nr=10, n=64, k=128)")
    print("匹配点：第 5 轮 SubCell，匹配 S-box {5,6,8}，即时匹配 (Instant Matching)")
    print("=" * 90)
    print(f"{'分块':>4} {'类型':>6} {'前向bit':>8} {'后向bit':>8} {'共享bit':>8} "
          f"{'真值':>8} {'候选数':>8} {'命中':>5} {'耗时(s)':>8}  恢复出的候选")
    print("-" * 90)

    all_ok = True
    for j in range(8):
        ctype = chunk_type(j)
        fwd_b, bwd_b, sh_b = chunk_bit_breakdown(j)
        num_pairs = 5
        survivors, elapsed = attack_chunk(j, true_vals[j], num_pairs=num_pairs)
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