"""
SITM 攻击 ILLcipher-64 - 辅助函数
"""

import numpy as np
import random
from illcipher import (
    encrypt, S1, S2,
    subcell, subcell_inv, diffmatrix, diffmatrix_inv,
    permucell1, permucell1_inv, permucell2, permucell2_inv,
    add_round_con, add_round_key, rotate_left,
)


# ============================================================
# 通用前向/后向计算到任意匹配点
# ============================================================

def _key_schedule_parts(k0, k1):
    """计算所有轮密钥 (与 illcipher.key_schedule 一致)"""
    k0_top4 = (k0 >> 60) & 0xF
    k0_s = (k0 & 0x0FFFFFFFFFFFFFFF) | (S1[k0_top4] << 60)
    k1_top4 = (k1 >> 60) & 0xF
    k1_s = (k1 & 0x0FFFFFFFFFFFFFFF) | (S1[k1_top4] << 60)
    shifts = [3, 5, 7, 10, 12, 15, 17, 20, 23, 25]
    sources = [k0_s, k0_s, k1_s, k1_s, k1_s, k1_s, k1_s, k1_s, k0_s, k0_s]
    return [rotate_left(src, sh, 64) for src, sh in zip(sources, shifts)]


def forward_to_subcell_input(plaintext, k0, k1, match_round):
    """
    前向计算 P → R_match 的 SubCell 输入。
    match_round: 3-8 (core rounds)
    """
    rks = _key_schedule_parts(k0, k1)
    t = plaintext ^ k0
    # R1, R2: Shell rounds
    for i in range(2):
        t = subcell(t); t = diffmatrix(t)
        t = add_round_con(t, i); t = add_round_key(t, rks[i])
        t = permucell1(t)
    # R3 to R_{match-1}: full Core rounds
    for i in range(2, match_round - 1):
        t = add_round_key(t, rks[i]); t = add_round_con(t, i)
        t = diffmatrix(t); t = subcell(t); t = permucell2(t)
    # R_match: partial Core round up to SubCell input
    i = match_round - 1
    t = add_round_key(t, rks[i]); t = add_round_con(t, i)
    t = diffmatrix(t)
    return t  # SubCell input


def backward_to_subcell_output(ciphertext, k0, k1, match_round):
    """
    后向计算 C → R_match 的 SubCell 输出。
    match_round: 3-8 (core rounds)
    """
    rks = _key_schedule_parts(k0, k1)
    t = ciphertext
    # R10, R9: Shell rounds (reverse)
    for i in [9, 8]:
        t = permucell1_inv(t); t = add_round_key(t, rks[i])
        t = add_round_con(t, i); t = diffmatrix_inv(t)
        t = subcell_inv(t)
    # R8 down to R_{match+1}: full Core rounds (reverse)
    for i in range(7, match_round - 1, -1):
        t = permucell2_inv(t); t = subcell_inv(t)
        t = diffmatrix_inv(t); t = add_round_con(t, i)
        t = add_round_key(t, rks[i])
    # R_match: partial reverse → SubCell output
    t = permucell2_inv(t)
    return t  # SubCell output


# ============================================================
# 影响矩阵计算
# ============================================================

def compute_influence_matrix(match_round, direction, num_samples=300):
    """
    计算 K1 bit 对匹配点状态的影响矩阵。
    
    返回: influence[k][s] (64×64 bool)
      influence[k][s] = True 表示翻转 K1[k] 会改变状态 bit s
    
    direction:
      'forward':  计算到 R_match SubCell 输入
      'backward': 计算到 R_match SubCell 输出
    """
    rng = random.Random(42)
    influence = np.zeros((64, 64), dtype=bool)
    
    for _ in range(num_samples):
        pt = rng.randint(0, (1 << 64) - 1)
        k0 = rng.randint(0, (1 << 64) - 1)
        k1 = rng.randint(0, (1 << 64) - 1)
        
        ct = encrypt(pt, (k0 << 64) | k1)
        if direction == 'forward':
            ref = forward_to_subcell_input(pt, k0, k1, match_round)
        else:
            # 正确方法: 密文 C 是固定的（攻击者观测值），只改变反向解密中的 K1
            ref = backward_to_subcell_output(ct, k0, k1, match_round)
        
        for b in range(64):
            k1_flip = k1 ^ (1 << (63 - b))
            if direction == 'forward':
                val = forward_to_subcell_input(pt, k0, k1_flip, match_round)
            else:
                # 固定密文 ct，仅翻转 K1[b]
                val = backward_to_subcell_output(ct, k0, k1_flip, match_round)
            
            diff = ref ^ val
            for s in range(64):
                if (diff >> (63 - s)) & 1:
                    influence[b][s] = True
    
    return influence


# ============================================================
# 辅助函数
# ============================================================

def known_state_bits(unknown_k1_bits, influence_matrix):
    """未知 K1 bit 不影响的状态比特 → 已确定"""
    uncertain = set()
    for b in unknown_k1_bits:
        for s in range(64):
            if influence_matrix[b][s]:
                uncertain.add(s)
    return set(range(64)) - uncertain


def known_bits_per_sbox(known_bits):
    """将已知状态比特集合转换为每个 S-box 的已知比特位列表"""
    per_sbox = []
    for i in range(12):
        start = i * 5
        known_in_sbox = [s - start for s in known_bits if start <= s < start + 5]
        per_sbox.append(sorted(known_in_sbox))
    known_in_sbox12 = [s - 60 for s in known_bits if 60 <= s < 64]
    per_sbox.append(sorted(known_in_sbox12))
    return per_sbox


def sieving_prob_for_sbox(sbox_lut, n_bits, input_known_positions, output_known_positions):
    """计算 S-box 在给定已知输入/输出比特下的筛选概率"""
    if not input_known_positions or not output_known_positions:
        return 1.0
    m = len(input_known_positions)
    p = len(output_known_positions)
    valid = set()
    for x in range(1 << n_bits):
        y = sbox_lut[x]
        u = 0
        for pos in input_known_positions:
            u = (u << 1) | ((x >> (n_bits - 1 - pos)) & 1)
        v = 0
        for pos in output_known_positions:
            v = (v << 1) | ((y >> (n_bits - 1 - pos)) & 1)
        valid.add((u, v))
    total = (1 << m) * (1 << p)
    return len(valid) / total

