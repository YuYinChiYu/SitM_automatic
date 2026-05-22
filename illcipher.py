"""
ILLcipher-64
Block size: 64 bits, Key size: 128 bits, Rounds: 10
Based on: "The ILLcipher family of low-latency block ciphers for IIoT"
"""

import math

# ============================================================
# S-box definitions
# ============================================================

# 4-bit S-box S1
S1 = [0x9, 0x8, 0xA, 0xC, 0xB, 0x0, 0xE, 0x5,
      0x1, 0x2, 0x3, 0x6, 0x7, 0x4, 0xF, 0xD]

# Inverse S1
S1_INV = [0] * 16
for i, v in enumerate(S1):
    S1_INV[v] = i

# 5-bit S-box S2 (32 entries, 0x00-0x1F)
S2 = [0x01, 0x00, 0x19, 0x1A, 0x11, 0x1D, 0x15, 0x1B,
      0x14, 0x05, 0x04, 0x17, 0x0E, 0x12, 0x02, 0x1C,
      0x0F, 0x08, 0x06, 0x03, 0x0D, 0x07, 0x18, 0x10,
      0x1E, 0x09, 0x1F, 0x0A, 0x16, 0x0C, 0x0B, 0x13]

# Inverse S2
S2_INV = [0] * 32
for i, v in enumerate(S2):
    S2_INV[v] = i

# ============================================================
# Permutation tables for ILLcipher-64
# ============================================================

# PermuCell1 (PT1-64) for shell rounds - V_out[P(i)] = V_in[i]
PT1_64 = [
     0,  1,  2,  3, 28, 29, 30, 31,
    40, 41, 42, 43, 52, 53, 54, 55,
    16, 17, 18, 19, 44, 45, 46, 47,
    56, 57, 58, 59,  4,  5,  6,  7,
    32, 33, 34, 35, 60, 61, 62, 63,
     8,  9, 10, 11, 20, 21, 22, 23,
    48, 49, 50, 51, 12, 13, 14, 15,
    24, 25, 26, 27, 36, 37, 38, 39,
]

# PermuCell2 (PT2-64) for core rounds - V_out[P(i)] = V_in[i]
PT2_64 = [
    25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 20, 21, 22, 23,
    24, 10, 11, 12, 13, 14,  0,  1,
     2,  3,  4,  5,  6,  7,  8,  9,
    15, 16, 17, 18, 19, 50, 51, 52,
    53, 54, 60, 61, 62, 63, 55, 56,
    57, 58, 59, 45, 46, 47, 48, 49,
]

# Inverse permutations
PT1_64_INV = [0] * 64
for i in range(64):
    PT1_64_INV[PT1_64[i]] = i

PT2_64_INV = [0] * 64
for i in range(64):
    PT2_64_INV[PT2_64[i]] = i

# ============================================================
# Round constants RC_i (64-bit hex)
# ============================================================
ROUND_CONSTANTS = [
    0x0000000000000000,  # RC1
    0x13198A2E03707344,  # RC2
    0xA4093822299F31D0,  # RC3
    0x082EFA98EC4E6C89,  # RC4
    0x452821E638D01377,  # RC5
    0x85840851F1AC43AA,  # RC6
    0xC882D32F25323C54,  # RC7
    0x64A51195E0E3610D,  # RC8
    0xD3B5A399CA0C2399,  # RC9
    0xC0AC29B7C97C50DD,  # RC10
]

# ============================================================
# Bit manipulation helpers
# ============================================================

def int_to_bits(val, n):
    """Convert integer to list of n bits (MSB first)."""
    return [(val >> (n - 1 - i)) & 1 for i in range(n)]

def bits_to_int(bits):
    """Convert list of bits (MSB first) to integer."""
    val = 0
    for b in bits:
        val = (val << 1) | b
    return val

def rotate_left(val, shift, width=64):
    """Left cyclic shift of 'val' by 'shift' positions within 'width' bits."""
    shift = shift % width
    mask = (1 << width) - 1
    return ((val << shift) | (val >> (width - shift))) & mask

def rotate_right_16(val, shift):
    """Right cyclic shift of 16-bit value."""
    shift = shift % 16
    return ((val >> shift) | (val << (16 - shift))) & 0xFFFF

# ============================================================
# SubCell: Apply S-boxes to 64-bit state
# ============================================================

def subcell(state):
    """Apply SubCell: 12 × S2(5-bit) + 1 × S1(4-bit) to 64-bit state."""
    bits = int_to_bits(state, 64)
    out_bits = [0] * 64
    # 12 instances of S2 (5-bit each, bits 0-59)
    for i in range(12):
        start = i * 5
        inp = bits_to_int(bits[start:start + 5])
        outp = S2[inp]
        ob = int_to_bits(outp, 5)
        out_bits[start:start + 5] = ob
    # 1 instance of S1 (4-bit, bits 60-63)
    inp = bits_to_int(bits[60:64])
    outp = S1[inp]
    ob = int_to_bits(outp, 4)
    out_bits[60:64] = ob
    return bits_to_int(out_bits)

def subcell_inv(state):
    """Inverse SubCell."""
    bits = int_to_bits(state, 64)
    out_bits = [0] * 64
    for i in range(12):
        start = i * 5
        inp = bits_to_int(bits[start:start + 5])
        outp = S2_INV[inp]
        ob = int_to_bits(outp, 5)
        out_bits[start:start + 5] = ob
    inp = bits_to_int(bits[60:64])
    outp = S1_INV[inp]
    ob = int_to_bits(outp, 4)
    out_bits[60:64] = ob
    return bits_to_int(out_bits)

# ============================================================
# DiffMatrix: Linear diffusion layer using circulant matrices
# ============================================================

def diffmatrix(state):
    """
    DiffMatrix for ILLcipher-64.
    State is split into 4 × 16-bit words: x0=state[0:16], x1=state[16:32],
    x2=state[32:48], x3=state[48:64].
    
    y0 = x0 ⊕ (x0 ⋙ 4) ⊕ (x0 ⋙ 6) ⊕ (x0 ⋙ 7) ⊕ (x0 ⋙ 13)
    y1 = x1 ⊕ (x1 ⋙ 2) ⊕ (x1 ⋙ 5) ⊕ (x1 ⋙ 7) ⊕ (x1 ⋙ 14)
    y2 = x2 ⊕ (x2 ⋙ 1) ⊕ (x2 ⋙ 6) ⊕ (x2 ⋙ 7) ⊕ (x2 ⋙ 10)
    y3 = x3 ⊕ (x3 ⋙ 4) ⊕ (x3 ⋙ 6) ⊕ (x3 ⋙ 7) ⊕ (x3 ⋙ 13)
    """
    x0 = (state >> 48) & 0xFFFF
    x1 = (state >> 32) & 0xFFFF
    x2 = (state >> 16) & 0xFFFF
    x3 = state & 0xFFFF

    y0 = x0 ^ rotate_right_16(x0, 4) ^ rotate_right_16(x0, 6) ^ rotate_right_16(x0, 7) ^ rotate_right_16(x0, 13)
    y1 = x1 ^ rotate_right_16(x1, 2) ^ rotate_right_16(x1, 5) ^ rotate_right_16(x1, 7) ^ rotate_right_16(x1, 14)
    y2 = x2 ^ rotate_right_16(x2, 1) ^ rotate_right_16(x2, 6) ^ rotate_right_16(x2, 7) ^ rotate_right_16(x2, 10)
    y3 = x3 ^ rotate_right_16(x3, 4) ^ rotate_right_16(x3, 6) ^ rotate_right_16(x3, 7) ^ rotate_right_16(x3, 13)

    return (y0 << 48) | (y1 << 32) | (y2 << 16) | y3

def _build_circulant_matrix_16(shifts):
    """Build a 16×16 binary matrix from cyclic right shifts for GF(2).
    Uses MSB-first convention to match _mat_vec_mul_gf2.
    y[i] = x[i] XOR x[(i-s1)%16] XOR ... (MSB-first indexing)."""
    M = [[0]*16 for _ in range(16)]
    for r in range(16):
        M[r][r] = 1  # identity term (x itself)
        for s in shifts:
            M[r][(r - s) % 16] = M[r][(r - s) % 16] ^ 1
    return M

def _mat_inv_gf2(M, n=16):
    """Invert an n×n matrix over GF(2) using Gauss-Jordan elimination."""
    # Augmented matrix [M | I]
    aug = [row[:] + [1 if j == i else 0 for j in range(n)] for i, row in enumerate(M)]
    for col in range(n):
        # Find pivot
        pivot = None
        for row in range(col, n):
            if aug[row][col] == 1:
                pivot = row
                break
        if pivot is None:
            raise ValueError("Matrix is not invertible")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(n):
            if row != col and aug[row][col] == 1:
                for j in range(2 * n):
                    aug[row][j] ^= aug[col][j]
    return [row[n:] for row in aug]

def _mat_vec_mul_gf2(M, x, n=16):
    """Multiply n×n GF(2) matrix by n-bit vector (integer)."""
    result = 0
    for r in range(n):
        bit = 0
        for c in range(n):
            bit ^= M[r][c] & ((x >> (n - 1 - c)) & 1)
        result = (result << 1) | bit
    return result

# Build inverse diffusion matrices
_M0 = _build_circulant_matrix_16([4, 6, 7, 13])
_M1 = _build_circulant_matrix_16([2, 5, 7, 14])
_M2 = _build_circulant_matrix_16([1, 6, 7, 10])
_M3 = _build_circulant_matrix_16([4, 6, 7, 13])

_M0_inv = _mat_inv_gf2(_M0)
_M1_inv = _mat_inv_gf2(_M1)
_M2_inv = _mat_inv_gf2(_M2)
_M3_inv = _mat_inv_gf2(_M3)

def diffmatrix_inv(state):
    """Inverse DiffMatrix for ILLcipher-64."""
    x0 = (state >> 48) & 0xFFFF
    x1 = (state >> 32) & 0xFFFF
    x2 = (state >> 16) & 0xFFFF
    x3 = state & 0xFFFF

    y0 = _mat_vec_mul_gf2(_M0_inv, x0)
    y1 = _mat_vec_mul_gf2(_M1_inv, x1)
    y2 = _mat_vec_mul_gf2(_M2_inv, x2)
    y3 = _mat_vec_mul_gf2(_M3_inv, x3)

    return (y0 << 48) | (y1 << 32) | (y2 << 16) | y3

# ============================================================
# PermuCell: Bit permutation
# ============================================================

def permucell1(state):
    """Apply PermuCell1 (shell round permutation). V_out[P(i)] = V_in[i]."""
    bits_in = int_to_bits(state, 64)
    bits_out = [0] * 64
    for i in range(64):
        bits_out[PT1_64[i]] = bits_in[i]
    return bits_to_int(bits_out)

def permucell1_inv(state):
    """Inverse PermuCell1."""
    bits_in = int_to_bits(state, 64)
    bits_out = [0] * 64
    for i in range(64):
        bits_out[PT1_64_INV[i]] = bits_in[i]
    return bits_to_int(bits_out)

def permucell2(state):
    """Apply PermuCell2 (core round permutation). V_out[P(i)] = V_in[i]."""
    bits_in = int_to_bits(state, 64)
    bits_out = [0] * 64
    for i in range(64):
        bits_out[PT2_64[i]] = bits_in[i]
    return bits_to_int(bits_out)

def permucell2_inv(state):
    """Inverse PermuCell2."""
    bits_in = int_to_bits(state, 64)
    bits_out = [0] * 64
    for i in range(64):
        bits_out[PT2_64_INV[i]] = bits_in[i]
    return bits_to_int(bits_out)

# ============================================================
# AddRoundCon and AddRoundKey
# ============================================================

def add_round_con(state, round_idx):
    """XOR state with round constant. round_idx is 0-based (0..9)."""
    return state ^ ROUND_CONSTANTS[round_idx]

def add_round_key(state, round_key):
    """XOR state with round key (64 bits)."""
    return state ^ round_key

# ============================================================
# Key Schedule
# ============================================================

def key_schedule(master_key):
    """
    Generate round keys for ILLcipher-64.
    master_key: 128-bit integer (K = K0 || K1, K0 = high 64 bits)
    Returns: list of 10 round keys (64-bit each)
    
    Steps:
    1. K0[0:3] = S1(K0[0:3]), K1[0:3] = S1(K1[0:3])
    2. For round i (1-based):
       Shell (i<3 or i>Nr-2): RK_i = K0 <<< ceil(i*5/2)
       Core (2<i<Nr-1):       RK_i = K1 <<< floor(i*5/2)
    """
    nr = 10
    k0 = (master_key >> 64) & ((1 << 64) - 1)
    k1 = master_key & ((1 << 64) - 1)

    # Apply S1 to the first 4 bits (MSB) of K0 and K1
    k0_top4 = (k0 >> 60) & 0xF
    k0 = (k0 & 0x0FFFFFFFFFFFFFFF) | (S1[k0_top4] << 60)

    k1_top4 = (k1 >> 60) & 0xF
    k1 = (k1 & 0x0FFFFFFFFFFFFFFF) | (S1[k1_top4] << 60)

    round_keys = []
    for i in range(1, nr + 1):
        if i < 3 or i > nr - 2:
            # Shell round: use K0
            shift = math.ceil(i * 5 / 2)
            rk = rotate_left(k0, shift, 64)
        else:
            # Core round: use K1
            shift = math.floor(i * 5 / 2)
            rk = rotate_left(k1, shift, 64)
        round_keys.append(rk)
    return round_keys

# ============================================================
# ILLcipher-64 Encryption
# ============================================================

def encrypt(plaintext, master_key):
    """
    Encrypt a 64-bit plaintext with 128-bit master_key.
    
    Algorithm 1:
    T = P ⊕ K0
    for i = 1 to Nr:
        if i < 3 or i > Nr - 2:  (Shell: rounds 1,2,9,10)
            SubCell -> DiffMatrix -> AddRoundCon -> AddRoundKey -> PermuCell1
        else:  (Core: rounds 3-8)
            AddRoundKey -> AddRoundCon -> DiffMatrix -> SubCell -> PermuCell2
    C = T
    """
    nr = 10
    k0 = (master_key >> 64) & ((1 << 64) - 1)
    round_keys = key_schedule(master_key)

    t = plaintext ^ k0
    for i in range(1, nr + 1):
        ri = i - 1  # 0-based round index
        if i < 3 or i > nr - 2:
            # Shell round
            t = subcell(t)
            t = diffmatrix(t)
            t = add_round_con(t, ri)
            t = add_round_key(t, round_keys[ri])
            t = permucell1(t)
        else:
            # Core round
            t = add_round_key(t, round_keys[ri])
            t = add_round_con(t, ri)
            t = diffmatrix(t)
            t = subcell(t)
            t = permucell2(t)
    return t

# ============================================================
# ILLcipher-64 Decryption
# ============================================================

def decrypt(ciphertext, master_key):
    """
    Decrypt a 64-bit ciphertext with 128-bit master_key.
    Inverse of encrypt: process rounds in reverse order with inverse operations.
    """
    nr = 10
    k0 = (master_key >> 64) & ((1 << 64) - 1)
    round_keys = key_schedule(master_key)

    t = ciphertext
    for i in range(nr, 0, -1):
        ri = i - 1  # 0-based round index
        if i < 3 or i > nr - 2:
            # Inverse shell round
            t = permucell1_inv(t)
            t = add_round_key(t, round_keys[ri])
            t = add_round_con(t, ri)
            t = diffmatrix_inv(t)
            t = subcell_inv(t)
        else:
            # Inverse core round
            t = permucell2_inv(t)
            t = subcell_inv(t)
            t = diffmatrix_inv(t)
            t = add_round_con(t, ri)
            t = add_round_key(t, round_keys[ri])
    t = t ^ k0
    return t

# ============================================================
# Partial encryption/decryption for SITM attack
# ============================================================

def encrypt_rounds(state, master_key, start_round, end_round):
    """
    Encrypt from start_round to end_round (1-based, inclusive).
    The initial K0 whitening is NOT applied here.
    Use this for partial forward/backward computation in SITM.
    """
    nr = 10
    round_keys = key_schedule(master_key)
    t = state
    for i in range(start_round, end_round + 1):
        ri = i - 1
        if i < 3 or i > nr - 2:
            t = subcell(t)
            t = diffmatrix(t)
            t = add_round_con(t, ri)
            t = add_round_key(t, round_keys[ri])
            t = permucell1(t)
        else:
            t = add_round_key(t, round_keys[ri])
            t = add_round_con(t, ri)
            t = diffmatrix(t)
            t = subcell(t)
            t = permucell2(t)
    return t

def decrypt_rounds(state, master_key, start_round, end_round):
    """
    Decrypt from end_round back to start_round (1-based, inclusive).
    The final K0 whitening is NOT applied here.
    """
    nr = 10
    round_keys = key_schedule(master_key)
    t = state
    for i in range(end_round, start_round - 1, -1):
        ri = i - 1
        if i < 3 or i > nr - 2:
            t = permucell1_inv(t)
            t = add_round_key(t, round_keys[ri])
            t = add_round_con(t, ri)
            t = diffmatrix_inv(t)
            t = subcell_inv(t)
        else:
            t = permucell2_inv(t)
            t = subcell_inv(t)
            t = diffmatrix_inv(t)
            t = add_round_con(t, ri)
            t = add_round_key(t, round_keys[ri])
    return t

def forward_to_round6_input(plaintext, master_key):
    """
    Compute forward from plaintext to the input of round 6's SubCell.
    Round 6 is a core round: AddRoundKey -> AddRoundCon -> DiffMatrix -> SubCell -> PermuCell2
    The "input of SubCell" is after AddRoundKey, AddRoundCon, and DiffMatrix.
    
    Steps: P ⊕ K0 -> round1 -> round2 -> round3 -> round4 -> round5 ->
           then in round6: AddRoundKey -> AddRoundCon -> DiffMatrix -> [here is SubCell input]
    """
    nr = 10
    k0 = (master_key >> 64) & ((1 << 64) - 1)
    round_keys = key_schedule(master_key)

    t = plaintext ^ k0
    # Rounds 1-5
    for i in range(1, 6):
        ri = i - 1
        if i < 3 or i > nr - 2:
            t = subcell(t)
            t = diffmatrix(t)
            t = add_round_con(t, ri)
            t = add_round_key(t, round_keys[ri])
            t = permucell1(t)
        else:
            t = add_round_key(t, round_keys[ri])
            t = add_round_con(t, ri)
            t = diffmatrix(t)
            t = subcell(t)
            t = permucell2(t)
    # Round 6 partial: AddRoundKey -> AddRoundCon -> DiffMatrix
    ri = 5  # round 6, 0-based
    t = add_round_key(t, round_keys[ri])
    t = add_round_con(t, ri)
    t = diffmatrix(t)
    return t  # This is the input to SubCell in round 6

def backward_to_round6_output(ciphertext, master_key):
    """
    Compute backward from ciphertext to the output of round 6's SubCell.
    Round 6 is a core round: AddRoundKey -> AddRoundCon -> DiffMatrix -> SubCell -> PermuCell2
    The "output of SubCell" is before PermuCell2.
    
    Steps: C <- round10_inv <- ... <- round7_inv <- PermuCell2_inv in round6
    """
    nr = 10
    round_keys = key_schedule(master_key)

    t = ciphertext
    # Inverse rounds 10 down to 7
    for i in range(10, 6, -1):
        ri = i - 1
        if i < 3 or i > nr - 2:
            t = permucell1_inv(t)
            t = add_round_key(t, round_keys[ri])
            t = add_round_con(t, ri)
            t = diffmatrix_inv(t)
            t = subcell_inv(t)
        else:
            t = permucell2_inv(t)
            t = subcell_inv(t)
            t = diffmatrix_inv(t)
            t = add_round_con(t, ri)
            t = add_round_key(t, round_keys[ri])
    # Round 6 partial inverse: PermuCell2_inv only
    t = permucell2_inv(t)
    return t  # This is the output of SubCell in round 6


if __name__ == "__main__":
    # Quick test
    key = 0x0123456789ABCDEF_FEDCBA9876543210
    pt = 0xDEADBEEF12345678
    ct = encrypt(pt, key)
    pt2 = decrypt(ct, key)
    print(f"Plaintext:  {pt:#018x}")
    print(f"Ciphertext: {ct:#018x}")
    print(f"Decrypted:  {pt2:#018x}")
    print(f"Match: {pt == pt2}")
