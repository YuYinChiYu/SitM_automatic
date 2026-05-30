"""
SITM 攻击 ILLcipher-64

关键公式：
  κ = 64(K0) + shared_K1
  main_cost = 2^κ × (2^f + 2^b)
  verify_cost = 2^{128+log_π}
  total = main_cost + verify_cost
"""

import numpy as np
import math
from illcipher import S1, S2
from sitm_functions import (
    compute_influence_matrix,
    known_state_bits,
    known_bits_per_sbox,
    sieving_prob_for_sbox,
)


def compute_main_cost(f: object, b: object, s: object) -> object:
    """
    主搜索阶段复杂度 (log2)
    κ = 64 + s, 前向列表 2^f, 后向列表 2^b
    main = 2^κ × (2^f + 2^b)
    """
    kappa = 64 + s
    if f == 0 and b == 0:
        return kappa  # 2^κ
    elif f == 0:
        return kappa + b
    elif b == 0:
        return kappa + f
    else:
        return kappa + max(f, b) + math.log2(1 + 2 ** (min(f, b) - max(f, b)))


def compute_sieving(fwd_unknown, bwd_unknown, fwd_infl, bwd_infl):
    """计算给定 unknown bit 集合下的筛分概率 log2(π)"""
    fwd_known = known_state_bits(fwd_unknown, fwd_infl)
    bwd_known = known_state_bits(bwd_unknown, bwd_infl)

    fwd_per = known_bits_per_sbox(fwd_known)
    bwd_per = known_bits_per_sbox(bwd_known)

    log_pi = 0.0
    details = []
    for i in range(13):
        n = 5 if i < 12 else 4
        lut = S2 if i < 12 else S1
        inp, out = fwd_per[i], bwd_per[i]
        if inp and out:
            pi = sieving_prob_for_sbox(lut, n, inp, out)
            lp = math.log2(pi) if pi > 0 else float('-inf')
        else:
            pi, lp = 1.0, 0.0
        log_pi += lp
        details.append((i, inp, out, lp))
    return log_pi, len(fwd_known), len(bwd_known), details


def evaluate(fwd_only_set, bwd_only_set, fwd_infl, bwd_infl):
    """评估一组 (fwd_only, bwd_only) 分配"""
    f = len(fwd_only_set)
    b = len(bwd_only_set)
    s = 64 - f - b
    fwd_unknown = bwd_only_set   # 前向不猜这些 → 前向未知
    bwd_unknown = fwd_only_set   # 后向不猜这些 → 后向未知
    log_pi, fwd_kn, bwd_kn, details = compute_sieving(
        fwd_unknown, bwd_unknown, fwd_infl, bwd_infl)
    log_main = compute_main_cost(f, b, s)
    log_verify = 128 + log_pi
    log_total = max(log_main, log_verify) if log_main != log_verify else log_main + 1
    return {
        'f': f, 'b': b, 's': s,
        'kappa': 64 + s,
        'log_main': log_main,
        'log_verify': log_verify,
        'log_total': log_total,
        'log_pi': log_pi,
        'fwd_known': fwd_kn,
        'bwd_known': bwd_kn,
        'fwd_only': sorted(fwd_only_set),
        'bwd_only': sorted(bwd_only_set),
        'details': details,
    }


def find_full_diffusion_bits(infl):
    """找出完全扩散的 K1 bit (影响 64/64 个状态 bit)"""
    return set(b for b in range(64) if infl[b].sum() == 64)


def find_zero_influence_bits(infl):
    """找出零影响的 K1 bit"""
    return set(b for b in range(64) if infl[b].sum() == 0)


# ============================================================
# 固定 (f, b) 下的最优比特选择
# ============================================================
def optimal_for_fb(f_target, b_target, fwd_candidates, bwd_candidates,
                   fwd_infl, bwd_infl):
    """
    固定 f 和 b，找最优的 K1 bit 分配。
    使用贪心：交替添加 fwd_only 和 bwd_only bit，每步选最优。
    """
    fwd_only = set()
    bwd_only = set()

    # 交替添加
    for _ in range(f_target + b_target):
        best_bit = None
        best_dir = None
        best_score = float('inf')

        if len(fwd_only) < f_target:
            for b in sorted(fwd_candidates - fwd_only - bwd_only):
                new_f = fwd_only | {b}
                res = evaluate(new_f, bwd_only, fwd_infl, bwd_infl)
                if res['log_total'] < best_score:
                    best_score = res['log_total']
                    best_bit = b
                    best_dir = 'fwd'

        if len(bwd_only) < b_target:
            for b in sorted(bwd_candidates - fwd_only - bwd_only):
                new_b = bwd_only | {b}
                res = evaluate(fwd_only, new_b, fwd_infl, bwd_infl)
                if res['log_total'] < best_score:
                    best_score = res['log_total']
                    best_bit = b
                    best_dir = 'bwd'

        if best_bit is None:
            break
        if best_dir == 'fwd':
            fwd_only.add(best_bit)
        else:
            bwd_only.add(best_bit)

    return evaluate(fwd_only, bwd_only, fwd_infl, bwd_infl)


# ============================================================
# 主分析
# ============================================================
def analyze_match_point(match_round, num_samples=500):
    print(f"\n{'='*70}")
    print(f"  匹配点 R{match_round} SubCell")
    print(f"{'='*70}")

    fwd_k1_rounds = match_round - 2
    bwd_k1_rounds = 8 - match_round
    print(f"  前向 K1 轮数: {fwd_k1_rounds}  |  后向 K1 轮数: {bwd_k1_rounds}")

    # 计算影响矩阵
    fwd_infl = compute_influence_matrix(match_round, 'forward', num_samples)
    bwd_infl = compute_influence_matrix(match_round, 'backward', num_samples)

    # 统计
    fwd_counts = [int(fwd_infl[b].sum()) for b in range(64)]
    bwd_counts = [int(bwd_infl[b].sum()) for b in range(64)]

    fwd_full = find_full_diffusion_bits(fwd_infl)
    bwd_full = find_full_diffusion_bits(bwd_infl)

    print(f"\n  前向影响: avg={np.mean(fwd_counts):.1f}/64, "
          f"全扩散={len(fwd_full)}/64")
    print(f"  后向影响: avg={np.mean(bwd_counts):.1f}/64, "
          f"全扩散={len(bwd_full)}/64")

    # 约束分析
    # fwd_full 中的 bit 不能做 bwd_only（否则前向全部未知）
    # bwd_full 中的 bit 不能做 fwd_only（否则后向全部未知）
    both_full = fwd_full & bwd_full  # 必须 shared
    fwd_candidates = set(range(64)) - bwd_full  # 可以做 fwd_only
    bwd_candidates = set(range(64)) - fwd_full  # 可以做 bwd_only
    max_f = len(fwd_candidates)
    max_b = len(bwd_candidates)

    print(f"\n  约束:")
    print(f"    fwd_full (不可 bwd_only): {len(fwd_full)} bits → {sorted(fwd_full)}")
    print(f"    bwd_full (不可 fwd_only): {len(bwd_full)} bits → {sorted(bwd_full)}")
    print(f"    both_full (必须 shared):  {len(both_full)} bits → {sorted(both_full)}")
    print(f"    max fwd_only (f): {max_f}")
    print(f"    max bwd_only (b): {max_b}")

    if max_f == 0 and max_b == 0:
        print(f"\n  ✗ 所有 K1 bit 两方向都全扩散 → 必须全部 shared → 复杂度 2^128")
        return match_round, 128.0, None, fwd_infl, bwd_infl

    # ─── 扫描所有 (f, b) 值 ───
    print(f"\n  === (f,b) 复杂度扫描 ===")
    print(f"  {'f':>3s} {'b':>3s} {'s':>3s} | {'κ':>4s} {'main':>8s} {'π':>10s} "
          f"{'verify':>8s} {'total':>8s} | {'fwd_kn':>6s} {'bwd_kn':>6s}")
    print(f"  {'-'*72}")

    best_result = None
    # 扫描关键 (f, b) 值 预设的稀疏采样点：小值密集（0-8），大值稀疏（10, 12, 16...）。因为通常最优解出现在 f 和 b 较小的情况
    scan_values = sorted(set([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32]
                             + list(range(max_f + 1))))
    for f in scan_values:
        if f > max_f:
            continue
        for b in scan_values:
            if b > max_b:
                continue
            if f + b > 64:
                continue
            if f == 0 and b == 0:
                continue

            res = optimal_for_fb(f, b, fwd_candidates, bwd_candidates,
                                 fwd_infl, bwd_infl)
            if best_result is None or res['log_total'] < best_result['log_total']:
                best_result = res
                print(f"  {f:3d} {b:3d} {64-f-b:3d} | {64+64-f-b:4d} "
                      f"{res['log_main']:8.2f} {res['log_pi']:10.2f} "
                      f"{res['log_verify']:8.2f} {res['log_total']:8.2f} | "
                      f"{res['fwd_known']:6d} {res['bwd_known']:6d}  ← 新最优")
            elif f <= 12 and b <= 12:
                print(f"  {f:3d} {b:3d} {64-f-b:3d} | {64+64-f-b:4d} "
                      f"{res['log_main']:8.2f} {res['log_pi']:10.2f} "
                      f"{res['log_verify']:8.2f} {res['log_total']:8.2f} | "
                      f"{res['fwd_known']:6d} {res['bwd_known']:6d}")

    # ─── 最终结果 ───
    print(f"\n  {'─'*60}")
    print(f"  R{match_round} 最优结果:")
    print(f"    复杂度: 2^{best_result['log_total']:.2f}")
    print(f"    优势:   2^{128 - best_result['log_total']:.2f} "
          f"{'(有优势!)' if best_result['log_total'] < 127.5 else '(无显著优势)'}")
    print(f"    κ = {best_result['kappa']} (K0=64 + shared={best_result['s']})")
    print(f"    fwd_only = {best_result['f']}: {best_result['fwd_only']}")
    print(f"    bwd_only = {best_result['b']}: {best_result['bwd_only']}")
    print(f"    π = 2^{best_result['log_pi']:.2f}")
    print(f"    main = 2^{best_result['log_main']:.2f}, "
          f"verify = 2^{best_result['log_verify']:.2f}")
    print(f"    前向已知: {best_result['fwd_known']}/64, "
          f"后向已知: {best_result['bwd_known']}/64")

    # S-box 筛选详情
    print(f"    S-box 筛选:")
    for idx, inp, out, lp in best_result['details']:
        if lp < 0:
            print(f"      S{idx:2d}: input={inp}, output={out}, π=2^{lp:.2f}")

    return match_round, best_result['log_total'], best_result, fwd_infl, bwd_infl


def main():
    all_results = []
    for m in range(3, 9):
        mr, total, best, _, _ = analyze_match_point(m, num_samples=500)
        all_results.append((mr, total, best))

    # ─── 总结 ───
    print(f"\n\n{'='*70}")
    print("  总结: 各匹配点最优攻击复杂度")
    print(f"{'='*70}")
    print(f"  {'Rm':>3s} | {'total':>8s} | {'优势':>8s} | {'κ':>4s} | "
          f"{'f':>3s} {'b':>3s} {'s':>3s} | {'π':>10s} ")
    print(f"  {'─'*72}")

    for mr, total, best in all_results:
        if best:
            print(f"  R{mr:d}  | 2^{total:5.1f} | 2^{128-total:5.1f} | "
                  f"{best['kappa']:4d} | {best['f']:3d} {best['b']:3d} {best['s']:3d} | "
                  f"2^{best['log_pi']:7.2f} ")
        else:
            print(f"  R{mr:d}  | 2^128.0 | 2^  0.0 |  128 |   0   0  64 | "
                  f"      N/A")

    best_overall = min(all_results, key=lambda x: x[1])
    print(f"\n  全局最优: R{best_overall[0]}, 复杂度 2^{best_overall[1]:.2f}, "
          f"优势 2^{128 - best_overall[1]:.2f}")
    if best_overall[1] >= 127.5:
        print(f"\n  结论: 10 轮 ILLcipher-64 对 SITM 攻击安全（无显著优势）")
    else:
        print(f"\n  结论: 发现有效 SITM 攻击! 最优匹配点 R{best_overall[0]}")


if __name__ == '__main__':
    main()
