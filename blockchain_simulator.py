"""
blockchain_simulator.py
比特币区块链仿真框架 —— 事件驱动核心模块
复现自：基于基因算法的比特币区块链难度调整算法研究与改进（张俊林，2022102899）
"""

import numpy as np


# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────
TARGET_BLOCK_TIME = 600        # 目标出块时间 600 秒
BASE_HASHRATE     = 8e19       # 基准哈希率 H₀ (H/s)
INITIAL_DIFF      = BASE_HASHRATE * TARGET_BLOCK_TIME   # D₀ = H₀ × T_target
BITCOIN_INTERVAL  = 2016       # 比特币原始调整间隔（块数）


# ─────────────────────────────────────────────
# 哈希率场景生成
# ─────────────────────────────────────────────
class HashRateScenario:
    """五种哈希率场景（论文表4-2）"""

    @staticmethod
    def constant(n_blocks: int) -> np.ndarray:
        """场景1：恒定哈希率"""
        return np.full(n_blocks, BASE_HASHRATE)

    @staticmethod
    def surge(n_blocks: int, surge_block: int = 1500) -> np.ndarray:
        """场景2：突发激增（第surge_block块后 × 7.5）"""
        h = np.full(n_blocks, BASE_HASHRATE)
        h[surge_block:] *= 7.5
        return h

    @staticmethod
    def staircase(n_blocks: int, step: int = 1500, factor: float = 1.6) -> np.ndarray:
        """场景3：阶梯式增长（每step块 × factor）"""
        h = np.empty(n_blocks)
        cur = BASE_HASHRATE
        for i in range(n_blocks):
            if i > 0 and i % step == 0:
                cur *= factor
            h[i] = cur
        return h

    @staticmethod
    def coin_hopping(n_blocks: int, interval: int = BITCOIN_INTERVAL,
                     attack_frac: float = 0.2) -> np.ndarray:
        """
        场景4：Coin-Hopping攻击
        每个调整周期的前(1-attack_frac)比例使用正常算力，
        后attack_frac比例撤出至0.4×H₀
        """
        h = np.empty(n_blocks)
        cutoff = int(interval * (1 - attack_frac))
        for i in range(n_blocks):
            pos_in_cycle = i % interval
            h[i] = BASE_HASHRATE if pos_in_cycle < cutoff else BASE_HASHRATE * 0.4
        return h

    @staticmethod
    def sinusoidal(n_blocks: int, period: int = 3000, amp: float = 0.4) -> np.ndarray:
        """场景5：正弦波动  H₀ × (1 + amp × sin(2π×t/period))"""
        t = np.arange(n_blocks)
        return BASE_HASHRATE * (1 + amp * np.sin(2 * np.pi * t / period))


# ─────────────────────────────────────────────
# Bitcoin 原始 DAA
# ─────────────────────────────────────────────
class BitcoinDAA:
    """
    比特币中本聪难度调整算法
    每 retarget_interval 块调整一次，调整倍数限制在 [1/4, 4]
    公式：D_new = D_old × (T_actual / T_target)
    """

    def __init__(self, retarget_interval: int = BITCOIN_INTERVAL,
                 block_interval: float = TARGET_BLOCK_TIME):
        self.retarget_interval = int(retarget_interval)
        self.block_interval    = block_interval        # T_target
        self.difficulty        = INITIAL_DIFF
        self._window_times: list[float] = []

    def record_block(self, block_time: float):
        self._window_times.append(block_time)

    def should_adjust(self, block_idx: int) -> bool:
        return (block_idx + 1) % self.retarget_interval == 0

    def adjust(self):
        if not self._window_times:
            return
        actual   = sum(self._window_times)
        target   = self.retarget_interval * self.block_interval
        factor   = max(0.25, min(4.0, actual / target))
        self.difficulty = self.difficulty * factor
        self._window_times.clear()

    def reset(self, difficulty=None):
        self.difficulty = difficulty if difficulty is not None else INITIAL_DIFF
        self._window_times.clear()


# ─────────────────────────────────────────────
# 出块时间采样
# ─────────────────────────────────────────────
def sample_block_time(hashrate: float, difficulty: float, rng: np.random.Generator) -> float:
    """t_block ~ Exponential(mean = D/H)"""
    mean = difficulty / hashrate
    return rng.exponential(mean)


# ─────────────────────────────────────────────
# 基础仿真器（仅 Bitcoin DAA）
# ─────────────────────────────────────────────
def simulate_bitcoin_daa(hashrate_seq: np.ndarray,
                          retarget_interval: int = BITCOIN_INTERVAL,
                          block_interval: float  = TARGET_BLOCK_TIME,
                          seed: int = 42) -> dict:
    """
    在给定哈希率序列上运行 Bitcoin DAA 仿真。
    返回每块的出块时间列表和难度列表。
    """
    rng  = np.random.default_rng(seed)
    daa  = BitcoinDAA(retarget_interval, block_interval)
    n    = len(hashrate_seq)

    block_times   = np.empty(n)
    difficulties  = np.empty(n)

    for i in range(n):
        bt = sample_block_time(hashrate_seq[i], daa.difficulty, rng)
        daa.record_block(bt)
        block_times[i]  = bt
        difficulties[i] = daa.difficulty
        if daa.should_adjust(i):
            daa.adjust()

    return {
        "block_times":  block_times,
        "difficulties": difficulties,
    }
