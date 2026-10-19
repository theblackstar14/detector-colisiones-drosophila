"""
Modelo fenomenológico de detección de colisiones tipo LGMD (Yue & Rind, 2006). Línea base (M1).
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import convolve

K_INH = np.array([[0.125, 0.25, 0.125],
                  [0.25,  0.0,  0.25],
                  [0.125, 0.25, 0.125]])
K_GRUPO = np.ones((3, 3)) / 9.0


@dataclass
class ParamsLGMD:
    W_I: float = 0.6
    C_w: float = 4.0
    delta_c: float = 0.01
    T_de: float = 15.0
    T_ffi: float = 25.0
    C_s: float = 8.0
    T_s: float = 0.75
    n_sp: int = 4
    warmup: int = 3


class LGMD:
    def __init__(self, p: ParamsLGMD = ParamsLGMD()):
        self.p = p
        self.reset()

    def reset(self):
        self.L_prev = None
        self.P_prev = None
        self.n = 0

    def paso(self, L):
        p = self.p
        self.n += 1
        if self.L_prev is None:
            self.L_prev = L
            self.P_prev = np.zeros_like(L)
            return 0.5, False, 0.0
        P = np.abs(L - self.L_prev)
        I = convolve(self.P_prev, K_INH, mode="constant")
        S = P - p.W_I * I
        S[S < 0] = 0
        Ce = convolve(S, K_GRUPO, mode="constant")
        w = p.delta_c + Ce.max() / p.C_w
        G = S * Ce / w
        G[G < p.T_de] = 0
        n = L.size
        K = G.sum() / n
        kappa = 1.0 / (1.0 + np.exp(-K / p.C_s))
        F = self.P_prev.sum() / n
        ffi = F > p.T_ffi
        self.L_prev, self.P_prev = L, P
        if self.n <= p.warmup:
            return 0.5, False, 0.0
        return kappa, ffi, K
