"""
Modelo 2: vía de escape ante looming de Drosophila, restringida por el conectoma FlyWire v783.
Retina -> ON/OFF -> T4/T5 (correlador HR) -> LPLC2 (4 brazos, inhibición LPi, coincidencia) [+ LC4] -> fibra gigante.
"""
import json, os
from dataclasses import dataclass
import numpy as np
from scipy.signal import fftconvolve

_C = json.load(open(os.path.join(os.path.dirname(__file__), "conectoma_flywire783.json")))


def pesos_conectoma():
    t4, t5 = _C["T4_LPLC2"], _C["T5_LPLC2"]
    dirs = {k: t4["T4" + k] + t5["T5" + k] for k in "abcd"}
    media = np.mean(list(dirs.values()))
    h = (dirs["a"] + dirs["b"]) / 2 / media
    w_brazo = {"der": h, "izq": h, "arr": dirs["c"] / media, "abj": dirs["d"] / media}
    T4, T5 = sum(t4.values()), sum(t5.values())
    w_t4, w_t5 = 2 * T4 / (T4 + T5), 2 * T5 / (T4 + T5)
    inhib = {"gaba", "glutamate"}
    lpi = sum(v for k, v in _C["LPi_LPLC2"].items() if _C["nt"].get(k) in inhib)
    w_inh = lpi / (T4 + T5)
    g_p, g_c = _C["GF_in"]["LPLC2"], _C["GF_in"]["LC4"]
    return dict(w_brazo=w_brazo, w_t4=w_t4, w_t5=w_t5, w_inh=w_inh,
                w_gf_lplc2=g_p / (g_p + g_c), w_gf_lc4=g_c / (g_p + g_c))


@dataclass
class ParamsLPLC2:
    deg_px: float = 1.25
    rf_lplc2_deg: float = 30.0
    rf_lc4_deg: float = 20.0
    paso_px: int = 4
    tau_hp: float = 0.25
    tau_d: float = 0.05
    dt: float = 1 / 30
    usar_lplc2: bool = True
    usar_lc4: bool = True
    usar_lpi: bool = True
    coincidencia: bool = True
    norm_lplc2: float = 1.0
    norm_lc4: float = 1.0
    warmup: int = 3


def _kernels(R):
    y, x = np.mgrid[-R:R + 1, -R:R + 1]
    disco = x**2 + y**2 <= R**2
    k = {"der": disco & (x > 0) & (np.abs(y) <= x),
         "izq": disco & (x < 0) & (np.abs(y) <= -x),
         "abj": disco & (y > 0) & (np.abs(x) <= y),
         "arr": disco & (y < 0) & (np.abs(x) <= -y)}
    return {n: v.astype(np.float32) / v.sum() for n, v in k.items()}, disco.astype(np.float32) / disco.sum()


def _corr(img, k):
    return fftconvolve(img, k[::-1, ::-1], mode="same")


OPUESTO = {"der": "izq", "izq": "der", "arr": "abj", "abj": "arr"}


class ViaLooming:
    def __init__(self, p: ParamsLPLC2 = ParamsLPLC2()):
        self.p = p
        self.w = pesos_conectoma()
        R = int(round(p.rf_lplc2_deg / p.deg_px))
        self.K, _ = _kernels(R)
        _, self.K_lc4 = _kernels(int(round(p.rf_lc4_deg / p.deg_px)))
        self.a_hp = p.dt / (p.tau_hp + p.dt)
        self.a_d = p.dt / (p.tau_d + p.dt)
        self.reset()

    def reset(self):
        self.lp = None
        self.D = None
        self.n = 0

    def _emd(self, C, D):
        h = np.zeros_like(C); v = np.zeros_like(C)
        h[:, :-1] = D[:, :-1] * C[:, 1:] - C[:, :-1] * D[:, 1:]
        v[:-1, :] = D[:-1, :] * C[1:, :] - C[:-1, :] * D[1:, :]
        return {"der": np.maximum(h, 0), "izq": np.maximum(-h, 0),
                "abj": np.maximum(v, 0), "arr": np.maximum(-v, 0)}

    def paso(self, L):
        p, w = self.p, self.w
        L = L.astype(np.float32) / 255.0
        self.n += 1
        if self.lp is None:
            self.lp = L.copy()
            self.D = {"on": np.zeros_like(L), "off": np.zeros_like(L)}
            return 0.0, False, {}
        hp = L - self.lp
        self.lp += self.a_hp * (L - self.lp)
        C = {"on": np.maximum(hp, 0), "off": np.maximum(-hp, 0)}
        m_on, m_off = self._emd(C["on"], self.D["on"]), self._emd(C["off"], self.D["off"])
        for k in C:
            self.D[k] += self.a_d * (C[k] - self.D[k])
        M = {d: w["w_t4"] * m_on[d] + w["w_t5"] * m_off[d] for d in m_on}
        s = (slice(p.paso_px // 2, None, p.paso_px),) * 2
        lplc2 = 0.0
        if p.usar_lplc2:
            brazo = {}
            for d, k in self.K.items():
                afuera = _corr(M[d], k)
                adentro = _corr(M[OPUESTO[d]], k) if p.usar_lpi else 0.0
                brazo[d] = np.maximum(w["w_brazo"][d] * (afuera - w["w_inh"] * adentro), 0)
            self.M_ultimo, self.brazo_ultimo = M, brazo
            if p.coincidencia:
                u = np.sqrt(brazo["der"] * brazo["izq"]) + np.sqrt(brazo["arr"] * brazo["abj"])
            else:
                u = sum(brazo.values()) / 2
            self.mapa_lplc2 = u[s]
            lplc2 = float(u[s].sum()) / p.norm_lplc2
        lc4 = 0.0
        if p.usar_lc4:
            e = _corr(C["on"] + C["off"], self.K_lc4)
            lc4 = float((e[s] ** 2).sum()) / p.norm_lc4
        gf = w["w_gf_lplc2"] * lplc2 + w["w_gf_lc4"] * lc4
        if self.n <= p.warmup:
            return 0.0, False, {}
        return gf, False, {"lplc2": lplc2, "lc4": lc4}
