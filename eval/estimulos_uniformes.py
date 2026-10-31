"""Comprobación: la selectividad al acercamiento de M2 ante estímulos de laboratorio no depende de los pesos del conectoma.
Compara el pico de la fibra gigante (relativo al acercamiento) con pesos del conectoma y con pesos uniformes
para cuatro discos: acercamiento, alejamiento, traslación y oscurecimiento (retina de 96 × 32 px a 1,25°/px, 30 cuadros/s)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from models.lplc2 import ViaLooming, ParamsLPLC2
H, W = 32, 96; yy, xx = np.mgrid[0:H, 0:W]
def disco(cx, cy, r, lum=0.0):
    img = np.full((H, W), 200.0); img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = lum; return img
n = 60; t = np.arange(n) / 30; d = np.maximum(10 - 4.9 * t, 0.3)        # objeto de 1 m que se acerca a 4,9 m/s
r = np.clip(np.degrees(2 * np.arctan(1 / (2 * d))) / 2 / 1.25, 0.5, 15)  # radio en píxeles
EST = {"acercamiento": [disco(48, 16, x) for x in r], "alejamiento": [disco(48, 16, x) for x in r[::-1]],
       "traslacion": [disco(20 + 56 * i / n, 16, 6) for i in range(n)], "oscurecimiento": [disco(48, 16, 8, 200 - 180 * i / n) for i in range(n)]}
def picos(uniformes):
    out = {}
    for k, cuadros in EST.items():
        m = ViaLooming(ParamsLPLC2(deg_px=1.25, usar_lc4=False))
        if uniformes:
            m.w["w_brazo"] = {b: 1.0 for b in m.w["w_brazo"]}; m.w["w_t4"] = m.w["w_t5"] = 1.0
        out[k] = max(m.paso(c)[0] for c in cuadros)
    return {k: round(v / out["acercamiento"], 3) for k, v in out.items()}
print("conectoma:", picos(False)); print("uniformes:", picos(True))
