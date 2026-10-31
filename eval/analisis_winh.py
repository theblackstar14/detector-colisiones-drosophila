"""Calibra (semillas pares 0–38, máx. detección con FA ≤ 5 %) y evalúa en prueba (40–79) M2 5° con w_inh = 0; 0,237; 0,611; 1.
También McNemar de las permutaciones de pesos frente a M2 5°."""
import sys, glob, pickle, json
RAIZ = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ)); 
import numpy as np
from scipy.stats import binomtest
src = open(RAIZ / "eval/analisis_revision.py").read().split("out = {")[0]
exec(src)  # primera, evalua, resumen, calibra, ic_boot
W = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(str(RAIZ / "resultados/simulacion/winh/*.pkl")))]
Wc = [r for r in W if r["semilla"] < 40]; Wt = [r for r in W if r["semilla"] >= 40]
def mc(A, B, col=True):
    x = [(p["alarma"], q["alarma"]) for p, q in zip(A, B) if p["col"] == col]
    b = sum(1 for p, q in x if p and not q); c = sum(1 for p, q in x if q and not p)
    return dict(b=b, c=c, p=binomtest(b, b + c, .5).pvalue if b + c else 1.0)
out = {}
res = {}
for k in ["w0", "w0237", "w0611", "w1"]:
    g = lambda r, k=k: (r["senal"][k], np.zeros(len(r["t"]), bool))
    T = calibra(g, Wc); res[k] = evalua(Wt, g, T); s = resumen(res[k]); s["T"] = T
    s["det_ic"] = ic_boot(res[k], "det").tolist(); out[k] = s
for k in ["w0", "w0611", "w1"]:
    out[k]["mc_det_vs_base"] = mc(res[k], res["w0237"]); out[k]["mc_fa_vs_base"] = mc(res[k], res["w0237"], False)
# permutaciones vs M2_5.0 (de la corrida principal)
TEST_ = sorted(TEST, key=lambda r: (r["escenario"], r["semilla"]))
base = evalua(TEST_, spec("M2_5.0"), U["M2_5.0"]["T"])
for k in ["M2p1_5.0", "M2p3_5.0"]:
    T = calibra(spec(k), CAL); e = evalua(TEST_, spec(k), T)
    out[k] = dict(det=resumen(e)["det"], fa=resumen(e)["fa"], mc_det=mc(e, base), mc_fa=mc(e, base, False))
json.dump(out, open(RAIZ / "resultados/simulacion/winh/analisis_winh.json", "w"), indent=1, default=float)
for k, v in out.items(): print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
