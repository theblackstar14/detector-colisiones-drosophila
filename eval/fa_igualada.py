"""Análisis post hoc: M2 a 1,25° con umbral que iguala la tasa de falsas alarmas de la línea base (5,6 %) en prueba,
y detección máxima de cada modelo con FA <= 5 % y <= 10 % según las curvas de analisis.json."""
import sys, json, numpy as np
RAIZ = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
src = open(RAIZ / "eval/analisis_revision.py").read().split("out = {")[0]
exec(src)
a = json.load(open(RAIZ / "resultados/simulacion/analisis.json"))
for m, r in a["roc"].items():
    fa = np.array(r["fa"]); det = np.array(r["det"])
    print(m, "det@FA<=5%:", det[fa <= 0.05].max(), " det@FA<=10%:", det[fa <= 0.10].max())
# M2 1.25 con umbral post hoc que iguala FA de tau (5,6 %) en prueba
TEST_ = sorted(TEST, key=lambda r: (r["escenario"], r["semilla"]))
g = spec("M2_1.25"); vals = np.concatenate([g(r)[0] for r in TEST_]); grid = np.unique(np.quantile(vals, np.linspace(0.5, 0.99999, 600)))
best = None
for T in grid:
    s = resumen(evalua(TEST_, g, T))
    if s["fa"] <= 0.05625 and (best is None or s["det"] > best[1]): best = (T, s["det"], s["fa"])
print("M2_1.25 a FA<=5,6%:", best)
e = evalua(TEST_, g, best[0]); t = evalua(TEST_, spec("tau"), a["nuevos"]["tau"]["T"])
pares = [(p["ant"], q["ant"]) for p, q in zip(e, t) if p["col"] and p["ant"] is not None and q["ant"] is not None]
d = np.array([p - q for p, q in pares]); walsh = np.median([(d[i] + d[j]) / 2 for i in range(len(d)) for j in range(i, len(d))])
from scipy.stats import wilcoxon, binomtest
x = [(p["alarma"], q["alarma"]) for p, q in zip(e, t) if p["col"]]
b = sum(1 for p, q in x if p and not q); c = sum(1 for p, q in x if q and not p)
print("HL vs tau", round(walsh, 3), "n", len(d), "p", wilcoxon(d).pvalue, "discord", b, c, binomtest(b, b + c).pvalue)
