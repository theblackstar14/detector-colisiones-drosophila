"""Prueba semisintética: detección por modelo, IC por bootstrap sobre tramos (Clopper–Pearson sobre tramos si 0 % o 100 %)
y análisis pareado en los tramos válidos para todos los modelos. Entrada: resultados_v2.csv (correr_v2.py)."""
import sys, json
import numpy as np, pandas as pd
from scipy.stats import beta
CSV = sys.argv[1] if len(sys.argv) > 1 else "resultados_v2.csv"
d = pd.read_csv(CSV); MOD = ["M1", "M2_1.25", "M2_5.0", "M2.1_1.25", "M2.1_5.0", "tau"]
ctrl = d[d.caso == "control"]
excl = {m: set(ctrl[(ctrl.modelo == m) & (ctrl.alarmas_tras_aparicion > 0)].tramo) for m in MOD}
d["det"] = d.t_alarma.notna() & (d.t_alarma <= 3 + 24 / d.v + 1e-6)   # tolerancia de redondeo del CSV
def cp(k, n, a=0.05):
    return (0 if k == 0 else beta.ppf(a / 2, k, n - k + 1), 1 if k == n else beta.ppf(1 - a / 2, k + 1, n - k))
out = {}; rng = np.random.default_rng(11)
for m in MOD:
    s = d[(d.modelo == m) & (d.caso == "mismo_carril") & ~d.tramo.isin(excl[m])]
    vec = d[(d.modelo == m) & (d.caso == "carril_vecino") & ~d.tramo.isin(excl[m])]
    por = s.groupby("tramo").det.agg(["sum", "count"]); k, n = int(por["sum"].sum()), int(por["count"].sum())
    if k in (0, n): lo, hi = cp(int((por["sum"] > 0).sum()) if k else 0, len(por))
    else:
        bs = [];  t = por.index.values
        for _ in range(10000):
            sel = por.loc[rng.choice(t, len(t))]; bs.append(sel["sum"].sum() / sel["count"].sum())
        lo, hi = np.percentile(bs, [2.5, 97.5])
    dd = s[s.det]
    med = lambda col: [round(float(dd[dd.v == v][col].median()), 2) if (dd.v == v).any() else None for v in (3, 6, 9)]
    out[m] = dict(excluidos=sorted(int(x) for x in excl[m]), det=f"{k}/{n}", ic95=[round(100 * lo), round(100 * hi)],
                  dist_med_3_6_9=med("distancia"), antic_med_3_6_9=med("anticipacion"),
                  vecino=f"{int(vec.t_alarma.notna().sum())}/{len(vec)}",
                  parachoques_1_5m=f"{int((s.det & (s.distancia > 1.5)).sum())}/{n}")
comunes = sorted(set(d.tramo) - set().union(*excl.values())); out["tramos_comunes"] = [int(x) for x in comunes]
c = d[d.tramo.isin(comunes) & (d.caso == "mismo_carril")]
out["comunes_det"] = {m: int(c[(c.modelo == m)].det.sum()) for m in MOD}
def pareado(a, b):
    x = c[c.modelo == a].set_index(["tramo", "v"]); y = c[c.modelo == b].set_index(["tramo", "v"])
    j = x.join(y, lsuffix="_a", rsuffix="_b"); j = j[j.det_a & j.det_b]
    return dict(n=len(j), a_mas_lejos=int((j.distancia_a > j.distancia_b).sum()), dif_mediana_m=round(float((j.distancia_a - j.distancia_b).median()), 2))
out["pareado_M2.1_5_vs_M2_5"] = pareado("M2.1_5.0", "M2_5.0"); out["pareado_M2.1_5_vs_tau"] = pareado("M2.1_5.0", "tau")
cv = d[d.tramo.isin(comunes) & (d.caso == "carril_vecino")]
out["comunes_vecino"] = {m: int(cv[cv.modelo == m].t_alarma.notna().sum()) for m in MOD}
print(json.dumps(out, indent=1, ensure_ascii=False))
