"""Análisis de la revisión: reproduce la Tabla II con los umbrales congelados, calibra los modelos nuevos con el mismo
protocolo (semillas pares 0–38, máx. detección con FA ≤ 5 %) y evalúa en prueba (40–79)."""
RAIZ = __import__("pathlib").Path(__file__).resolve().parents[1]
import sys, glob, pickle, json
sys.path.insert(0, str(RAIZ))
import numpy as np
from scipy.stats import binomtest, wilcoxon
from models.rumbo import compuerta
R = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(str(RAIZ / "resultados/simulacion/corridas/*.pkl")))]
CAL = [r for r in R if r["semilla"] < 40]; TEST = [r for r in R if r["semilla"] >= 40]
U = json.load(open(RAIZ / "video/umbrales_congelados.json"))
U["M2.1_1.25"] = {"T": 0.04679129604743574, "eps": 5.0}   # congelado original (umbrales_congelados del paquete de video real)

def primera(s, f, T, n=4):
    c = 0
    for i, (a, b) in enumerate(zip(s, f)):
        c = c + 1 if (a > T and not b) else 0
        if c >= n: return i
    return None

def spec(nombre):
    """(señal, compuerta) para cada modelo."""
    if nombre == "M1": return lambda r: (r["senal"]["M1"], r["ffi"])
    if nombre.startswith("M2.1_"):
        g = nombre.split("_")[1]; eps = U[nombre]["eps"] if nombre in U else None
        return lambda r, e=None: (r["senal"][f"M2_{g}"], compuerta(r["beta"][g], e if e is not None else eps))
    return lambda r: (r["senal"][nombre], np.zeros(len(r["t"]), bool))

def evalua(runs, get, T):
    out = []
    for r in runs:
        s, f = get(r); i = primera(s, f, T)
        out.append(dict(esc=r["escenario"], sem=r["semilla"], col=r["tc"] is not None, alarma=i is not None,
                        ant=(r["tc"] - r["t"][i]) if (i is not None and r["tc"] is not None) else None))
    return out

def resumen(res):
    col = [x for x in res if x["col"]]; nc = [x for x in res if not x["col"]]
    ants = [x["ant"] for x in col if x["ant"] is not None]
    return dict(det=np.mean([x["alarma"] for x in col]), fa=np.mean([x["alarma"] for x in nc]),
                det05=np.mean([x["ant"] is not None and x["ant"] >= 0.5 for x in col]),
                det1=np.mean([x["ant"] is not None and x["ant"] >= 1.0 for x in col]),
                ant_med=float(np.median(ants)) if ants else np.nan, n_det=len(ants))

def calibra(get, runs):
    vals = np.concatenate([get(r)[0] for r in runs]); grid = np.unique(np.quantile(vals, np.linspace(0.5, 0.99999, 400)))
    best = None
    for T in grid:
        s = resumen(evalua(runs, get, T))
        if s["fa"] <= 0.05 and (best is None or s["det"] > best[1] or (s["det"] == best[1] and T > best[0])): best = (float(T), s["det"])
    return best[0]

def ic_boot(res, clave, B=4000, seed=1):
    rng = np.random.default_rng(seed); v = np.array([x["alarma"] for x in res if x["col"] == (clave == "det")], float)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]; return np.percentile(m, [2.5, 97.5])

out = {"congelados": {}, "nuevos": {}}
congelados = {"M1": U["M1_LGMD"]["T"], "M2_1.25": U["M2_1.25"]["T"], "M2_5.0": U["M2_5.0"]["T"],
              "M2.1_1.25": U["M2.1_1.25"]["T"], "M2.1_5.0": U["M2.1_5.0"]["T"]}
RES = {}
for m, T in congelados.items():
    RES[m] = evalua(TEST, spec(m), T); s = resumen(RES[m]); s["det_ic"] = ic_boot(RES[m], "det").tolist(); s["fa_ic"] = ic_boot(RES[m], "fa").tolist()
    out["congelados"][m] = dict(T=T, **s)
nuevos = ["M2u_1.25", "M2u_5.0"] + [f"M2p{k}_5.0" for k in range(5)] + ["tau"]
for m in nuevos:
    T = calibra(spec(m), CAL); RES[m] = evalua(TEST, spec(m), T); s = resumen(RES[m])
    s["det_ic"] = ic_boot(RES[m], "det").tolist(); s["fa_ic"] = ic_boot(RES[m], "fa").tolist(); out["nuevos"][m] = dict(T=T, **s)

def mcnemar(a, b, col=True):
    x = [(p["alarma"], q["alarma"]) for p, q in zip(RES[a], RES[b]) if p["col"] == col]
    b_ = sum(1 for p, q in x if p and not q); c_ = sum(1 for p, q in x if q and not p)
    return dict(b=b_, c=c_, p=binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ else 1.0)

def hl_pareado(a, b, B=4000):
    pares = [(p["ant"], q["ant"]) for p, q in zip(RES[a], RES[b]) if p["col"] and p["ant"] is not None and q["ant"] is not None]
    d = np.array([p - q for p, q in pares]); w = wilcoxon(d)
    walsh = lambda v: np.median([(v[i] + v[j]) / 2 for i in range(len(v)) for j in range(i, len(v))])
    rng = np.random.default_rng(3); bs = [walsh(d[rng.integers(0, len(d), len(d))]) for _ in range(1000)]
    return dict(n=len(d), W=float(w.statistic), p=float(w.pvalue), HL=float(walsh(d)), HL_ic=np.percentile(bs, [2.5, 97.5]).tolist())

out["pruebas"] = {
    "det M2_1.25 vs M1": mcnemar("M2_1.25", "M1"), "det M2.1_5 vs M2_5": mcnemar("M2.1_5.0", "M2_5.0"),
    "FA M2.1_5 vs M2_5": mcnemar("M2.1_5.0", "M2_5.0", col=False),
    "det M2_5 vs M2u_5": mcnemar("M2_5.0", "M2u_5.0"), "det M2_1.25 vs M2u_1.25": mcnemar("M2_1.25", "M2u_1.25"),
    "det M2_1.25 vs tau": mcnemar("M2_1.25", "tau"), "det M2.1_5 vs tau": mcnemar("M2.1_5.0", "tau"),
    "ant M2_1.25 vs M1": hl_pareado("M2_1.25", "M1"), "ant M2_1.25 vs tau": hl_pareado("M2_1.25", "tau"),
    "ant M2_5 vs M2u_5": hl_pareado("M2_5.0", "M2u_5.0"), "ant M2_1.25 vs M2u_1.25": hl_pareado("M2_1.25", "M2u_1.25")}

# curva de anticipación sobre las 120 colisiones: fracción detectada con anticipación >= t
ts = np.round(np.arange(0, 1.51, 0.05), 2)
out["curva_ant"] = {m: [float(np.mean([x["ant"] is not None and x["ant"] >= t for x in RES[m] if x["col"]])) for t in ts]
                    for m in ["M1", "M2_1.25", "M2_5.0", "M2.1_1.25", "M2.1_5.0", "M2u_1.25", "M2u_5.0", "tau"]}
out["curva_t"] = ts.tolist()
# curvas detección–FA en prueba (barrido de umbral, análisis sin umbral fijo)
roc = {}
for m in ["M1", "M2_1.25", "M2_5.0", "M2.1_5.0", "M2u_5.0", "tau"]:
    get = spec(m); vals = np.concatenate([get(r)[0] for r in TEST]); pts = []
    for T in np.unique(np.quantile(vals, np.linspace(0.3, 0.99999, 150))):
        s = resumen(evalua(TEST, get, T)); pts.append((s["fa"], s["det"]))
    pts = sorted(pts); fx = [p[0] for p in pts]; dy = np.maximum.accumulate([p[1] for p in pts]).tolist()
    roc[m] = dict(fa=fx, det=dy, det_fa05=float(max([d for f, d in zip(fx, dy) if f <= 0.05], default=0)))
out["roc"] = roc
# por escenario (prueba)
out["por_escenario"] = {m: {e: float(np.mean([x["alarma"] for x in RES[m] if x["esc"] == e])) for e in sorted({x["esc"] for x in RES[m]})} for m in RES}
json.dump(out, open(RAIZ / "resultados/simulacion/analisis.json", "w"), indent=1, default=float)
U2 = {m: {"T": v["T"]} for m, v in out["nuevos"].items()}; json.dump(U2, open(RAIZ / "resultados/simulacion/umbrales_nuevos.json", "w"), indent=1)
print(json.dumps({k: {m: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in s.items() if kk not in ("det_ic", "fa_ic")} for m, s in out[k].items()} for k in ("congelados", "nuevos")}, indent=1))
print(json.dumps(out["pruebas"], indent=1, default=float)); print({m: round(v["det_fa05"], 3) for m, v in roc.items()})
