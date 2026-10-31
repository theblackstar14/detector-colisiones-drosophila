"""Prueba de equivalencia (dos pruebas unilaterales, TOST) entre pesos del conectoma y pesos uniformes en M2.
Márgenes fijados después del análisis principal: ±5 puntos de detección y ±0,05 s de anticipación.
Equivalencia si el IC bootstrap pareado al 90 % cae dentro de los márgenes. Usa las corridas de prueba (semillas 40–79)."""
import os, sys, json
import numpy as np
AQUI = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(AQUI, "analisis_revision.py")).read()
exec(src[:src.index('out = {"congelados"')])          # funciones y datos comunes (primera, spec, evalua, calibra, ...)
M_DET, M_ANT, B = 0.05, 0.05, 10000
walsh = lambda v: np.median([(v[i] + v[j]) / 2 for i in range(len(v)) for j in range(i, len(v))]) if len(v) else np.nan
out = {}
for g in ("5.0", "1.25"):
    con = evalua(TEST, spec(f"M2_{g}"), U[f"M2_{g}"]["T"])
    uni = evalua(TEST, spec(f"M2u_{g}"), calibra(spec(f"M2u_{g}"), CAL))
    det = np.array([[p["alarma"], q["alarma"]] for p, q in zip(con, uni) if p["col"]], float)
    d = det[:, 0] - det[:, 1]
    rng = np.random.default_rng(7); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    ic90 = np.percentile(bs, [5, 95])
    ant = np.array([p["ant"] - q["ant"] for p, q in zip(con, uni) if p["col"] and p["ant"] is not None and q["ant"] is not None])
    rng = np.random.default_rng(8); hb = [walsh(ant[rng.integers(0, len(ant), len(ant))]) for _ in range(1000)]
    out[g] = dict(det_conectoma=float(det[:, 0].mean()), det_uniforme=float(det[:, 1].mean()), dif_det=float(d.mean()),
                  ic90_det=ic90.tolist(), equiv_det=bool(ic90[0] > -M_DET and ic90[1] < M_DET),
                  n_ant=int(len(ant)), HL_ant=float(walsh(ant)), ic95_HL=np.percentile(hb, [2.5, 97.5]).tolist(),
                  ic90_HL=np.percentile(hb, [5, 95]).tolist(),
                  equiv_ant=bool(np.percentile(hb, 5) > -M_ANT and np.percentile(hb, 95) < M_ANT))
os.makedirs(os.path.join(AQUI, "..", "resultados", "simulacion"), exist_ok=True)
json.dump(out, open(os.path.join(AQUI, "..", "results", "revision", "equivalencia.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
