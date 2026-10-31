"""Kappa de Cohen entre los dos revisores ciegos de los 24 eventos del video del bus, con IC al 95 % por bootstrap
de percentiles sobre eventos y sobre bloques de 60 s del video (10 000 remuestreos), y distribución de causas."""
import csv, json, collections, os
import numpy as np
AQUI = os.path.dirname(os.path.abspath(__file__))
F = list(csv.DictReader(open(os.path.join(AQUI, "anotaciones_bus.csv"), encoding="utf-8")))
r1 = [f["causa_revisor_1"] for f in F]; r2 = [f["causa_revisor_2"] for f in F]; co = [f["causa_consenso"] for f in F]
seg = lambda t: int(t.split(":")[0]) * 60 + float(t.split(":")[1])
bloque = np.array([int(seg(f["tiempo_video"]) // 60) for f in F])
def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan"), po
k, po = kappa(r1, r2); grp = lambda s: ["V" if c in "AB" else "O" for c in s]
rng = np.random.default_rng(0); n = len(F)
ke = [kappa([r1[i] for i in s], [r2[i] for i in s])[0] for s in (rng.integers(0, n, n) for _ in range(10000))]
ub = np.unique(bloque); kb = []
for _ in range(10000):
    idx = [i for b in rng.choice(ub, len(ub)) for i in np.where(bloque == b)[0]]
    kb.append(kappa([r1[i] for i in idx], [r2[i] for i in idx])[0])
out = dict(n=n, acuerdo=po, kappa=k, ic95_eventos=np.nanpercentile(ke, [2.5, 97.5]).tolist(),
           ic95_bloques=np.nanpercentile(kb, [2.5, 97.5]).tolist(), bloques=int(len(ub)),
           kappa_cercano_vs_resto=kappa(grp(r1), grp(r2))[0], desacuerdos=[F[i]["evento"] for i in range(n) if r1[i] != r2[i]],
           revisor_1=dict(collections.Counter(r1)), revisor_2=dict(collections.Counter(r2)), consenso=dict(collections.Counter(co)))
json.dump(out, open(os.path.join(AQUI, "kappa_bus.json"), "w"), indent=1, ensure_ascii=False); print(json.dumps(out, indent=1, ensure_ascii=False))
