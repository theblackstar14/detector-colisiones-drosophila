"""Ablación de la inhibición LPi (w_inh) en M2 a 5°: 0, 0,237 (conectoma, sin umbral), 0,611 (umbral de 5 sinapsis) y 1.
Mismos escenarios, semillas y cuadros que corrida_revision.py."""
import sys, os, zlib, copy, pickle, time
RAIZ = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import numpy as np
from multiprocessing import Pool
from sim.world import ESCENARIOS
from sim.retina import render
from models.lplc2 import ViaLooming, ParamsLPLC2
OUT = str(RAIZ / "resultados/simulacion/winh")
os.makedirs(OUT, exist_ok=True)
VALS = {"w0": 0.0, "w0237": None, "w0611": 0.611, "w1": 1.0}

def correr(args):
    nombre, s = args
    dest = f"{OUT}/{nombre}_{s}.pkl"
    if os.path.exists(dest): return dest
    rng = np.random.default_rng(s * 1000 + zlib.crc32(nombre.encode()) % 997)
    esc = ESCENARIOS[nombre](rng); tc = esc.tiempo_colision()
    ts = np.arange(0, tc if tc is not None else esc.duracion, esc.dt)
    m = {}
    for k, v in VALS.items():
        mod = ViaLooming(ParamsLPLC2(deg_px=5.0, usar_lc4=False, paso_px=1))
        if v is not None: w = copy.deepcopy(mod.w); w["w_inh"] = v; mod.w = w
        m[k] = mod
    sen = {k: [] for k in m}
    for t in ts:
        img = render(esc, t, rng=rng); im5 = img.reshape(8, 4, 24, 4).mean(axis=(1, 3))
        for k, mod in m.items(): sen[k].append(mod.paso(im5)[0])
    pickle.dump(dict(escenario=nombre, semilla=s, tc=tc, t=ts, senal={k: np.array(v, float) for k, v in sen.items()}), open(dest, "wb"))
    return dest

if __name__ == "__main__":
    tareas = [(n, s) for n in ESCENARIOS for s in list(range(0, 40, 2)) + list(range(40, 80))]
    if len(sys.argv) > 1:
        t0 = time.time(); correr(("frontal_detenido", 71)); print(round(time.time() - t0, 1), "s"); sys.exit()
    t0 = time.time()
    with Pool(2) as p:
        for i, _ in enumerate(p.imap_unordered(correr, tareas), 1):
            if i % 60 == 0: print(i, len(tareas), round(time.time() - t0), flush=True)
    print("listo")
