"""
Corrida para la revisión: mismos escenarios y semillas del paper (calibración: pares 0–38; prueba: 40–79),
mismos cuadros para todos los modelos. Guarda las señales crudas por corrida.
Modelos: M1 (LGMD), M2 con pesos del conectoma (1,25° y 5°) + rumbo, M2 con pesos uniformes (1,25° y 5°),
M2 con pesos permutados entre brazos (5°, 5 permutaciones) y línea base τ por divergencia del flujo óptico.
"""
RAIZ = __import__("pathlib").Path(__file__).resolve().parents[1]
import sys, os, zlib, copy, itertools, pickle, time
sys.path.insert(0, str(RAIZ))
import numpy as np, cv2
from multiprocessing import Pool
from sim.world import ESCENARIOS
from sim.retina import render
from models.lgmd import LGMD, ParamsLGMD
from models.lplc2 import ViaLooming, ParamsLPLC2
from models.rumbo import rumbo

OUT = str(RAIZ / "resultados/simulacion/corridas")
os.makedirs(OUT, exist_ok=True)
BASE = ViaLooming().w
VAL = [BASE["w_brazo"]["der"], BASE["w_brazo"]["arr"], BASE["w_brazo"]["abj"]]   # h, arr, abj
PERMS = [p for p in itertools.permutations(range(3)) if p != (0, 1, 2)]           # 5 permutaciones

def con_pesos(m, modo, perm=None):
    w = copy.deepcopy(m.w)
    if modo == "uniforme":
        w["w_brazo"] = {k: 1.0 for k in w["w_brazo"]}; w["w_t4"] = w["w_t5"] = 1.0
    elif modo == "permutado":
        h, a, b = (VAL[i] for i in perm)
        w["w_brazo"] = {"der": h, "izq": h, "arr": a, "abj": b}
    m.w = w; return m

def tau_signal(prev, cur):
    if prev is None: return 0.0
    f = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 2, 7, 3, 5, 1.1, 0)
    div = np.gradient(f[..., 0], axis=1) + np.gradient(f[..., 1], axis=0)
    roi = div[:, 32:64]                       # ±20° alrededor del frente
    return max(0.0, float(roi.mean())) * 30 / 2  # 1/τ en s⁻¹

def correr(args):
    nombre, s = args
    dest = f"{OUT}/{nombre}_{s}.pkl"
    if os.path.exists(dest): return dest
    rng = np.random.default_rng(s * 1000 + zlib.crc32(nombre.encode()) % 997)
    esc = ESCENARIOS[nombre](rng); tc = esc.tiempo_colision()
    ts = np.arange(0, tc if tc is not None else esc.duracion, esc.dt)
    lg = LGMD(ParamsLGMD(T_de=5))
    m = {"M2_1.25": ViaLooming(ParamsLPLC2(deg_px=1.25, usar_lc4=False, paso_px=4)),
         "M2_5.0": ViaLooming(ParamsLPLC2(deg_px=5.0, usar_lc4=False, paso_px=1)),
         "M2u_1.25": con_pesos(ViaLooming(ParamsLPLC2(deg_px=1.25, usar_lc4=False, paso_px=4)), "uniforme"),
         "M2u_5.0": con_pesos(ViaLooming(ParamsLPLC2(deg_px=5.0, usar_lc4=False, paso_px=1)), "uniforme")}
    for k, p in enumerate(PERMS):
        m[f"M2p{k}_5.0"] = con_pesos(ViaLooming(ParamsLPLC2(deg_px=5.0, usar_lc4=False, paso_px=1)), "permutado", p)
    sen = {k: [] for k in m}; sen["M1"] = []; ffi = []; b125 = []; b5 = []; tau = []; prev = None
    for t in ts:
        img = render(esc, t, rng=rng); im5 = img.reshape(8, 4, 24, 4).mean(axis=(1, 3))
        k1, f, _ = lg.paso(img); sen["M1"].append(k1); ffi.append(f)
        for k, mod in m.items():
            sen[k].append(mod.paso(im5 if k.endswith("5.0") else img)[0])
        b125.append(rumbo(getattr(m["M2_1.25"], "mapa_lplc2", None), 1.25, 4, 96) if m["M2_1.25"].n > 3 else np.nan)
        b5.append(rumbo(getattr(m["M2_5.0"], "mapa_lplc2", None), 5.0, 1, 24) if m["M2_5.0"].n > 3 else np.nan)
        cur = np.clip(img, 0, 255).astype(np.uint8); tau.append(tau_signal(prev, cur)); prev = cur
    sen["tau"] = tau
    r = dict(escenario=nombre, semilla=s, tc=tc, t=ts, senal={k: np.array(v, float) for k, v in sen.items()},
             ffi=np.array(ffi), beta={"1.25": np.array(b125), "5.0": np.array(b5)})
    pickle.dump(r, open(dest, "wb")); return dest

if __name__ == "__main__":
    tareas = [(n, s) for n in ESCENARIOS for s in list(range(0, 40, 2)) + list(range(40, 80))]
    if len(sys.argv) > 1 and sys.argv[1] == "prueba":
        t0 = time.time(); correr(("frontal_detenido", 71)); print("una corrida:", round(time.time() - t0, 1), "s"); sys.exit()
    t0 = time.time()
    with Pool(2) as p:
        for i, _ in enumerate(p.imap_unordered(correr, tareas), 1):
            if i % 50 == 0: print(i, "/", len(tareas), round(time.time() - t0), "s", flush=True)
    print("listo", len(tareas))
