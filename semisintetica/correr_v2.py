"""Prueba semisintética v2 (revisión): 10 tramos sorteados (semilla 2026) entre los tramos en movimiento,
descartando los que tienen un vehículo a menos de ~10 m delante; 6 configuraciones; mismo protocolo."""
import subprocess, json, os, sys, csv
RAIZ = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
AQUI = __import__("pathlib").Path(__file__).resolve().parent
import numpy as np, cv2
from models.lgmd import LGMD, ParamsLGMD
from models.lplc2 import ViaLooming, ParamsLPLC2
from models.rumbo import rumbo
from video.procesar_video import preparar_retina, Alarmas, U, DT
from eval.corrida_revision import tau_signal
UN = json.load(open(RAIZ / "resultados/simulacion/umbrales_nuevos.json"))
TRAMOS = [145, 231, 419, 424, 456, 513, 726, 765, 831, 981]
DESCARTADOS = [289, 290, 721, 777, 934, 938]
VELS = [3, 6, 9]; LATS = {"mismo_carril": 0.0, "carril_vecino": 3.5}; K = int(round(0.3 / DT))
TH = {"M1": U["M1_LGMD"]["T"], "M2_1.25": U["M2_1.25"]["T"], "M2_5.0": U["M2_5.0"]["T"], "M2.1_1.25": 0.04679129604743574,
      "M2.1_5.0": U["M2.1_5.0"]["T"], "tau": UN["tau"]["T"]}
EPS = {"M2.1_1.25": 5.0, "M2.1_5.0": U["M2.1_5.0"]["eps"]}

def procesar(clip):
    cap = cv2.VideoCapture(clip); m1 = LGMD(ParamsLGMD(T_de=5))
    a = ViaLooming(ParamsLPLC2(deg_px=1.25, usar_lc4=False, paso_px=4)); b = ViaLooming(ParamsLPLC2(deg_px=5.0, usar_lc4=False, paso_px=1))
    al = {k: Alarmas() for k in TH}; be = {"1.25": [], "5.0": []}; relleno = None; prev = None; i = 0
    def bloq(lst, eps):
        return len(lst) > K and np.isfinite(lst[-1]) and np.isfinite(lst[-1 - K]) and (abs(lst[-1]) - abs(lst[-1 - K])) / 0.3 > eps
    while True:
        ok, fr = cap.read()
        if not ok: break
        if relleno is None: relleno = float(np.median(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)))
        ret = preparar_retina(fr, 70.0, relleno)
        k1, ffi, _ = m1.paso(ret); g1 = a.paso(ret)[0]; g5 = b.paso(ret.reshape(8, 4, 24, 4).mean(axis=(1, 3)))[0]
        be["1.25"].append(rumbo(getattr(a, "mapa_lplc2", None), 1.25, 4, 96) if a.n > 3 else np.nan)
        be["5.0"].append(rumbo(getattr(b, "mapa_lplc2", None), 5.0, 1, 24) if b.n > 3 else np.nan)
        cur = ret.astype(np.uint8); ts = tau_signal(prev, cur); prev = cur
        al["M1"].paso(i, k1 > TH["M1"] and not ffi); al["M2_1.25"].paso(i, g1 > TH["M2_1.25"]); al["M2_5.0"].paso(i, g5 > TH["M2_5.0"])
        al["M2.1_1.25"].paso(i, g1 > TH["M2.1_1.25"] and not bloq(be["1.25"], EPS["M2.1_1.25"]))
        al["M2.1_5.0"].paso(i, g5 > TH["M2.1_5.0"] and not bloq(be["5.0"], EPS["M2.1_5.0"])); al["tau"].paso(i, ts > TH["tau"])
        i += 1
    return {k: [j * DT for j in v.inicios] for k, v in al.items()}

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "surco.mp4"   # video original de YouTube (no se incluye en el repositorio)
filas = []; os.makedirs("clips_v2", exist_ok=True)
for t0 in TRAMOS:
    for caso, lat in [("control", None)] + list(LATS.items()):
        for v in ([3] if caso == "control" else VELS):
            clip = f"clips_v2/t{t0}_{caso}_v{v}.mp4"
            args = [sys.executable, str(AQUI / "insertar.py"), VIDEO, "--t0", str(t0), "--v", str(v), "--salida", clip]
            args += ["--sin-objeto"] if caso == "control" else ["--lateral", str(lat)]
            info = json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
            al = procesar(clip); os.remove(clip)
            for m, ini in al.items():
                ini2 = [x for x in ini if x >= 2.0]
                ta = ini2[0] if ini2 else None
                filas.append(dict(tramo=t0, caso=caso, v=v, modelo=m, alarmas_todas=len(ini), alarmas_tras_aparicion=len(ini2),
                                  t_alarma=ta, anticipacion=None if ta is None else round(info["t_contacto"] - ta, 3),
                                  distancia=None if ta is None or caso == "control" else round(max(1.0, 25 - v * (ta - 3.0)), 2)))
    print("tramo", t0, flush=True)
with open(RAIZ / "resultados/semisintetica/resultados_v2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
print("ok", len(filas))
