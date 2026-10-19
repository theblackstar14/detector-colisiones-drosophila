"""Revisión: procesa el video con configuraciones adicionales (M2 a 5°, M2.1 a 1,25° y línea base τ), con umbrales congelados.
   python video/procesar_extra.py VIDEO --fov 70 --inicio 125 --salida carpeta"""
RAIZ = __import__("pathlib").Path(__file__).resolve().parents[1]
import argparse, json, os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
from models.lplc2 import ViaLooming, ParamsLPLC2
from models.rumbo import rumbo
from video.procesar_video import preparar_retina, Alarmas, U, DT
from eval.corrida_revision import tau_signal
UN = json.load(open(RAIZ / "resultados/simulacion/umbrales_nuevos.json"))
T = {"M2_5.0": U["M2_5.0"]["T"], "M2.1_1.25": 0.04679129604743574, "tau": UN["tau"]["T"]}; EPS125 = 5.0; K = int(round(0.3 / DT))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--fov", type=float, required=True)
    ap.add_argument("--inicio", type=float, default=0); ap.add_argument("--fin", type=float, default=None); ap.add_argument("--salida", required=True)
    a = ap.parse_args(); os.makedirs(a.salida, exist_ok=True)
    cap = cv2.VideoCapture(a.video); fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    m125 = ViaLooming(ParamsLPLC2(deg_px=1.25, usar_lc4=False, paso_px=4)); m5 = ViaLooming(ParamsLPLC2(deg_px=5.0, usar_lc4=False, paso_px=1))
    al = {k: Alarmas() for k in T}; betas = []; relleno = None; i = 0; t_obj = a.inicio; prev = None
    if a.inicio > 0: cap.set(cv2.CAP_PROP_POS_MSEC, max(0, a.inicio - 1) * 1000)
    while True:
        ok, fr = cap.read()
        if not ok: break
        t_in = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - 1 / fps_in
        if t_in + 1e-6 < t_obj - 0.5 / 30: continue
        if a.fin is not None and t_in > a.fin: break
        t_obj += DT
        if relleno is None: relleno = float(np.median(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)))
        ret = preparar_retina(fr, a.fov, relleno)
        g125 = m125.paso(ret)[0]; g5 = m5.paso(ret.reshape(8, 4, 24, 4).mean(axis=(1, 3)))[0]
        b = rumbo(getattr(m125, "mapa_lplc2", None), 1.25, 4, 96) if m125.n > 3 else np.nan; betas.append(b)
        bloq = len(betas) > K and np.isfinite(betas[-1]) and np.isfinite(betas[-1 - K]) and (abs(betas[-1]) - abs(betas[-1 - K])) / 0.3 > EPS125
        cur = ret.astype(np.uint8); ts_ = tau_signal(prev, cur); prev = cur
        al["M2_5.0"].paso(i, g5 > T["M2_5.0"]); al["M2.1_1.25"].paso(i, g125 > T["M2.1_1.25"] and not bloq); al["tau"].paso(i, ts_ > T["tau"])
        i += 1
    minutos = i * DT / 60
    res = dict(fov=a.fov, duracion_s=round(i * DT, 1), alarmas={k: dict(episodios=len(v.inicios), por_minuto=round(len(v.inicios) / minutos, 2),
               inicios_s=[round(a.inicio + j * DT, 2) for j in v.inicios]) for k, v in al.items()})
    json.dump(res, open(os.path.join(a.salida, "resumen_extra.json"), "w"), indent=1); print(json.dumps({k: v["por_minuto"] for k, v in res["alarmas"].items()}))

if __name__ == "__main__":
    main()
