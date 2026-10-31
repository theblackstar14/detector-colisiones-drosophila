"""Segunda prueba piloto: video público desde la cabina de un bus del Metropolitano (Lima).
El video publicado es un montaje con cortes; cada toma continua se procesa por separado (modelos reiniciados),
se descartan las alarmas que empiezan en los primeros 2 s de cada toma (arranque) y se enmascara con un valor
constante la franja superior del cuadro (22 % de la altura), donde el video superpone rótulos de estaciones.
Umbrales y parámetros congelados de la simulación; no se recalibra nada.
  python procesar_bus.py clips.json carpeta_videos --fov 70 --salida carpeta"""
import argparse, json, os, sys, time
RAIZ = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import numpy as np, cv2
from models.lgmd import LGMD, ParamsLGMD
from models.lplc2 import ViaLooming, ParamsLPLC2
from models.rumbo import rumbo
from video.procesar_video import preparar_retina, Alarmas, U, DT
from eval.corrida_revision import tau_signal
UN = json.load(open(RAIZ / "resultados/simulacion/umbrales_nuevos.json"))
T = {"M1": U["M1_LGMD"]["T"], "M2_1.25": U["M2_1.25"]["T"], "M2_5.0": U["M2_5.0"]["T"], "M2.1_1.25": 0.04679129604743574,
     "M2.1_5.0": U["M2.1_5.0"]["T"], "tau": UN["tau"]["T"]}
EPS = {"1.25": 5.0, "5.0": U["M2.1_5.0"]["eps"]}; K = int(round(0.3 / DT)); ARRANQUE = 2.0; MASCARA = 0.22

def bloqueo(b):
    return len(b) > K and np.isfinite(b[-1]) and np.isfinite(b[-1 - K])

def procesar_toma(cap, t0, t1, fov):
    m1 = LGMD(ParamsLGMD(T_de=5)); m125 = ViaLooming(ParamsLPLC2(deg_px=1.25, usar_lc4=False, paso_px=4))
    m5 = ViaLooming(ParamsLPLC2(deg_px=5.0, usar_lc4=False, paso_px=1))
    al = {k: Alarmas() for k in T}; b125, b5 = [], []; prev = None; lat = []
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000); n = int((t1 - t0) / DT)
    for i in range(n):
        ok, fr = cap.read()
        if not ok: break
        s = time.perf_counter()
        fr = fr.copy(); fr[: int(round(MASCARA * fr.shape[0]))] = 128
        ret = preparar_retina(fr, fov, 128.0)
        k1, ffi, _ = m1.paso(ret); g125 = m125.paso(ret)[0]; g5 = m5.paso(ret.reshape(8, 4, 24, 4).mean(axis=(1, 3)))[0]
        b125.append(rumbo(getattr(m125, "mapa_lplc2", None), 1.25, 4, 96) if m125.n > 3 else np.nan)
        b5.append(rumbo(getattr(m5, "mapa_lplc2", None), 5.0, 1, 24) if m5.n > 3 else np.nan)
        bl125 = bloqueo(b125) and (abs(b125[-1]) - abs(b125[-1 - K])) / 0.3 > EPS["1.25"]
        bl5 = bloqueo(b5) and (abs(b5[-1]) - abs(b5[-1 - K])) / 0.3 > EPS["5.0"]
        cur = ret.astype(np.uint8); ts_ = tau_signal(prev, cur); prev = cur
        lat.append((time.perf_counter() - s) * 1000)
        al["M1"].paso(i, k1 > T["M1"] and not ffi); al["M2_1.25"].paso(i, g125 > T["M2_1.25"]); al["M2_5.0"].paso(i, g5 > T["M2_5.0"])
        al["M2.1_1.25"].paso(i, g125 > T["M2.1_1.25"] and not bl125); al["M2.1_5.0"].paso(i, g5 > T["M2.1_5.0"] and not bl5)
        al["tau"].paso(i, ts_ > T["tau"])
    ini = {k: [round(t0 + j * DT, 2) for j in v.inicios if j * DT >= ARRANQUE] for k, v in al.items()}
    return ini, max(0.0, i * DT - ARRANQUE), lat

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("clips"); ap.add_argument("carpeta"); ap.add_argument("--fov", type=float, default=70)
    ap.add_argument("--salida", required=True); a = ap.parse_args(); os.makedirs(a.salida, exist_ok=True)
    clips = [c for c in json.load(open(a.clips)) if c.get("usar")]
    caps = {}; total = 0.0; res = {k: [] for k in T}; lat = []; por_toma = []
    for c in clips:
        cap = caps.setdefault(c["seg"], cv2.VideoCapture(os.path.join(a.carpeta, c["seg"] + ".mp4")))
        ini, dur, l = procesar_toma(cap, c["t0"], c["t1"], a.fov); total += dur; lat += l
        for k, v in ini.items(): res[k] += [dict(toma=c["id"], seg=c["seg"], t=t, t_video=round(t + c["offset"], 2)) for t in v]
        por_toma.append(dict(toma=c["id"], efectivo_s=round(dur, 1), **{k: len(v) for k, v in ini.items()}))
    minutos = total / 60
    out = dict(fov=a.fov, tomas=len(clips), minutos_efectivos=round(minutos, 2),
               latencia_ms=dict(media=round(float(np.mean(lat)), 2), p95=round(float(np.percentile(lat, 95)), 2)),
               por_minuto={k: round(len(v) / minutos, 2) for k, v in res.items()}, alarmas=res, por_toma=por_toma)
    json.dump(out, open(os.path.join(a.salida, "resumen_bus.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("fov", "tomas", "minutos_efectivos", "latencia_ms", "por_minuto")}))

if __name__ == "__main__":
    main()
