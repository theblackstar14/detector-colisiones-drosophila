"""
Prueba piloto con video real: procesa un video con M1 (LGMD), M2 (LPLC2, 1.25°) y M2.1 (LPLC2 + rumbo, 5°)
usando EXACTAMENTE los parámetros y umbrales congelados de la simulación (no se recalibra nada).

  python video/procesar_video.py VIDEO.mp4 --fov 70 [--inicio s] [--fin s] [--marcas "20:t,15:t,..."] [--sin-video]
"""
import argparse, json, os, sys, time, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
from models.lgmd import LGMD, ParamsLGMD
from models.lplc2 import ViaLooming, ParamsLPLC2
from models.rumbo import rumbo

AQUI = os.path.dirname(os.path.abspath(__file__))
U = json.load(open(os.path.join(AQUI, "umbrales_congelados.json")))
W, H, DEG = 96, 32, 1.25
FPS = 30.0; DT = 1 / FPS
N_SP = 4
FIN_EPISODIO = 15
MODELOS = {"M1_LGMD": "M1 LGMD", "M2_1.25": "M2 1.25", "M2.1_5.0": "M2.1 5"}


def preparar_retina(frame, fov_h, relleno):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    f = (w / 2) / np.tan(np.radians(fov_h / 2))
    mitad_v = f * np.tan(np.radians(20))
    y0, y1 = int(round(h / 2 - mitad_v)), int(round(h / 2 + mitad_v))
    if y0 < 0:
        pad = -y0; g = cv2.copyMakeBorder(g, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=relleno); y0 += pad; y1 += pad
    recorte = g[y0:y1]
    cols = min(W, int(round(fov_h / DEG)))
    peq = cv2.resize(recorte, (cols, H), interpolation=cv2.INTER_AREA)
    ret = np.full((H, W), relleno, np.float32)
    c0 = (W - cols) // 2
    ret[:, c0:c0 + cols] = peq
    return ret


class Alarmas:
    def __init__(self): self.racha = 0; self.bajo = 0; self.activa = False; self.inicios = []
    def paso(self, i, arriba):
        self.racha = self.racha + 1 if arriba else 0
        if not self.activa and self.racha >= N_SP:
            self.activa = True; self.inicios.append(i); self.bajo = 0
        elif self.activa:
            self.bajo = 0 if arriba else self.bajo + 1
            if self.bajo >= FIN_EPISODIO: self.activa = False
        return self.activa


def dibujar(frame, ret, i, t_abs, s, act, bloq):
    alto = 540; ancho = int(frame.shape[1] * alto / frame.shape[0])
    img = cv2.resize(frame, (ancho, alto))
    r = cv2.resize(ret.astype(np.uint8), (W * 4, H * 4), interpolation=cv2.INTER_NEAREST)
    img[10:10 + H * 4, 10:10 + W * 4] = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (10, 10), (10 + W * 4, 10 + H * 4), (255, 255, 255), 1)
    cv2.putText(img, "retina 96x32", (14, 22 + H * 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y = 190
    for k, nom in MODELOS.items():
        v = min(2.0, max(0.0, s[k])); x0 = 14
        cv2.rectangle(img, (x0, y), (x0 + 200, y + 14), (60, 60, 60), -1)
        cv2.rectangle(img, (x0, y), (x0 + int(100 * v), y + 14), (40, 40, 220) if act[k] else (200, 170, 60), -1)
        cv2.line(img, (x0 + 100, y - 2), (x0 + 100, y + 16), (255, 255, 255), 1)
        etq = nom + ("  ALARMA" if act[k] else "") + ("  (bloqueada por rumbo)" if k == "M2.1_5.0" and bloq and not act[k] else "")
        cv2.putText(img, etq, (x0 + 210, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 24
    cv2.putText(img, f"t = {t_abs:7.2f} s", (ancho - 170, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if any(act.values()):
        cv2.rectangle(img, (0, alto - 36), (ancho, alto), (30, 30, 200), -1)
        cv2.putText(img, "ALARMA: " + ", ".join(MODELOS[k] for k in act if act[k]), (12, alto - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--fov", type=float, required=True)
    ap.add_argument("--marcas", default=None); ap.add_argument("--inicio", type=float, default=0); ap.add_argument("--fin", type=float, default=None)
    ap.add_argument("--sin-video", action="store_true"); ap.add_argument("--salida", default=None); a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened(): sys.exit(f"No se pudo abrir {a.video}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nombre = os.path.splitext(os.path.basename(a.video))[0]
    out = a.salida or os.path.join(os.path.dirname(os.path.abspath(a.video)), f"salida_{nombre}"); os.makedirs(out, exist_ok=True)
    print(f"Video: {a.video} | {fps_in:.1f} fps -> {FPS:.0f} fps | FOV {a.fov}° | desde {a.inicio} s", flush=True)

    m1 = LGMD(ParamsLGMD(T_de=5))
    m2 = ViaLooming(ParamsLPLC2(deg_px=1.25, usar_lc4=False, paso_px=4))
    m5 = ViaLooming(ParamsLPLC2(deg_px=5.0, usar_lc4=False, paso_px=1))
    T = {k: U[k]["T"] for k in MODELOS}; EPS = U["M2.1_5.0"]["eps"]; K = int(round(0.3 / DT))
    alar = {k: Alarmas() for k in MODELOS}; betas = []
    filas = []; lat = {k: [] for k in list(MODELOS) + ["total"]}; retinas = []
    vw = None; relleno = None; i = 0; t_obj = a.inicio
    if a.inicio > 0: cap.set(cv2.CAP_PROP_POS_MSEC, max(0, a.inicio - 1) * 1000)
    while True:
        ok, frame = cap.read()
        if not ok: break
        t_in = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - 1 / fps_in
        if t_in + 1e-6 < t_obj - 0.5 / FPS: continue
        if a.fin is not None and t_in > a.fin: break
        t_obj += DT
        t0 = time.perf_counter()
        if relleno is None: relleno = float(np.median(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        ret = preparar_retina(frame, a.fov, relleno)
        t1 = time.perf_counter(); k1, ffi, _ = m1.paso(ret); lat["M1_LGMD"].append((time.perf_counter() - t1) * 1000)
        t1 = time.perf_counter(); g2 = m2.paso(ret)[0]; lat["M2_1.25"].append((time.perf_counter() - t1) * 1000)
        t1 = time.perf_counter()
        r5 = ret.reshape(8, 4, 24, 4).mean(axis=(1, 3)); g5 = m5.paso(r5)[0]
        b = rumbo(getattr(m5, "mapa_lplc2", None), 5.0, 1, 24) if m5.n > 3 else np.nan; betas.append(b)
        bloq = False
        if len(betas) > K and np.isfinite(betas[-1]) and np.isfinite(betas[-1 - K]):
            bloq = (abs(betas[-1]) - abs(betas[-1 - K])) / 0.3 > EPS
        lat["M2.1_5.0"].append((time.perf_counter() - t1) * 1000)
        lat["total"].append((time.perf_counter() - t0) * 1000)
        s = {"M1_LGMD": (k1 - 0.5) / (T["M1_LGMD"] - 0.5), "M2_1.25": g2 / T["M2_1.25"], "M2.1_5.0": g5 / T["M2.1_5.0"]}
        act = {"M1_LGMD": alar["M1_LGMD"].paso(i, k1 > T["M1_LGMD"] and not ffi),
               "M2_1.25": alar["M2_1.25"].paso(i, g2 > T["M2_1.25"]),
               "M2.1_5.0": alar["M2.1_5.0"].paso(i, g5 > T["M2.1_5.0"] and not bloq)}
        t_abs = a.inicio + i * DT
        filas.append([round(t_abs, 4)] + [round(s[k], 5) for k in MODELOS] + [int(act[k]) for k in MODELOS] +
                     ["" if not np.isfinite(b) else round(b, 2), int(bloq)])
        retinas.append(ret.astype(np.uint8))
        if not a.sin_video:
            img = dibujar(frame, ret, i, t_abs, s, act, bloq)
            if vw is None:
                vw = cv2.VideoWriter(os.path.join(out, "anotado.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (img.shape[1], img.shape[0]))
            vw.write(img)
        i += 1
        if i % 1800 == 0: print(f"  {i * DT / 60:5.1f} min procesados", flush=True)
    cap.release()
    if vw is not None: vw.release()
    if i == 0: sys.exit("No se procesó ningún cuadro")
    with open(os.path.join(out, "cuadros.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_s"] + [f"senal_{k}" for k in MODELOS] + [f"alarma_{k}" for k in MODELOS] + ["rumbo_grados", "bloqueo_rumbo"]); w.writerows(filas)
    minutos = i * DT / 60
    with open(os.path.join(out, "alarmas.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["modelo", "inicio_s", "causa"])
        for k in MODELOS:
            for j in alar[k].inicios: w.writerow([k, round(a.inicio + j * DT, 2), ""])
    res = dict(video=os.path.basename(a.video), fov_grados=a.fov, inicio_s=a.inicio, duracion_s=round(i * DT, 1), cuadros=i,
               latencia_ms={k: dict(media=round(float(np.mean(v)), 2), p95=round(float(np.percentile(v, 95)), 2), max=round(float(np.max(v)), 2)) for k, v in lat.items()},
               tiempo_real=bool(np.percentile(lat["total"], 95) < 1000 * DT),
               alarmas={k: dict(episodios=len(alar[k].inicios), por_minuto=round(len(alar[k].inicios) / minutos, 2)) for k in MODELOS},
               nota="Umbrales congelados de la simulación; no se recalibró nada.")
    if a.marcas:
        pares = sorted([(float(t), float(d)) for d, t in (m.split(":") for m in a.marcas.split(","))])
        ts, ds = np.array([p[0] for p in pares]), np.array([p[1] for p in pares])
        pend, orig = np.polyfit(ts, ds, 1); t_contacto = -orig / pend
        res["prueba_B"] = {"velocidad_panel_m_s": round(-pend, 2), "t_contacto_estimado_s": round(t_contacto, 2)}
        for k in MODELOS:
            ini = [a.inicio + j * DT for j in alar[k].inicios if a.inicio + j * DT >= ts[0] - 1.0]
            res["prueba_B"][k] = (dict(t_alarma_s=round(ini[0], 2), anticipacion_s=round(t_contacto - ini[0], 2),
                                      distancia_al_alarmar_m=round(float(orig + pend * ini[0]), 1)) if ini else "sin alarma")
    json.dump(res, open(os.path.join(out, "resumen.json"), "w"), indent=1, ensure_ascii=False)
    np.savez_compressed(os.path.join(out, "retina.npz"), retina=np.array(retinas))
    print(json.dumps(res, indent=1, ensure_ascii=False)); print(f"Resultados en: {out}")


if __name__ == "__main__":
    main()
