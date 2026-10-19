"""
Prueba B semisintética: inserta la parte trasera de un vehículo (1,8 m x 1,5 m) sobre un tramo REAL del video,
con geometría de cámara estenopeica (FOV horizontal conocido), y una distancia exacta d(t) = d0 - v*t.
  python insertar.py surco.mp4 --t0 454 --v 6 --lateral 0 --salida clip.mp4   (lateral 0 = mismo carril; 3.5 = carril vecino)
  --sin-objeto  genera el clip de control idéntico, sin vehículo
Devuelve en stdout el JSON con las marcas exactas (distancia:segundo) para procesar_video.py.
"""
import argparse, json
import numpy as np, cv2

ANCHO, ALTO = 1.8, 1.5        # m, parte trasera de un auto
H_CAM = 1.3                   # m, altura supuesta de la cámara
D0, D_FIN = 25.0, 1.0         # m
PREVIO, POST = 3.0, 1.0       # s de video antes de que aparezca el objeto y después de llegar a D_FIN
SS = 4                        # supermuestreo para bordes suaves


def dibujar(frame, d, lat, f, cx, cy, rng):
    h, w = frame.shape[:2]
    x0 = cx + f * (lat - ANCHO / 2) / d; x1 = cx + f * (lat + ANCHO / 2) / d
    yb = cy + f * H_CAM / d;              yt = cy - f * (ALTO - H_CAM) / d
    m = np.zeros((h * SS, w * SS), np.float32)
    def rect(a, b, c, e, val):
        a, b = int(round(a * SS)), int(round(b * SS)); c, e = int(round(c * SS)), int(round(e * SS))
        a, c = max(a, 0), max(c, 0); b, e = min(b, w * SS), min(e, h * SS)
        if b > a and e > c: m[c:e, a:b] = val
    rect(x0, x1, yt, yb, 1.0)
    mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)[..., None]
    # apariencia: carrocería gris oscuro, luneta más oscura, dos luces traseras rojas, sombra en la base
    obj = np.full((h, w, 3), (58, 58, 62), np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xs - x0) / max(x1 - x0, 1e-3); v = (ys - yt) / max(yb - yt, 1e-3)
    obj[(v < 0.38) & (v > 0.05) & (u > 0.12) & (u < 0.88)] = (30, 30, 34)
    luz = (v > 0.42) & (v < 0.55) & (((u > 0.03) & (u < 0.2)) | ((u > 0.8) & (u < 0.97)))
    obj[luz] = (40, 40, 170)
    obj[v > 0.9] = (25, 25, 25)
    obj = cv2.GaussianBlur(obj, (0, 0), 0.6) + rng.normal(0, 2.0, obj.shape)
    return np.clip(frame * (1 - mask) + obj * mask, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--v", type=float, default=6.0); ap.add_argument("--lateral", type=float, default=0.0)
    ap.add_argument("--fov", type=float, default=70.0); ap.add_argument("--salida", required=True)
    ap.add_argument("--sin-objeto", action="store_true"); a = ap.parse_args()
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(3)), int(cap.get(4)); f = (w / 2) / np.tan(np.radians(a.fov / 2)); cx, cy = w / 2, h / 2
    dur_obj = (D0 - D_FIN) / a.v; total = PREVIO + dur_obj + POST
    cap.set(cv2.CAP_PROP_POS_MSEC, a.t0 * 1000)
    vw = cv2.VideoWriter(a.salida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)); rng = np.random.default_rng(0)
    for i in range(int(round(total * fps))):
        ok, fr = cap.read()
        if not ok: break
        t = i / fps
        if not a.sin_objeto and t >= PREVIO:
            d = max(D_FIN, D0 - a.v * (t - PREVIO))
            fr = dibujar(fr.astype(np.float32), d, a.lateral, f, cx, cy, rng)
        vw.write(fr)
    vw.release()
    marcas = {dd: round(PREVIO + (D0 - dd) / a.v, 3) for dd in (25, 20, 15, 10, 5, 2, 1)}
    print(json.dumps({"marcas": ",".join(f"{k}:{v}" for k, v in marcas.items()), "duracion": round(total, 2),
                      "t_contacto": round(PREVIO + D0 / a.v, 3)}))


if __name__ == "__main__":
    main()
