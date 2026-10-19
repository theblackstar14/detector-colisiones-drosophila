"""Retina de baja resolución (120° x 40°, 96 x 32 px, 1.25°/px) montada al frente del vehículo propio."""
import numpy as np

FOV_H, FOV_V = np.deg2rad(120), np.deg2rad(40)
W, H = 96, 32
SS = 4
H_CAM = 1.8
CIELO, PISTA = 210.0, 110.0


def render(escena, t, ruido=2.0, rng=None, escala=1):
    w, h = W * escala, H * escala
    Ws, Hs = w * SS, h * SS
    img = np.empty((Hs, Ws), dtype=np.float32)
    img[: Hs // 2] = CIELO
    img[Hs // 2:] = PISTA
    xe, ye, he = escena.ego.traj(t)
    Le = escena.ego.dims[0]
    cam = np.array([xe + Le / 2 * np.cos(he), ye + Le / 2 * np.sin(he)])
    c, s = np.cos(-he), np.sin(-he)
    R = np.array([[c, -s], [s, c]])
    objetos = []
    for o in escena.otros:
        loc = (o.esquinas(t) - cam) @ R.T
        delante = loc[:, 0] > 0.3
        if not delante.any():
            continue
        loc = loc.copy()
        loc[~delante, 0] = 0.3
        az = np.arctan2(loc[:, 1], loc[:, 0])
        prof = loc[:, 0].min()
        objetos.append((prof, az.min(), az.max(), o))
    for prof, az0, az1, o in sorted(objetos, key=lambda z: -z[0]):
        alto = o.dims[2]
        el_top = np.arctan((alto - H_CAM) / prof)
        el_bot = np.arctan(-H_CAM / prof)
        c0 = int(np.floor((FOV_H / 2 - az1) / FOV_H * Ws))
        c1 = int(np.ceil((FOV_H / 2 - az0) / FOV_H * Ws))
        r0 = int(np.floor((FOV_V / 2 - el_top) / FOV_V * Hs))
        r1 = int(np.ceil((FOV_V / 2 - el_bot) / FOV_V * Hs))
        c0, c1 = max(c0, 0), min(c1, Ws)
        r0, r1 = max(r0, 0), min(r1, Hs)
        if c1 > c0 and r1 > r0:
            img[r0:r1, c0:c1] = o.luminancia
    img = img.reshape(h, SS, w, SS).mean(axis=(1, 3))
    if ruido and rng is not None:
        img = img + rng.normal(0, ruido, img.shape)
    return np.clip(img, 0, 255)
