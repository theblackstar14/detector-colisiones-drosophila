"""M2.1: regla de rumbo constante calculada solo desde la retina (centroide de LPLC2)."""
import numpy as np


def rumbo(mapa, deg_px, paso, ancho_img):
    if mapa is None or mapa.max() <= 0:
        return np.nan
    m = np.where(mapa >= 0.5 * mapa.max(), mapa, 0.0)
    cols = paso // 2 + paso * np.arange(mapa.shape[1])
    az = (ancho_img / 2 - (cols + 0.5)) * deg_px
    return float((m.sum(axis=0) * az).sum() / m.sum())


def compuerta(beta, eps, dt=1 / 30, ventana=0.3):
    k = int(round(ventana / dt))
    a = np.abs(np.asarray(beta))
    bloq = np.zeros(len(a), bool)
    if eps is None or not np.isfinite(eps):
        return bloq
    deriva = (a[k:] - a[:-k]) / ventana
    ok = np.isfinite(deriva)
    bloq[k:][ok] = deriva[ok] > eps
    return bloq
