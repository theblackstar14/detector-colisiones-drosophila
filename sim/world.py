"""Mundo 2D (vista cenital) para escenarios de tránsito. x adelante, y izquierda; Perú circula por la derecha."""
from dataclasses import dataclass, field
import numpy as np

KMH = 1 / 3.6
TIPOS = {"auto": (4.5, 1.8, 1.5), "combi": (5.5, 2.0, 2.3), "bus": (12.0, 2.5, 3.2), "moto": (2.0, 0.8, 1.4), "poste": (0.3, 0.3, 4.0)}


@dataclass
class Vehiculo:
    nombre: str
    tipo: str
    traj: callable
    luminancia: float = 50.0
    estatico: bool = False

    @property
    def dims(self):
        return TIPOS[self.tipo]

    def esquinas(self, t):
        x, y, h = self.traj(t)
        L, W, _ = self.dims
        c, s = np.cos(h), np.sin(h)
        loc = np.array([[L/2, W/2], [L/2, -W/2], [-L/2, -W/2], [-L/2, W/2]])
        R = np.array([[c, -s], [s, c]])
        return loc @ R.T + np.array([x, y])


def se_superponen(a, b):
    for poly in (a, b):
        for i in range(4):
            eje = poly[(i + 1) % 4] - poly[i]
            eje = np.array([-eje[1], eje[0]])
            pa, pb = a @ eje, b @ eje
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


@dataclass
class Escenario:
    nombre: str
    ego: Vehiculo
    otros: list
    duracion: float = 8.0
    dt: float = 1 / 30
    meta: dict = field(default_factory=dict)

    def tiempo_colision(self):
        for t in np.arange(0, self.duracion, self.dt / 4):
            e = self.ego.esquinas(t)
            for o in self.otros:
                if not o.estatico and se_superponen(e, o.esquinas(t)):
                    return float(t)
        return None


def vel_constante(x0, y0, vx, vy):
    h = np.arctan2(vy, vx) if (vx or vy) else 0.0
    return lambda t: (x0 + vx * t, y0 + vy * t, h)


def frenado(x0, y0, v0, t_freno, decel):
    t_stop = t_freno + v0 / decel
    def f(t):
        if t < t_freno:
            return (x0 + v0 * t, y0, 0.0)
        tb = min(t, t_stop) - t_freno
        return (x0 + v0 * t_freno + v0 * tb - 0.5 * decel * tb**2, y0, 0.0)
    return f


def cambio_carril(x0, y0, y1, v, t1, dur):
    def f(t):
        x = x0 + v * t
        if t <= t1:
            return (x, y0, 0.0)
        if t >= t1 + dur:
            return (x, y1, 0.0)
        u = (t - t1) / dur
        y = y0 + (y1 - y0) * (1 - np.cos(np.pi * u)) / 2
        vy = (y1 - y0) * np.pi / (2 * dur) * np.sin(np.pi * u)
        return (x, y, np.arctan2(vy, v))
    return f


def _lum(rng):
    return rng.uniform(20, 80) if rng.random() < 0.6 else rng.uniform(180, 235)


def _ego(v):
    return Vehiculo("ego", "combi", vel_constante(0, 0, v, 0))


def _postes(rng, x_max=150):
    out = []
    for lado in (-6.5, 9.0):
        x = rng.uniform(5, 15)
        while x < x_max:
            out.append(Vehiculo("poste", "poste", vel_constante(x, lado, 0, 0), luminancia=rng.uniform(40, 90), estatico=True))
            x += rng.uniform(15, 25)
    return out


def frontal_detenido(rng):
    v = rng.uniform(30, 50) * KMH
    tipo = rng.choice(["combi", "bus", "auto"])
    L_o = TIPOS[tipo][0]
    tc = rng.uniform(3.5, 6.0)
    x0 = v * tc + (L_o + TIPOS["combi"][0]) / 2
    o = Vehiculo("detenido", tipo, vel_constante(x0, rng.uniform(-0.4, 0.4), 0, 0), _lum(rng))
    return Escenario("frontal_detenido", _ego(v), [o] + _postes(rng))


def frenado_brusco(rng):
    v = rng.uniform(30, 50) * KMH
    tipo = rng.choice(["combi", "auto", "bus"])
    gap = rng.uniform(10, 20)
    x0 = gap + (TIPOS[tipo][0] + TIPOS["combi"][0]) / 2
    o = Vehiculo("lider", tipo, frenado(x0, 0, v, rng.uniform(1.0, 2.0), rng.uniform(5, 8)), _lum(rng))
    return Escenario("frenado_brusco", _ego(v), [o] + _postes(rng))


def cierre_peligroso(rng):
    v = rng.uniform(35, 50) * KMH
    tipo = rng.choice(["combi", "auto", "moto"])
    v_o = v - rng.uniform(12, 20) * KMH
    x0 = rng.uniform(6, 11) + (TIPOS[tipo][0] + TIPOS["combi"][0]) / 2
    o = Vehiculo("cierre", tipo, cambio_carril(x0, -3.3, 0, v_o, rng.uniform(0.5, 1.2), 1.4), _lum(rng))
    return Escenario("cierre_peligroso", _ego(v), [o] + _postes(rng))


def cruce_sin_riesgo(rng):
    v = rng.uniform(30, 45) * KMH
    tipo = rng.choice(["auto", "combi", "moto"])
    L_o, W_o, _ = TIPOS[tipo]
    v_o = rng.uniform(20, 35) * KMH
    x_c = rng.uniform(35, 55)
    t_llega = (x_c - W_o / 2 - TIPOS["combi"][0] / 2) / v
    margen = rng.uniform(1.2, 2.5)
    t_despeja = t_llega - margen
    t_centro = t_despeja - (L_o / 2 + TIPOS["combi"][1] / 2) / v_o
    o = Vehiculo("cruce", tipo, vel_constante(x_c, -v_o * t_centro, 0, v_o), _lum(rng))
    return Escenario("cruce_sin_riesgo", _ego(v), [o] + _postes(rng))


def contrasentido_adyacente(rng):
    v = rng.uniform(30, 45) * KMH
    tipo = rng.choice(["auto", "combi", "bus"])
    v_o = rng.uniform(30, 55) * KMH
    o = Vehiculo("contrario", tipo, vel_constante(rng.uniform(70, 110), rng.uniform(3.2, 3.8), -v_o, 0), _lum(rng))
    return Escenario("contrasentido_adyacente", _ego(v), [o] + _postes(rng))


def seguimiento(rng):
    v = rng.uniform(30, 45) * KMH
    tipo = rng.choice(["auto", "combi", "bus"])
    x0 = rng.uniform(10, 25) + (TIPOS[tipo][0] + TIPOS["combi"][0]) / 2
    o = Vehiculo("lider", tipo, vel_constante(x0, 0, v + rng.uniform(0, 6) * KMH, 0), _lum(rng))
    return Escenario("seguimiento", _ego(v), [o] + _postes(rng))


def cierre_seguro(rng):
    v = rng.uniform(30, 45) * KMH
    tipo = rng.choice(["combi", "auto", "moto"])
    x0 = rng.uniform(14, 22) + (TIPOS[tipo][0] + TIPOS["combi"][0]) / 2
    o = Vehiculo("cierre", tipo, cambio_carril(x0, -3.3, 0, v + rng.uniform(3, 10) * KMH, rng.uniform(0.5, 1.5), 1.6), _lum(rng))
    return Escenario("cierre_seguro", _ego(v), [o] + _postes(rng))


ESCENARIOS = {"frontal_detenido": frontal_detenido, "frenado_brusco": frenado_brusco, "cierre_peligroso": cierre_peligroso,
              "cruce_sin_riesgo": cruce_sin_riesgo, "contrasentido_adyacente": contrasentido_adyacente,
              "seguimiento": seguimiento, "cierre_seguro": cierre_seguro}
