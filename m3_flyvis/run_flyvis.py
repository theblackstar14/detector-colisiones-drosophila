"""
Modelo 3: frente visual de flyvis (Lappalainen et al. 2024, red preentrenada
restringida por conectoma) + capa LPLC2 / LC4 / fibra gigante con pesos FlyWire.

Reemplaza el correlador T4/T5 simplificado del modelo 2 por la red de flyvis:
  escena -> retina hexagonal (721 omatidios, 5.8°) -> 65 tipos celulares del
  lóbulo óptico -> T4a-d / T5a-d -> LPLC2 ; T2/TmY3/Tm4/Tm2 -> LC4 -> fibra gigante

Uso (Windows, PowerShell, desde la carpeta mosca_lgmd):
    python run_flyvis.py prueba
    python run_flyvis.py completo

'prueba'   : autoverificaciones (geometría, direcciones T4/T5, estímulos de nivel 1,
             normalización, tiempo por corrida). ~5-10 min. OBLIGATORIO antes de 'completo'.
'completo' : 280 corridas (mismas escenas y semillas que M1/M2) + evaluación.
             Se puede interrumpir y retomar: guarda avance por escenario.
"""
import os, sys, json, time, zlib, pickle, csv

# --- log propio, baja prioridad y sin suspensión (antes de importar flyvis) ---------
_MODO = sys.argv[1] if len(sys.argv) > 1 else "prueba"
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), exist_ok=True)
_LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"log_{_MODO}.txt"),
            "a", encoding="utf-8")


class _Tee:
    def __init__(self, a, b): self.a, self.b = a, b
    def write(self, x):
        try: self.a.write(x)
        except UnicodeEncodeError: self.a.write(x.encode("ascii", "replace").decode())
        self.b.write(x); self.b.flush()
    def flush(self): self.a.flush(); self.b.flush()
    def isatty(self): return False


sys.stdout, sys.stderr = _Tee(sys.stdout, _LOG), _Tee(sys.stderr, _LOG)
try:
    if os.name == "nt":
        import ctypes
        _k = ctypes.windll.kernel32
        _k.SetPriorityClass(_k.GetCurrentProcess(), 0x00004000)   # prioridad "por debajo de lo normal"
        _k.SetThreadExecutionState(0x80000000 | 0x00000001)      # no suspender mientras corre
    else:
        os.nice(10)
except Exception:
    pass
if os.name == "nt":                       # solo Windows (ver parche_datamate.py)
    os.environ.setdefault("FLYVIS_ROOT_DIR", r"C:\flyvis-data")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import torch
import flyvis
from flyvis.datasets.rendering.eye import BoxEye

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim.world import ESCENARIOS
from sim.retina import render as render96, CIELO, PISTA, H_CAM
from models.lgmd import LGMD, ParamsLGMD
from models.lplc2 import ViaLooming, ParamsLPLC2, pesos_conectoma
from eval.multi import NORM_LPLC2, NORM_LC4, calibrar, evaluar, resumen

RES = "results"
os.makedirs(RES, exist_ok=True)
torch.set_num_threads(os.cpu_count() or 1)
torch.set_grad_enabled(False)

# ------------------------------------------------------------------ constantes
EXTENT, KSIZE, OMM_DEG = 15, 13, 5.8      # retina de flyvis
DEG_PX = OMM_DEG / KSIZE                  # ~0.446 °/px en el plano de render
DT_CAM = 1 / 30                           # cámara (igual que M1/M2)
SUB = 2                                   # subpasos por cuadro -> dt red = 1/60 s
DT_NET = DT_CAM / SUB
RF_LPLC2, RF_LC4 = 30.0, 20.0             # grados (igual que M2)
# Entradas a LC4 en FlyWire v783 (sinapsis; ver models/extraer_conectoma.py)
LC4_IN = {"T2": 22194, "TmY3": 19737, "Tm4": 14234, "Tm2": 12409}
WARMUP = 3

W = pesos_conectoma()
eye = BoxEye(extent=EXTENT, kernel_size=KSIZE)
FH, FW = [int(x) for x in eye.min_frame_size.tolist()]


# ------------------------------------------------------------------ red
SEL_MIN = 1.5                               # criterio a priori de selectividad direccional
MODELOS_ENSAMBLE = [f"flow/0000/{i:03d}" for i in range(10)]


def cargar_red(nombre="flow/0000/000"):
    nv = flyvis.NetworkView(nombre)
    net = nv.init_network()
    net.eval()
    nodes = net.connectome.nodes
    tipos = [t.decode() if isinstance(t, bytes) else str(t) for t in net.connectome.unique_cell_types[:]]
    u_all, v_all = nodes.u[:], nodes.v[:]
    idx = {t: np.asarray(nodes.layer_index[t][:]) for t in tipos}
    uv = {t: (u_all[idx[t]], v_all[idx[t]]) for t in tipos}
    return net, tipos, idx, uv


def uv_a_grados(u, v):
    """Posición angular (x a la derecha, y hacia abajo) según la geometría de BoxEye."""
    y = KSIZE * (u + v / 2.0) * DEG_PX
    x = KSIZE * v * DEG_PX
    return x, y


PRE_S = 1.0   # s de adaptación al primer cuadro antes de cada simulación


def simular(net, frames01, reposo=None, relativo=True):
    """frames01: (n, FH, FW) en [0,1] a 30 fps.
    La red ve primero el cuadro inicial fijo durante PRE_S segundos (adaptación, evita
    el transitorio de encendido) y la actividad se expresa relativa a ese estado adaptado.
    Devuelve (n, n_nodos), un valor al final de cada cuadro. 'reposo' se ignora (compatibilidad)."""
    n_pre = int(round(PRE_S / DT_CAM))
    todo = np.concatenate([np.repeat(frames01[:1], n_pre, axis=0), frames01])
    seq = np.repeat(todo, SUB, axis=0)[None].astype(np.float32)
    hexs = eye(torch.tensor(seq))
    act = net.simulate(hexs, dt=DT_NET).cpu().numpy()[0][SUB - 1::SUB]
    base = act[n_pre - 1]
    act = act[n_pre:]
    return act - base if relativo else act


def reposo_gris(net):
    gris = np.full((int(1.0 / DT_CAM), FH, FW), 0.5, np.float32)
    return simular(net, gris, relativo=False)[-1]


# ------------------------------------------------------------------ render ancho
def render_ancho(esc, t, rng=None, ruido=2.0, ss=2):
    """Misma escena que sim/retina.py pero con el campo visual de la retina de flyvis (~±87°)."""
    hs, ws = FH * ss, FW * ss
    fov_h, fov_v = np.deg2rad(FW * DEG_PX), np.deg2rad(FH * DEG_PX)
    img = np.empty((hs, ws), np.float32)
    img[: hs // 2] = CIELO; img[hs // 2:] = PISTA
    xe, ye, he = esc.ego.traj(t)
    Le = esc.ego.dims[0]
    cam = np.array([xe + Le / 2 * np.cos(he), ye + Le / 2 * np.sin(he)])
    c, s = np.cos(-he), np.sin(-he)
    R = np.array([[c, -s], [s, c]])
    objs = []
    for o in esc.otros:
        loc = (o.esquinas(t) - cam) @ R.T
        dl = loc[:, 0] > 0.3
        if not dl.any():
            continue
        loc = loc.copy(); loc[~dl, 0] = 0.3
        az = np.arctan2(loc[:, 1], loc[:, 0])
        objs.append((loc[:, 0].min(), az.min(), az.max(), o))
    for prof, az0, az1, o in sorted(objs, key=lambda z: -z[0]):
        et = np.arctan((o.dims[2] - H_CAM) / prof); eb = np.arctan(-H_CAM / prof)
        c0 = max(int(np.floor((fov_h / 2 - az1) / fov_h * ws)), 0)
        c1 = min(int(np.ceil((fov_h / 2 - az0) / fov_h * ws)), ws)
        r0 = max(int(np.floor((fov_v / 2 - et) / fov_v * hs)), 0)
        r1 = min(int(np.ceil((fov_v / 2 - eb) / fov_v * hs)), hs)
        if c1 > c0 and r1 > r0:
            img[r0:r1, c0:c1] = o.luminancia
    img = img.reshape(FH, ss, FW, ss).mean(axis=(1, 3))
    if ruido and rng is not None:
        img = img + rng.normal(0, ruido, img.shape)
    return np.clip(img, 0, 255) / 255.0


# ------------------------------------------------------------------ capa LPLC2/LC4/GF
class CapaEscape:
    def __init__(self, x, y, dir_t4, dir_t5, idx, norm_p=1.0, norm_c=1.0, w_inh=None):
        self.dir_t4, self.dir_t5, self.idx = dir_t4, dir_t5, idx
        self.w_inh = W["w_inh"] if w_inh is None else w_inh
        self.norm_p, self.norm_c = norm_p, norm_c
        dx, dy = x[None, :] - x[:, None], y[None, :] - y[:, None]   # [centro, celda]
        d = np.hypot(dx, dy)
        def cuña(mask):
            m = (mask & (d <= RF_LPLC2)).astype(np.float32)
            return m / np.maximum(m.sum(1, keepdims=True), 1)
        self.K = {"der": cuña((dx > 0) & (np.abs(dy) <= dx)), "izq": cuña((dx < 0) & (np.abs(dy) <= -dx)),
                  "abj": cuña((dy > 0) & (np.abs(dx) <= dy)), "arr": cuña((dy < 0) & (np.abs(dx) <= -dy))}
        m = (d <= RF_LC4).astype(np.float32)
        self.K_lc4 = m / m.sum(1, keepdims=True)
        tot = sum(LC4_IN.values())
        self.w_lc4 = {k: v / tot for k, v in LC4_IN.items()}

    def paso(self, a, usar_lc4=True):
        """a: actividad relativa al reposo (n_nodos,) de UN cuadro."""
        op = {"der": "izq", "izq": "der", "arr": "abj", "abj": "arr"}
        r = lambda t: np.maximum(a[self.idx[t]], 0)
        def rl(ts):
            ts = [ts] if isinstance(ts, str) else ts
            return np.mean([r(t) for t in ts], axis=0) if ts else np.zeros(len(self.idx["T4a"]), np.float32)
        M = {d: W["w_t4"] * rl(self.dir_t4[d]) + W["w_t5"] * rl(self.dir_t5[d]) for d in op}
        br = {d: np.maximum(W["w_brazo"][d] * (self.K[d] @ M[d] - self.w_inh * (self.K[d] @ M[op[d]])), 0) for d in op}
        u = np.sqrt(br["der"] * br["izq"]) + np.sqrt(br["arr"] * br["abj"])
        lplc2 = float(u.sum()) / self.norm_p
        lc4 = 0.0
        if usar_lc4:
            e = sum(w * r(t) for t, w in self.w_lc4.items() if t in self.idx)
            lc4 = float(((self.K_lc4 @ e) ** 2).sum()) / self.norm_c
        return W["w_gf_lplc2"] * lplc2 + W["w_gf_lc4"] * lc4, lplc2, lc4


# ------------------------------------------------------------------ estímulos de prueba
def coords_px():
    yy, xx = np.mgrid[0:FH, 0:FW]
    return (xx - FW // 2) * DEG_PX, (yy - FH // 2) * DEG_PX


def disco(r_deg, cx=0.0, cy=0.0, contraste=1.0):
    X, Y = coords_px()
    m = ((X - cx) ** 2 + (Y - cy) ** 2 <= r_deg ** 2).astype(np.float32)
    return 0.65 - 0.5 * contraste * m


def estimulos_nivel1():
    tc, rv = 2.0, 0.1
    ts = np.arange(0, tc - 0.02, DT_CAM)
    th = np.minimum(np.degrees(2 * np.arctan(rv / (tc - ts))), 100)
    loom = np.stack([disco(x / 2) for x in th])
    xs = np.arange(-80, 80, 60 * DT_CAM)
    tras = np.stack([disco(6, cx=x) for x in xs])
    n = len(ts)
    osc = np.stack([disco(12, contraste=min(1, max(0, (i - n / 3) / (n / 3)))) for i in range(n)])
    return {"looming": loom, "alejamiento": loom[::-1], "traslacion": tras, "oscurecimiento": osc}


def rejilla(direccion, periodo=30.0, vel=45.0, dur=1.5):
    X, Y = coords_px()
    coord = {"der": X, "izq": -X, "abj": Y, "arr": -Y}[direccion]
    ts = np.arange(0, dur, DT_CAM)
    return np.stack([0.5 + 0.4 * np.sign(np.sin(2 * np.pi * (coord - vel * t) / periodo)) for t in ts]).astype(np.float32)


# ------------------------------------------------------------------ PRUEBA
def prueba():
    t0 = time.time()
    print(f"flyvis {flyvis.__version__} | raíz {flyvis.root_dir} | cuadro {FH}x{FW} px ({DEG_PX:.3f}°/px)")
    net, tipos, idx, uv = cargar_red()
    faltan = [t for t in ["R1", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"] + list(LC4_IN) if t not in idx]
    print("Tipos celulares:", len(tipos), "| faltan:", faltan or "ninguno")
    reposo = reposo_gris(net)

    # 1) geometría: un punto de luz en omatidios conocidos debe activar el R1 correcto
    ur, vr = uv["R1"]
    x_r1, y_r1 = uv_a_grados(ur, vr)
    interior = np.nonzero((np.abs(ur) <= 9) & (np.abs(vr) <= 9) & (np.abs(ur + vr) <= 9))[0]
    ks = interior[np.linspace(0, len(interior) - 1, 5).astype(int)]
    ok, detalle = 0, []
    X, Y = coords_px()
    for k in ks:
        f = np.full((8, FH, FW), 0.2, np.float32)
        f[2:, (np.abs(X - x_r1[k]) < 2.0) & (np.abs(Y - y_r1[k]) < 2.0)] = 1.0
        a = simular(net, f)[-1][idx["R1"]]
        j = int(np.argmax(np.abs(a)))
        du, dv = int(ur[j] - ur[k]), int(vr[j] - vr[k])
        dist = (abs(du) + abs(dv) + abs(du + dv)) // 2
        ok += int(dist <= 1); detalle.append(f"(u,v)=({ur[k]},{vr[k]})->dist {dist}")
    print(f"[1] Geometría retina: {ok}/5 dentro de 1 omatidio", "OK" if ok >= 4 else "<-- REVISAR", "|", "; ".join(detalle))

    # 2) direcciones de T4/T5 y 3) nivel 1, para cada red del ensamble y cada fuerza de inhibición LPi
    subt = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
    est = estimulos_nivel1()

    def direccional(net, idx, uv):
        resp = {}
        x, y = uv_a_grados(*uv["T4a"])
        centro = np.hypot(x, y) <= 25
        for d in ["der", "izq", "arr", "abj"]:
            a = simular(net, rejilla(d))[15:]
            resp[d] = {t: float(np.maximum(a[:, idx[t]][:, centro], 0).mean()) for t in subt}
        pref = {t: max(resp, key=lambda d: resp[d][t]) for t in subt}
        sel = {t: max(resp[d][t] for d in resp) / (sum(resp[d][t] for d in resp) / 4 + 1e-9) for t in subt}
        buenos = [t for t in subt if sel[t] >= SEL_MIN]
        dir_t4 = {d: [t for t in buenos if t.startswith("T4") and pref[t] == d] for d in resp}
        dir_t5 = {d: [t for t in buenos if t.startswith("T5") and pref[t] == d] for d in resp}
        return resp, pref, sel, dir_t4, dir_t5

    def nivel1(acts, capa):
        pk = {}
        for n, a in acts.items():
            r = np.array([capa.paso(ai) for ai in a])
            pk[n] = r.max(axis=0)          # (GF, LPLC2, LC4)
        capa.norm_p, capa.norm_c = max(pk["looming"][1], 1e-9), max(pk["looming"][2], 1e-9)
        rel = {n: dict(GF=pk[n][0] / max(pk["looming"][0], 1e-9), LPLC2=pk[n][1] / capa.norm_p,
                       LC4=pk[n][2] / max(pk["looming"][2], 1e-9)) for n in pk}
        S = 1 - max(rel[n]["LPLC2"] for n in ("alejamiento", "traslacion", "oscurecimiento"))
        S = S if pk["looming"][1] > 1e-9 else -np.inf     # sin respuesta al looming = inválido
        return rel, S

    W_INH_OPC = sorted({round(W["w_inh"], 3), 0.5, 1.0})
    candidatos = []
    x, y = uv_a_grados(*uv["T4a"])
    for nombre in MODELOS_ENSAMBLE:
        net_i, _, idx_i, _ = cargar_red(nombre) if nombre != "flow/0000/000" else (net, tipos, idx, uv)
        resp, pref, sel, d4, d5 = direccional(net_i, idx_i, uv)
        vacias = [d for d in d4 if not d4[d] and not d5[d]]
        acts = {n: simular(net_i, fr) for n, fr in est.items()}
        fila = []
        for wi in W_INH_OPC:
            capa_i = CapaEscape(x, y, d4, d5, idx_i, w_inh=wi)
            rel, S = nivel1(acts, capa_i)
            candidatos.append(dict(modelo=nombre, w_inh=wi, S=S, rel=rel, vacias=vacias, resp=resp, pref=pref,
                                   sel=sel, dir_t4=d4, dir_t5=d5, norm_p=capa_i.norm_p, norm_c=capa_i.norm_c))
            fila.append(f"w_inh={wi}: S={S:.2f} (alej {rel['alejamiento']['LPLC2']:.2f}, tras {rel['traslacion']['LPLC2']:.2f})")
        print(f"    {nombre} | selectivos {sum(v >= SEL_MIN for v in sel.values())}/8 | sin canal: {vacias or '-'} | " + " | ".join(fila))

    validos = [c for c in candidatos if not c["vacias"]] or candidatos
    buenos = [c for c in validos if c["S"] >= 0.7]
    if buenos:
        elegido = min(buenos, key=lambda c: (abs(c["w_inh"] - W["w_inh"]), -c["S"]))
        regla = "S>=0.7 con inhibición más cercana al conectoma"
    else:
        elegido = max(validos, key=lambda c: c["S"])
        regla = "ninguna alcanza S>=0.7: se toma la de mayor S  <-- AVISAR"
    mejor = elegido["modelo"]
    print(f"[2-3] Elegido: {mejor}, w_inh={elegido['w_inh']} (conectoma={W['w_inh']:.3f}) | regla: {regla}")
    print("    Preferencias:", elegido["pref"], "| excluidos:", [t for t in subt if elegido["sel"][t] < SEL_MIN] or "ninguno")
    print("    Nivel 1 (pico relativo al looming):")
    for n, v in elegido["rel"].items():
        print(f"      {n:15s} LPLC2={v['LPLC2']:.2f}  LC4={v['LC4']:.2f}  GF={v['GF']:.2f}")

    net, tipos, idx, uv = cargar_red(mejor)
    capa = CapaEscape(x, y, elegido["dir_t4"], elegido["dir_t5"], idx, elegido["norm_p"], elegido["norm_c"], w_inh=elegido["w_inh"])

    # 4) tiempo por corrida
    t1 = time.time(); c = correr_escenario(net, idx, capa, "frontal_detenido", 1)
    dur = time.time() - t1
    print(f"[4] Una corrida ({len(c['t'])} cuadros): {dur:.1f} s -> 280 corridas ≈ {dur*280/60*1.3:.0f} min")

    json.dump(dict(modelo=mejor, w_inh=elegido["w_inh"], w_inh_conectoma=W["w_inh"], regla=regla,
                   excluidos=[t for t in subt if elegido["sel"][t] < SEL_MIN], selectividad=elegido["sel"],
                   dir_t4=elegido["dir_t4"], dir_t5=elegido["dir_t5"], pref_subtipo=elegido["pref"],
                   norm_lplc2=elegido["norm_p"], norm_lc4=elegido["norm_c"], nivel1=elegido["rel"], S=elegido["S"],
                   barrido=[{k: c[k] for k in ("modelo", "w_inh", "S", "vacias")} for c in candidatos],
                   geometria_ok=ok, version_script="1.3-ensamble-inhibicion"),
              open(f"{RES}/flyvis_calibracion.json", "w"), indent=1, default=float)
    print(f"Listo en {time.time()-t0:.0f} s. Guardado {RES}/flyvis_calibracion.json. Envíame toda esta salida.")


def correr_escenario(net, idx, capa, nombre, semilla, reposo=None):
    rng = np.random.default_rng(semilla * 1000 + zlib.crc32(nombre.encode()) % 997)   # = M1/M2
    rng2 = np.random.default_rng(10**6 + semilla * 1000 + zlib.crc32(nombre.encode()) % 997)
    esc = ESCENARIOS[nombre](rng)
    tc = esc.tiempo_colision()
    ts = np.arange(0, tc if tc is not None else esc.duracion, esc.dt)
    m1 = LGMD(ParamsLGMD(T_de=5))
    base = dict(norm_lplc2=NORM_LPLC2, norm_lc4=NORM_LC4)
    m2 = ViaLooming(ParamsLPLC2(**base)); m2s = ViaLooming(ParamsLPLC2(usar_lc4=False, **base))
    s = {k: [] for k in ["M1_LGMD", "M2_completo", "M2_sin_LC4", "M3_flyvis_completo", "M3_flyvis_sin_LC4"]}
    fly = []
    for i, t in enumerate(ts):
        img = render96(esc, t, rng=rng)                  # idéntico a M1/M2 (control de reproducibilidad)
        s["M1_LGMD"].append(m1.paso(img)[0]); s["M2_completo"].append(m2.paso(img)[0]); s["M2_sin_LC4"].append(m2s.paso(img)[0])
        fly.append(render_ancho(esc, t, rng=rng2))
    a = simular(net, np.stack(fly))
    for i, ai in enumerate(a):
        gf, lp, lc = capa.paso(ai)
        ok = i >= WARMUP
        s["M3_flyvis_completo"].append(gf if ok else 0.0)
        s["M3_flyvis_sin_LC4"].append(W["w_gf_lplc2"] * lp if ok else 0.0)
    return dict(escenario=nombre, semilla=semilla, tc=tc, t=ts,
                senal={k: np.array(v) for k, v in s.items()},
                ffi={k: np.zeros(len(ts), bool) for k in s})


def completo():
    cal = json.load(open(f"{RES}/flyvis_calibracion.json"))
    net, tipos, idx, uv = cargar_red(cal.get("modelo", "flow/0000/000"))
    print("Red:", cal.get("modelo"), "| excluidos:", cal.get("excluidos"))
    x, y = uv_a_grados(*uv["T4a"])
    capa = CapaEscape(x, y, cal["dir_t4"], cal["dir_t5"], idx, cal["norm_lplc2"], cal["norm_lc4"], w_inh=cal.get("w_inh"))
    ruta = f"{RES}/corridas_flyvis.pkl"
    corr = pickle.load(open(ruta, "rb")) if os.path.exists(ruta) else []
    hechos = {(c["escenario"], c["semilla"]) for c in corr}
    t0 = time.time()
    for n in ESCENARIOS:
        for sd in range(40):
            if (n, sd) in hechos:
                continue
            corr.append(correr_escenario(net, idx, capa, n, sd))
        pickle.dump(corr, open(ruta, "wb"))
        print(f"{n} listo ({len(corr)}/280, {time.time()-t0:.0f} s)", flush=True)

    cal_c = [c for c in corr if c["semilla"] % 2 == 0]; test = [c for c in corr if c["semilla"] % 2 == 1]
    with open(f"{RES}/tabla_M3_flyvis.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "escenario", "deteccion", "antic_>=0.5s", "antic_>=1s", "falsos_pos", "anticip_mediana_s", "umbral"])
        for m in corr[0]["senal"]:
            T, _ = calibrar(cal_c, m)
            res = evaluar(test, m, T)
            g = resumen(res)
            print(f"{m:22s} det={g['deteccion']:.2f} >=0.5s={g['antic_05s']:.2f} FP={g['falsos_pos']:.3f} antic_med={g['anticip_mediana']:.2f}")
            for n in ESCENARIOS:
                r = resumen([x for x in res if x["escenario"] == n])
                w.writerow([m, n] + [round(v, 3) if v == v else "" for v in r.values()] + [T])
            w.writerow([m, "GLOBAL"] + [round(v, 3) if v == v else "" for v in g.values()] + [T])
    print(f"Guardado {RES}/tabla_M3_flyvis.csv. Envíame la tabla, flyvis_calibracion.json y corridas_flyvis.pkl.")


def final():
    """Conjunto de prueba NUEVO (semillas 40-79) con M3 CONGELADO (congelado_M3.json).
    No recalibra nada: solo genera señales; los umbrales congelados se aplican en el análisis."""
    base = os.path.dirname(os.path.abspath(__file__))
    cal = json.load(open(os.path.join(base, "congelado_M3.json")))
    net, tipos, idx, uv = cargar_red(cal["modelo"])
    print("FINAL | red:", cal["modelo"], "| w_inh:", cal["w_inh"], "| excluidos:", cal["excluidos"])
    x, y = uv_a_grados(*uv["T4a"])
    capa = CapaEscape(x, y, cal["dir_t4"], cal["dir_t5"], idx, cal["norm_lplc2"], cal["norm_lc4"], w_inh=cal["w_inh"])
    ruta = f"{RES}/corridas_final_flyvis.pkl"
    corr = pickle.load(open(ruta, "rb")) if os.path.exists(ruta) else []
    hechos = {(c["escenario"], c["semilla"]) for c in corr}
    t0 = time.time()
    for n in ESCENARIOS:
        for sd in range(40, 80):
            if (n, sd) in hechos:
                continue
            corr.append(correr_escenario(net, idx, capa, n, sd))
        pickle.dump(corr, open(ruta, "wb"))
        print(f"{n} listo ({len(corr)}/280, {time.time()-t0:.0f} s)", flush=True)
    print(f"Guardado {ruta}.")


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "prueba"
    {"prueba": prueba, "completo": completo, "final": final}[modo]()
