RAIZ = __import__("pathlib").Path(__file__).resolve().parents[1]
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.labelsize": 7, "ytick.labelsize": 7})
d = json.load(open(RAIZ / "resultados/simulacion/analisis.json"))
EST = {"M1": ("M1 LGMD", "#6B7B8C", "-"), "M2_1.25": ("M2 1,25°", "#1D6BB0", "-"), "M2_5.0": ("M2 5°", "#7447B0", "-"),
       "M2u_5.0": ("M2 5°, pesos uniformes", "#7447B0", (0, (3, 2))), "M2.1_5.0": ("M2.1 5°", "#C23A2C", "-"), "tau": ("Línea base τ", "#2E7D4F", "-")}
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.55), gridspec_kw={"wspace": 0.28})
a = ax[0]
for m, (lab, c, ls) in EST.items():
    r = d["roc"][m]; a.step(r["fa"], r["det"], where="post", c=c, ls=ls, lw=1.2, label=lab)
    s = d["congelados"].get(m) or d["nuevos"].get(m); a.plot(s["fa"], s["det"], "o", c=c, ms=4, mec="white", mew=0.6, zorder=5)
a.axvline(0.05, c="#9AA4AD", lw=0.6, ls=":"); a.text(0.052, 0.04, "5 %", fontsize=6.5, color="#566370")
a.set_xlim(0, 0.3); a.set_ylim(0, 1); a.set_xlabel("tasa de falsas alarmas (corridas sin choque)"); a.set_ylabel("fracción de colisiones detectadas")
a.set_title("A  Detección frente a falsas alarmas", loc="left", fontsize=8)
a.legend(frameon=False, fontsize=6.3, loc="lower right", handlelength=2.2)
b = ax[1]; t = np.array(d["curva_t"])
for m, (lab, c, ls) in EST.items():
    b.plot(t, d["curva_ant"][m], c=c, ls=ls, lw=1.2)
b.axvline(0.5, c="#9AA4AD", lw=0.6, ls=":"); b.axvline(1.0, c="#9AA4AD", lw=0.6, ls=":")
b.set_xlim(0, 1.5); b.set_ylim(0, 1); b.set_xlabel("anticipación mínima t (s)"); b.set_ylabel("colisiones detectadas con ≥ t (de 120)")
b.set_title("B  Anticipación sobre todas las colisiones", loc="left", fontsize=8)
fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.17)
fig.savefig(RAIZ / "resultados/simulacion/fig_simulacion.png", dpi=600)
