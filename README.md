# Detector de colisiones bioinspirado en el circuito de escape de *Drosophila melanogaster*

Código, resultados y anotaciones del artículo:

> L. E. Acosta Medina, J. Avendaño Castro, S. Llacsahuache Ortiz, G. Calderón Chávez y J. Alcalde Tirado, «A Bio-Inspired Model of the *Drosophila melanogaster* Escape Circuit for Collision Detection: An Evaluation in Simulation and Real Driving Video from Lima», 6th International Computer Sciences and Informatics Conference (ICSIC 2026), Amman, Jordania (aceptado).

**Versión 1.1.0** (revisión tras los comentarios de los revisores): añade la segunda prueba piloto con video de un bus del Metropolitano, la prueba de equivalencia de pesos, los IC por tramo y el análisis de tramos comunes de la prueba semisintética, y la comprobación de selectividad con pesos uniformes. Ver «Cambios en la versión 1.1.0».

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23485345.svg)](https://doi.org/10.5281/zenodo.23485345)

DOI de este repositorio: https://doi.org/10.5281/zenodo.23485345

*English summary:* code and data for a collision detector built on the architecture of the Drosophila LPLC2–giant fiber escape pathway, with weights derived from the FlyWire v783 connectome, compared with an LGMD model, an optical-flow time-to-contact baseline and a flyvis-based variant in a 2D traffic simulation of Lima, real driving video from a private car and from a Metropolitano bus cabin (Lima), and a semi-synthetic approach test.

## Estructura

| Carpeta | Contenido |
|---|---|
| `models/` | M1 (LGMD, `lgmd.py`), M2 (LPLC2 con pesos del conectoma, `lplc2.py`), regla de rumbo de M2.1 (`rumbo.py`) y recuentos de FlyWire v783 (`conectoma_flywire783.json`) |
| `sim/` | Simulador 2D: siete escenarios de tránsito (`world.py`) y retina de 96 × 32 px a 1,25°/px (`retina.py`) |
| `eval/` | Corrida de evaluación (`corrida_revision.py`), análisis estadístico (`analisis_revision.py`), figura de resultados (`fig_revision.py`), ablación de inhibición (`corrida_winh.py`, `analisis_winh.py`), comparación con falsas alarmas igualadas (`fa_igualada.py`), prueba de equivalencia (`equivalencia.py`) y comprobación con estímulos de laboratorio y pesos uniformes (`estimulos_uniformes.py`) |
| `video/` | Procesamiento de video real con umbrales congelados (`procesar_video.py`, `procesar_extra.py`), umbrales congelados y un video sintético de verificación |
| `semisintetica/` | Inserción de un vehículo con distancia conocida sobre tramos reales (`insertar.py`), corrida de la prueba (`correr_v2.py`) e IC por tramo y análisis de tramos comunes (`ic_tramos.py`) |
| `bus/` | Segunda prueba piloto: procesamiento por tomas del video del bus del Metropolitano (`procesar_bus.py`), tomas usadas y descartadas (`tomas.json`) y cortes de escena detectados |
| `doble_revision_bus/` | Clasificación ciega de los 24 eventos del bus por ambos revisores y consenso (`anotaciones_bus.csv`), kappa con bootstrap (`kappa_bus.py`, `kappa_bus.json`) |
| `m3_flyvis/` | Integración con el lóbulo óptico preentrenado de *flyvis* (M3) |
| `conectoma/` | Script que extrae de FlyWire v783 los recuentos de la Tabla I |
| `doble_revision/` | Anotaciones de la doble revisión ciega (`anotaciones_eventos.csv`), plantillas y cálculo del kappa con bootstrap |
| `resultados/` | Resultados usados en el artículo: simulación (incluida `equivalencia.json`), video del auto (`video_real/`, FOV 60°, 70° y 80°), video del bus (`video_bus/`) y prueba semisintética (incluido `ic_tramos.json`) |

## Instalación

Python 3.10 o superior:

    pip install -r requirements.txt

## Reproducir los resultados

**Simulación** (280 escenas de prueba, semillas 40–79; calibración con semillas pares 0–38). Unos 25 min con 2 núcleos:

    python eval/corrida_revision.py prueba      # verificación rápida: una corrida (frontal_detenido, semilla 71)
    python eval/corrida_revision.py             # las 420 corridas
    python eval/analisis_revision.py            # métricas, pruebas pareadas, ablación y curvas -> resultados/simulacion/analisis.json
    python eval/fig_revision.py                 # figura de resultados
    python eval/corrida_winh.py                 # ablación de la inhibición LPi (w_inh = 0; 0,237; 0,611; 1) en M2 a 5°, ~5 min
    python eval/analisis_winh.py                # -> resultados/simulacion/winh/analisis_winh.json (incluye McNemar de las permutaciones)
    python eval/fa_igualada.py                  # comparación post hoc con falsas alarmas igualadas (M2 1,25° frente a la línea base)
    python eval/equivalencia.py                 # TOST conectoma frente a pesos uniformes (±5 puntos, ±0,05 s) -> resultados/simulacion/equivalencia.json
    python eval/estimulos_uniformes.py          # selectividad ante estímulos de laboratorio con pesos del conectoma y uniformes (no requiere corridas)

La verificación rápida debe dar alarmas en los cuadros 152, 129, 140 y 134 para M1, M2 1,25°, M2 5° y M2.1 5°.

**Video real.** El video no se incluye porque pertenece a su autor: «Manejando ando en Lima Perú #60», canal G. Palmares, YouTube (https://youtu.be/YH5MjnOnscE). Descárgalo con una herramienta como `yt-dlp` y procésalo desde el minuto 2:05:

    python video/procesar_video.py surco.mp4 --fov 70 --inicio 125 --sin-video
    python video/procesar_extra.py surco.mp4 --fov 70 --inicio 125 --salida salida_extra

Para verificar la instalación sin el video:

    python video/procesar_video.py video/panel_prueba.mp4 --fov 70 --marcas "20:0,15:3.57,10:7.14,5:10.71,2:12.86" --sin-video

(debe dar anticipación de 0,92 s en M2 y 1,05 s en M2.1).

**Prueba semisintética** (requiere el video y los umbrales de `resultados/simulacion/umbrales_nuevos.json`):

    cd semisintetica && python correr_v2.py RUTA/surco.mp4
    python ic_tramos.py ../resultados/semisintetica/resultados_v2.csv   # IC por tramo, tramos comunes y comparación pareada

Una detección es la primera alarma emitida antes de que el objeto llegue a 1 m (t ≤ 3 + 24/v s); un tramo se excluye para un modelo si este alarma en el control sin objeto después del instante de aparición.

**Video del bus (segunda prueba piloto).** Video público «Recorriendo TODO el Metropolitano de Lima, Perú», canal Víctor de viaje, YouTube (https://www.youtube.com/watch?v=I7AMOVLiFow), no incluido. Se usan los tramos con vista de cabina 2:53–7:18 y 8:46–16:00, divididos en tomas continuas en los cortes de escena (`bus/tomas.json`; 48 tomas, 4 descartadas, 44 usadas, 9,5 min efectivos). Cada toma se procesa por separado con los modelos reiniciados, se ignoran las alarmas de sus primeros 2 s y se enmascara el 22 % superior del cuadro, donde el video superpone los nombres de las estaciones:

    python bus/procesar_bus.py bus/tomas.json CARPETA_CON_seg_a.mp4_Y_seg_b.mp4 --fov 70 --salida salida_bus

`seg_a.mp4` y `seg_b.mp4` son los dos tramos de cabina recortados del video (desde 2:53 y desde 8:46) a 640 × 360 px.

**Conectoma** (Tabla I): ver las instrucciones al inicio de `conectoma/extraer_conectoma.py`. Se suman todas las conexiones de `Connectivity_783.parquet`, sin umbral mínimo de sinapsis por par.

**Doble revisión:** `doble_revision/anotaciones_eventos.csv` tiene la causa asignada por cada revisor (antes del consenso) y la causa final. Las categorías son A (vehículo que invade la trayectoria), B (vehículo en carril contiguo), C (objetos fijos al borde), D (giro propio), E (cambio de luz), F (marcas en la pista) y G (otra). El kappa y su IC por bootstrap se calculan con `kappa_bootstrap.py` sobre el libro maestro.

**Doble revisión del bus:** `doble_revision_bus/anotaciones_bus.csv` tiene la causa, la confianza y las notas de cada revisor y la causa de consenso; en este video, un peatón, ciclista o moto que cruza delante del bus cuenta como A. `python doble_revision_bus/kappa_bus.py` calcula el kappa (0,75) y sus IC.

## Cambios en la versión 1.1.0

- Segunda prueba piloto con video de la cabina de un bus del Metropolitano: código, tomas, alarmas por minuto a 60°, 70° y 80° y eventos (`bus/`, `resultados/video_bus/`).
- Doble revisión ciega de los 24 eventos del bus, consenso y kappa (`doble_revision_bus/`).
- Prueba de equivalencia (TOST) entre pesos del conectoma y uniformes (`eval/equivalencia.py`).
- Prueba semisintética: IC por bootstrap sobre tramos y análisis pareado en los 4 tramos válidos para todos los modelos (`semisintetica/ic_tramos.py`).
- Comprobación de que la selectividad ante estímulos de laboratorio no cambia con pesos uniformes (`eval/estimulos_uniformes.py`).
- Referencia del artículo actualizada al título revisado.

## Privacidad

No se incluyen los videos originales ni cuadros de ellos. Los revisores aparecen como revisor 1 y revisor 2.

## Licencia

Código bajo licencia MIT (`LICENSE`). Resultados y anotaciones bajo CC BY 4.0. Los recuentos del conectoma derivan de FlyWire (Dorkenwald *et al.*, 2024; Schlegel *et al.*, 2024) y del modelo de Shiu *et al.* (2024); cita esas fuentes si los usas.

## Cómo citar

Usa el archivo `CITATION.cff` o el botón «Cite this repository» de GitHub.
