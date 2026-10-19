# Detector de colisiones bioinspirado en el circuito de escape de *Drosophila melanogaster*

Código, resultados y anotaciones del artículo:

> J. Avendaño Castro, S. Llacsahuache Ortiz, G. Calderón Chávez, J. Alcalde Tirado y L. E. Acosta Medina, «Modelo bioinspirado en el circuito de escape de *Drosophila melanogaster* para la detección de colisiones en el transporte público de Lima», enviado a ICSIC 2026.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23485345.svg)](https://doi.org/10.5281/zenodo.23485345)

DOI de este repositorio: https://doi.org/10.5281/zenodo.23485345

*English summary:* code and data for a collision detector built on the architecture of the Drosophila LPLC2–giant fiber escape pathway, with weights derived from the FlyWire v783 connectome, compared with an LGMD model, an optical-flow time-to-contact baseline and a flyvis-based variant in a 2D traffic simulation of Lima, real dashcam video and a semi-synthetic approach test.

## Estructura

| Carpeta | Contenido |
|---|---|
| `models/` | M1 (LGMD, `lgmd.py`), M2 (LPLC2 con pesos del conectoma, `lplc2.py`), regla de rumbo de M2.1 (`rumbo.py`) y recuentos de FlyWire v783 (`conectoma_flywire783.json`) |
| `sim/` | Simulador 2D: siete escenarios de tránsito (`world.py`) y retina de 96 × 32 px a 1,25°/px (`retina.py`) |
| `eval/` | Corrida de evaluación (`corrida_revision.py`), análisis estadístico (`analisis_revision.py`) y figura de resultados (`fig_revision.py`) |
| `video/` | Procesamiento de video real con umbrales congelados (`procesar_video.py`, `procesar_extra.py`), umbrales congelados y un video sintético de verificación |
| `semisintetica/` | Inserción de un vehículo con distancia conocida sobre tramos reales (`insertar.py`) y corrida de la prueba (`correr_v2.py`) |
| `m3_flyvis/` | Integración con el lóbulo óptico preentrenado de *flyvis* (M3) |
| `conectoma/` | Script que extrae de FlyWire v783 los recuentos de la Tabla I |
| `doble_revision/` | Anotaciones de la doble revisión ciega (`anotaciones_eventos.csv`), plantillas y cálculo del kappa con bootstrap |
| `resultados/` | Resultados usados en el artículo: simulación, video real (FOV 60°, 70° y 80°) y prueba semisintética |

## Instalación

Python 3.10 o superior:

    pip install -r requirements.txt

## Reproducir los resultados

**Simulación** (280 escenas de prueba, semillas 40–79; calibración con semillas pares 0–38). Unos 25 min con 2 núcleos:

    python eval/corrida_revision.py prueba      # verificación rápida: una corrida (frontal_detenido, semilla 71)
    python eval/corrida_revision.py             # las 420 corridas
    python eval/analisis_revision.py            # métricas, pruebas pareadas, ablación y curvas -> resultados/simulacion/analisis.json
    python eval/fig_revision.py                 # figura de resultados

La verificación rápida debe dar alarmas en los cuadros 152, 129, 140 y 134 para M1, M2 1,25°, M2 5° y M2.1 5°.

**Video real.** El video no se incluye porque pertenece a su autor: «Manejando ando en Lima Perú #60», canal G. Palmares, YouTube (https://youtu.be/YH5MjnOnscE). Descárgalo con una herramienta como `yt-dlp` y procésalo desde el minuto 2:05:

    python video/procesar_video.py surco.mp4 --fov 70 --inicio 125 --sin-video
    python video/procesar_extra.py surco.mp4 --fov 70 --inicio 125 --salida salida_extra

Para verificar la instalación sin el video:

    python video/procesar_video.py video/panel_prueba.mp4 --fov 70 --marcas "20:0,15:3.57,10:7.14,5:10.71,2:12.86" --sin-video

(debe dar anticipación de 0,92 s en M2 y 1,05 s en M2.1).

**Prueba semisintética** (requiere el video y los umbrales de `resultados/simulacion/umbrales_nuevos.json`):

    cd semisintetica && python correr_v2.py RUTA/surco.mp4

**Conectoma** (Tabla I): ver las instrucciones al inicio de `conectoma/extraer_conectoma.py`. Se suman todas las conexiones de `Connectivity_783.parquet`, sin umbral mínimo de sinapsis por par.

**Doble revisión:** `doble_revision/anotaciones_eventos.csv` tiene la causa asignada por cada revisor (antes del consenso) y la causa final. Las categorías son A (vehículo que invade la trayectoria), B (vehículo en carril contiguo), C (objetos fijos al borde), D (giro propio), E (cambio de luz), F (marcas en la pista) y G (otra). El kappa y su IC por bootstrap se calculan con `kappa_bootstrap.py` sobre el libro maestro.

## Privacidad

No se incluyen el video original ni cuadros en los que se lean placas o se identifique a personas. Los revisores aparecen como revisor 1 y revisor 2.

## Licencia

Código bajo licencia MIT (`LICENSE`). Resultados y anotaciones bajo CC BY 4.0. Los recuentos del conectoma derivan de FlyWire (Dorkenwald *et al.*, 2024; Schlegel *et al.*, 2024) y del modelo de Shiu *et al.* (2024); cita esas fuentes si los usas.

## Cómo citar

Usa el archivo `CITATION.cff` o el botón «Cite this repository» de GitHub.
