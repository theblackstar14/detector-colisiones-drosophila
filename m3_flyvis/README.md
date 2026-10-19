# M3: integración con flyvis

`run_flyvis.py` reemplaza el correlador de M2 por la red preentrenada de *flyvis* (Lappalainen *et al.*, 2024), conserva la capa LPLC2 y la fibra gigante y evalúa con los mismos escenarios y semillas. `congelado_M3.json` guarda la configuración congelada (red flow/0000/000, inhibición 1,0, sin T4c ni T4d) y su umbral.

Instalación (entorno aparte, recomendado):

    pip install flyvis

En Windows, si aparece `WinError 32` al cerrar archivos de datamate, ejecuta una vez `python parche_datamate.py`.

Uso:

    python run_flyvis.py prueba      # verificación rápida
    python run_flyvis.py completo    # evaluación completa
