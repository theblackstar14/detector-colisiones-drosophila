"""
Parche para datamate 1.0.0 en Windows (dependencia de flyvis).

Bug: en datamate/io.py, _write_h5 abre el archivo .h5 y, si falla, intenta
borrarlo SIN cerrarlo antes. Linux lo permite; Windows no (WinError 32).
El parche cierra el archivo antes de borrarlo. Crea un respaldo io.py.bak.

Uso:
    python parche_datamate.py
"""
import shutil
from pathlib import Path
import datamate.io as dio

ruta = Path(dio.__file__)
s = ruta.read_text(encoding="utf-8")

MARCA = "# parche-windows: cerrar antes de borrar"
if MARCA in s:
    print("Ya estaba parchado:", ruta)
    raise SystemExit(0)

viejo = """    except Exception:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():"""
nuevo = f"""    except Exception:
        {MARCA}
        try:
            f.close()
        except Exception:
            pass
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():"""

if s.count(viejo) != 1:
    raise SystemExit("No encontré el bloque esperado; la versión de datamate es distinta. Avísame.")

shutil.copy(ruta, ruta.with_suffix(".py.bak"))
ruta.write_text(s.replace(viejo, nuevo), encoding="utf-8")
print("Parche aplicado:", ruta)
print("Respaldo:", ruta.with_suffix(".py.bak"))
