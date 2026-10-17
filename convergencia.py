import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def guardar_convergencia(
    historial: Sequence[int],
    metodo: str,
    instancia: str,
    semilla: int,
    salida: Path,
    raiz: Path,
) -> tuple[Path, Path]:
    """Guarda los valores por iteración y su gráfico de convergencia."""
    nombre = f"{salida.stem}_{metodo}_{instancia}_semilla_{semilla}_convergencia"
    archivo_csv = salida.with_name(f"{nombre}.csv")
    archivo_grafico = raiz / "graficos" / f"{nombre}.png"
    archivo_grafico.parent.mkdir(parents=True, exist_ok=True)

    with archivo_csv.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["iteracion", "mejor_makespan"])
        escritor.writerows(enumerate(historial))

    figura, eje = plt.subplots()
    eje.plot(range(len(historial)), historial)
    eje.set_title(f"Convergencia {metodo.upper()} - {instancia}")
    eje.set_xlabel("Iteración")
    eje.set_ylabel("Mejor makespan")
    eje.grid(True)
    figura.tight_layout()
    figura.savefig(archivo_grafico)
    plt.close(figura)
    return archivo_csv, archivo_grafico
