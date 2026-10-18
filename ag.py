import argparse
import csv
from pathlib import Path
import sys
import time

from algoritmo_genetico import algoritmo_genetico, algoritmo_memetico
from convergencia import guardar_convergencia
from operadores import leer_instancia

ENCABEZADOS_RESULTADO = [
    "semilla", "metodo", "instancia", "tamano_poblacion",
    "probabilidad_cruce", "probabilidad_mutacion", "numero_iteraciones",
    "frecuencia_busqueda", "tiempo_cpu", "mejor_generacion",
    "mejor_makespan", "limite_superior", "limite_inferior", "rpd",
    "mejor_solucion",
]


def main(metodo: str = "genetico") -> None:
    try:
        _ejecutar(metodo)
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


def _ejecutar(metodo: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Ejecuta el algoritmo {metodo} para PFSP"
    )
    parser.add_argument("entrada", type=Path)
    parser.add_argument("semilla", type=int)
    parser.add_argument("tamaño_poblacion", type=int)
    parser.add_argument("numero_iteraciones", type=int)
    parser.add_argument("probabilidad_cruce", type=float)
    parser.add_argument("probabilidad_mutacion", type=float)
    if metodo == "memetico":
        parser.add_argument("frecuencia_busqueda", type=int)
    parser.add_argument("salida", type=Path)
    argumentos = parser.parse_args()

    raiz = Path(__file__).resolve().parent
    entrada = argumentos.entrada
    if not entrada.is_absolute():
        entrada = raiz / entrada
    salida = argumentos.salida
    if not salida.is_absolute():
        salida = raiz / salida

    instancia = leer_instancia(entrada)
    salida.parent.mkdir(parents=True, exist_ok=True)
    existe = salida.exists() and salida.stat().st_size > 0
    if existe:
        with salida.open("r", encoding="utf-8-sig", newline="") as archivo:
            encabezado = next(csv.reader(archivo), [])
        if encabezado != ENCABEZADOS_RESULTADO:
            raise ValueError(
                f"{salida} usa un formato CSV antiguo o incompatible; "
                "elige otro archivo de salida"
            )

    opciones = {
        "tamaño_poblacion": argumentos.tamaño_poblacion,
        "generaciones": argumentos.numero_iteraciones,
        "probabilidad_cruce": argumentos.probabilidad_cruce,
        "probabilidad_mutacion": argumentos.probabilidad_mutacion,
        "semilla": argumentos.semilla,
    }
    if metodo == "memetico":
        opciones["frecuencia_busqueda"] = argumentos.frecuencia_busqueda
    ejecutar = algoritmo_memetico if metodo == "memetico" else algoritmo_genetico

    inicio = time.process_time()
    resultado = ejecutar(instancia.tiempos_procesamiento, **opciones)
    tiempo_cpu = time.process_time() - inicio

    guardar_convergencia(
        resultado.historial, metodo, entrada.stem, argumentos.semilla, salida, raiz
    )

    with salida.open("a", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo)
        if not existe:
            escritor.writerow(ENCABEZADOS_RESULTADO)
        rpd = (
            100 * (resultado.mejor_fitness - instancia.limite_superior)
            / instancia.limite_superior
            if instancia.limite_superior
            else None
        )
        escritor.writerow([
            argumentos.semilla, metodo, entrada.stem,
            argumentos.tamaño_poblacion,
            argumentos.probabilidad_cruce, argumentos.probabilidad_mutacion,
            argumentos.numero_iteraciones,
            argumentos.frecuencia_busqueda if metodo == "memetico" else "",
            tiempo_cpu, resultado.mejor_generacion,
            resultado.mejor_fitness, instancia.limite_superior,
            instancia.limite_inferior, rpd, list(resultado.mejor_permutacion),
        ])
    print(f"Mejor makespan: {resultado.mejor_fitness} (resultado: {salida})")


if __name__ == "__main__":
    main()
