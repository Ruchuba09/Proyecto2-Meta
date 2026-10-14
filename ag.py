import argparse
import csv
from pathlib import Path
import time

from algoritmo_genetico import algoritmo_genetico, algoritmo_memetico
from operadores import leer_instancia

ENCABEZADOS_RESULTADO = [
    "semilla", "metodo", "instancia", "tamano_poblacion",
    "probabilidad_cruce", "probabilidad_mutacion", "numero_iteraciones",
    "frecuencia_busqueda", "tiempo_ejecucion", "mejor_generacion",
    "mejor_makespan", "limite_superior", "limite_inferior", "rpd",
    "mejor_solucion",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta el algoritmo genetico PFSP")
    parser.add_argument("semilla", type=int)
    parser.add_argument("tamaño_poblacion", type=int)
    parser.add_argument("probabilidad_cruce", type=float)
    parser.add_argument("probabilidad_mutacion", type=float)
    parser.add_argument("numero_iteraciones", type=int)
    parser.add_argument("entrada", type=Path)
    parser.add_argument("salida", type=Path)
    parser.add_argument("--metodo", choices=("genetico", "memetico"), default="genetico")
    parser.add_argument("--frecuencia-busqueda", type=int, default=1)
    argumentos = parser.parse_args()

    raiz = Path(__file__).resolve().parent
    entrada = argumentos.entrada if argumentos.entrada.is_absolute() else raiz / argumentos.entrada
    salida = argumentos.salida if argumentos.salida.is_absolute() else raiz / "result" / argumentos.salida
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

    inicio = time.perf_counter()
    ejecutar = algoritmo_memetico if argumentos.metodo == "memetico" else algoritmo_genetico
    opciones = {
        "tamaño_poblacion": argumentos.tamaño_poblacion,
        "generaciones": argumentos.numero_iteraciones,
        "probabilidad_cruce": argumentos.probabilidad_cruce,
        "probabilidad_mutacion": argumentos.probabilidad_mutacion,
        "semilla": argumentos.semilla,
    }
    if argumentos.metodo == "memetico":
        opciones["frecuencia_busqueda"] = argumentos.frecuencia_busqueda
    resultado = ejecutar(instancia.tiempos_procesamiento, **opciones)
    tiempo = time.perf_counter() - inicio
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
            argumentos.semilla, argumentos.metodo, entrada.stem,
            argumentos.tamaño_poblacion,
            argumentos.probabilidad_cruce, argumentos.probabilidad_mutacion,
            argumentos.numero_iteraciones,
            argumentos.frecuencia_busqueda if argumentos.metodo == "memetico" else "",
            tiempo, resultado.mejor_generacion,
            resultado.mejor_fitness, instancia.limite_superior,
            instancia.limite_inferior, rpd, list(resultado.mejor_permutacion),
        ])
    print(f"Mejor makespan: {resultado.mejor_fitness} (resultado: {salida})")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=__import__("sys").stderr)
        raise SystemExit(2)


