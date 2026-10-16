import argparse
import csv
from pathlib import Path
import random
import numpy as np
import time
import sys

from convergencia import guardar_convergencia

# se traen todas las funciones necesarias para AC
from operadores import (
    leer_instancia, calcular_fitness, calcular_heuristica_pfsp,
    regla_transicion_acs, evaporacion_local, evaporacion_global
)

ENCABEZADOS_RESULTADO = [
    "semilla", "metodo", "instancia", "n_hormigas", "iteraciones",
    "evaporacion_rho", "peso_heuristica_beta", "prob_limite_q0",
    "tiempo_cpu", "mejor_makespan", "limite_superior",
    "rpd", "mejor_solucion"
]

def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta el algoritmo ACS para PFSP")
    # nuevas entradas de AC
    parser.add_argument("entrada", type=Path)
    parser.add_argument("semilla", type=int)
    parser.add_argument("n_hormigas", type=int)
    parser.add_argument("iteraciones", type=int)
    parser.add_argument("rho", type=float)
    parser.add_argument("beta", type=float)
    parser.add_argument("q0", type=float)
    parser.add_argument("salida", type=Path)
    
    argumentos = parser.parse_args()

    # semillas rand
    random.seed(argumentos.semilla)
    np.random.seed(argumentos.semilla)

    raiz = Path(__file__).resolve().parent
    entrada = argumentos.entrada if argumentos.entrada.is_absolute() else raiz / argumentos.entrada
    salida = argumentos.salida if argumentos.salida.is_absolute() else raiz / argumentos.salida
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

    instancia = leer_instancia(entrada)
    tiempos = instancia.tiempos_procesamiento
    num_trabajos = instancia.trabajos

    inicio = time.process_time()
   #logica para AC
    secuencia_inicial = list(range(num_trabajos))
    random.shuffle(secuencia_inicial)
    cmax_inicial = calcular_fitness(tiempos, secuencia_inicial)
    
    tau_0 = 1.0 / (num_trabajos * cmax_inicial)
    matriz_feromonas = np.full((num_trabajos, num_trabajos), tau_0, dtype=float)
    matriz_heuristica = calcular_heuristica_pfsp(tiempos)

    mejor_secuencia_global = None
    mejor_cmax_global = float('inf')
    historial = []

    for iteracion in range(argumentos.iteraciones):
        for h in range(argumentos.n_hormigas):
            nodo_inicial = random.randint(0, num_trabajos - 1)
            secuencia_hormiga = [nodo_inicial]
            
            no_visitados = list(range(num_trabajos))
            no_visitados.remove(nodo_inicial)
            nodo_actual = nodo_inicial

            while no_visitados:
                siguiente_nodo = regla_transicion_acs(
                    nodo_actual, no_visitados, matriz_feromonas, 
                    matriz_heuristica, argumentos.beta, argumentos.q0
                )
                secuencia_hormiga.append(siguiente_nodo)
                no_visitados.remove(siguiente_nodo)
                evaporacion_local(matriz_feromonas, nodo_actual, siguiente_nodo, argumentos.rho, tau_0)
                nodo_actual = siguiente_nodo

            cmax_hormiga = calcular_fitness(tiempos, secuencia_hormiga)
            if cmax_hormiga < mejor_cmax_global:
                mejor_cmax_global = cmax_hormiga
                mejor_secuencia_global = secuencia_hormiga[:]

        historial.append(mejor_cmax_global)
        evaporacion_global(matriz_feromonas, mejor_secuencia_global, mejor_cmax_global, argumentos.rho)

    tiempo_cpu = time.process_time() - inicio
    guardar_convergencia(
        historial, "acs", entrada.stem, argumentos.semilla, salida, raiz
    )

    with salida.open("a", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo)
        if not existe:
            escritor.writerow(ENCABEZADOS_RESULTADO)
            
        #sacasmos el rpd para poder comparar
        rpd = None
        if instancia.limite_superior:
            rpd = 100 * (mejor_cmax_global - instancia.limite_superior) / instancia.limite_superior
            
        escritor.writerow([
            argumentos.semilla, "acs", entrada.stem, argumentos.n_hormigas,
            argumentos.iteraciones, argumentos.rho, argumentos.beta, argumentos.q0,
            round(tiempo_cpu, 4), mejor_cmax_global,
            instancia.limite_superior, round(rpd, 4) if rpd is not None else "", 
            list(mejor_secuencia_global)
        ])
        
    print(f"\nFinalizado ACS. Mejor makespan: {mejor_cmax_global} (resultado: {salida})")

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)