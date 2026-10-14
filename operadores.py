"""Lectura y evaluacion para el problema de flow shop permutacional."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence
import random
import numpy as np


@dataclass(frozen=True)
class Instancia:
    """Instancia Taillard: tiempos_procesamiento[maquina][trabajo]."""

    trabajos: int
    maquinas: int
    tiempos_procesamiento: tuple[tuple[int, ...], ...]
    semilla: int | None = None
    limite_superior: float | None = None
    limite_inferior: float | None = None


def leer_instancia(ruta: str | Path) -> Instancia:
    """Lee una instancia Taillard y devuelve sus dimensiones y tiempos."""
    archivo = Path(ruta)
    lineas = archivo.read_text(encoding="utf-8").splitlines()
    candidatos = [
        (indice, re.findall(r"-?\d+(?:\.\d+)?", linea))
        for indice, linea in enumerate(lineas)
        if linea.strip()
    ]
    indice_metadatos, metadatos = next(
        ((indice, valores) for indice, valores in candidatos if len(valores) >= 5),
        (None, []),
    )
    if indice_metadatos is None:
        raise ValueError(f"{archivo} no contiene un encabezado Taillard completo")

    metadatos = metadatos[:5]
    trabajos, maquinas = int(metadatos[0]), int(metadatos[1])
    if trabajos <= 0 or maquinas <= 0:
        raise ValueError("El número de trabajos y máquinas debe ser positivo")

    inicio_matriz = indice_metadatos + 1
    while inicio_matriz < len(lineas) and (
        not lineas[inicio_matriz].strip()
        or lineas[inicio_matriz].lstrip().startswith("#")
    ):
        inicio_matriz += 1
    if (
        inicio_matriz < len(lineas)
        and len(lineas[inicio_matriz].split()) == 2
    ):
        inicio_matriz += 1
    valores_matriz = [
        valor
        for linea in lineas[inicio_matriz:]
        if linea.strip() and not linea.lstrip().startswith("#")
        for valor in linea.split()
    ]
    valores_esperados = trabajos * maquinas
    if len(valores_matriz) != valores_esperados:
        raise ValueError(
            f"{archivo} debe contener {valores_esperados} tiempos y contiene "
            f"{len(valores_matriz)}"
        )

    tiempos = [int(valor) for valor in valores_matriz]
    tiempos_procesamiento = tuple(
        tuple(tiempos[fila * trabajos : (fila + 1) * trabajos])
        for fila in range(maquinas)
    )
    return Instancia(
        trabajos=trabajos,
        maquinas=maquinas,
        tiempos_procesamiento=tiempos_procesamiento,
        semilla=int(metadatos[2]),
        limite_superior=float(metadatos[3]),
        limite_inferior=float(metadatos[4]),
    )


def _validar_permutacion(permutacion: Sequence[int], trabajos: int) -> None:
    """Comprueba que cada trabajo aparezca exactamente una vez."""
    vistos = [False] * trabajos
    if len(permutacion) != trabajos:
        raise ValueError(
            f"La permutación debe contener exactamente los trabajos 0 a {trabajos - 1}"
        )
    for trabajo in permutacion:
        if isinstance(trabajo, bool) or not isinstance(trabajo, int) or trabajo < 0 or trabajo >= trabajos or vistos[trabajo]:
            raise ValueError(
                f"La permutación debe contener exactamente los trabajos 0 a {trabajos - 1}"
            )
        vistos[trabajo] = True


def tiempos_finalizacion(
    tiempos_procesamiento: Sequence[Sequence[int]],
    permutacion: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Calcula los tiempos de finalización por máquina y trabajo programado."""
    maquinas = len(tiempos_procesamiento)
    if maquinas == 0:
        raise ValueError("Debe existir al menos una máquina")
    trabajos = len(tiempos_procesamiento[0])
    if trabajos == 0 or any(
        len(fila) != trabajos for fila in tiempos_procesamiento
    ):
        raise ValueError("La matriz de tiempos debe ser rectangular y no vacía")
    _validar_permutacion(permutacion, trabajos)

    # Cada operación espera al trabajo anterior y a la máquina anterior.
    finalizacion = [[0] * trabajos for _ in range(maquinas)]
    for posicion, trabajo in enumerate(permutacion):
        anterior = 0
        for maquina, pm in enumerate(tiempos_procesamiento):
            trabajo_anterior = finalizacion[maquina][posicion - 1] if posicion else 0
            inicio = anterior if anterior > trabajo_anterior else trabajo_anterior
            anterior = inicio + pm[trabajo]
            finalizacion[maquina][posicion] = anterior
    return tuple(tuple(fila) for fila in finalizacion)


def calcular_fitness(
    tiempos_procesamiento: Sequence[Sequence[int]],
    permutacion: Sequence[int],
) -> int:
    """Devuelve el makespan de una permutación; menor valor es mejor."""
    maquinas = len(tiempos_procesamiento)
    if maquinas == 0:
        raise ValueError("Debe existir al menos una máquina")
    trabajos = len(tiempos_procesamiento[0])
    if trabajos == 0 or any(len(fila) != trabajos for fila in tiempos_procesamiento):
        raise ValueError("La matriz de tiempos debe ser rectangular y no vacía")
    _validar_permutacion(permutacion, trabajos)

    finales_trabajo = [0] * trabajos
    makespan = 0
    for pm in tiempos_procesamiento:
        final_maquina = 0
        for trabajo in permutacion:
            final_previo = finales_trabajo[trabajo]
            inicio = final_maquina if final_maquina > final_previo else final_previo
            final_maquina = inicio + pm[trabajo]
            finales_trabajo[trabajo] = final_maquina
        makespan = final_maquina
    return makespan


def calcular_heuristica_pfsp(tiempos_procesamiento):

    trabajos = len(tiempos_procesamiento[0])
    matriz_heuristica = np.zeros((trabajos, trabajos))
    
    # Sumar el tiempo de cada trabajo en todas las máquinas
    tiempo_total_trabajo = [sum(pm[j] for pm in tiempos_procesamiento) for j in range(trabajos)]
    
    for i in range(trabajos):
        for j in range(trabajos):
            if i != j:
                # La visibilidad es inversamente proporcional al tiempo total del trabajo j
                matriz_heuristica[i][j] = 1.0 / tiempo_total_trabajo[j]
                
    return matriz_heuristica

def regla_transicion_acs(nodo_actual, no_visitados, feromonas, heuristica, beta, q0):
    q = random.random()
    
    # Calcular los valores para los nodos no visitados
    valores = []
    for j in no_visitados:
        tau = feromonas[nodo_actual][j]
        eta = heuristica[nodo_actual][j] ** beta
        valores.append((j, tau * eta))

        #Explotacion vs exploracion
    if q <= q0:
        siguiente_nodo = max(valores, key=lambda x: x[1])[0]
    else:
        suma_total = sum(v[1] for v in valores)
        if suma_total == 0:
            siguiente_nodo = random.choice(no_visitados)
        else:
            probabilidades = [v[1] / suma_total for v in valores]
            nodos = [v[0] for v in valores]
            # random.choices devuelve una lista, sacamos el primer elemento [0]
            siguiente_nodo = random.choices(nodos, weights=probabilidades, k=1)[0]
            
    return siguiente_nodo

def evaporacion_local(feromonas, nodo_i, nodo_j, rho, tau_0):
    feromonas[nodo_i][nodo_j] = ((1 - rho) * feromonas[nodo_i][nodo_j]) + (rho * tau_0)

def evaporacion_global(feromonas, mejor_secuencia, mejor_cmax, rho):
    # 1. Evaporar feromona en todos los segmentos
    feromonas *= (1 - rho)
    
    # 2. Depositar feromona en la mejor secuencia
    deposito = 1.0 / mejor_cmax
    for i in range(len(mejor_secuencia) - 1):
        nodo_actual = mejor_secuencia[i]
        siguiente = mejor_secuencia[i+1]
        feromonas[nodo_actual][siguiente] += (rho * deposito)