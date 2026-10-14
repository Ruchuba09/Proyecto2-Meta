"""Algoritmo genético para el flow shop permutacional (PFSP)."""

from dataclasses import dataclass
import random
from typing import Sequence

from operadores import calcular_fitness


Permutacion = tuple[int, ...]


@dataclass(frozen=True)
class ResultadoGenetico:
    """Resultado de una ejecución del algoritmo genético."""

    mejor_permutacion: Permutacion
    mejor_fitness: int
    historial: tuple[int, ...]
    mejor_generacion: int


@dataclass(frozen=True)
class ResultadoMemetico:
    """Resultado de una ejecución del algoritmo memético."""

    mejor_permutacion: Permutacion
    mejor_fitness: int
    historial: tuple[int, ...]
    mejor_generacion: int


def generar_poblacion_inicial(
    trabajos: int,
    tamaño_poblacion: int,
    rng: random.Random | None = None,
) -> list[Permutacion]:
    """Genera individuos como permutaciones aleatorias de los trabajos."""
    if trabajos <= 0:
        raise ValueError("El número de trabajos debe ser positivo")
    if tamaño_poblacion <= 0:
        raise ValueError("El tamaño de población debe ser positivo")

    generador = rng or random.Random()
    base = list(range(trabajos))
    poblacion = []
    for _ in range(tamaño_poblacion):
        individuo = base.copy()
        generador.shuffle(individuo)
        poblacion.append(tuple(individuo))
    return poblacion


def seleccion_torneo(
    poblacion: Sequence[Permutacion],
    fitnesses: Sequence[int],
    rng: random.Random,
    tamaño_torneo: int = 3,
) -> Permutacion:
    """Selecciona el mejor individuo entre varios candidatos aleatorios."""
    if not poblacion or len(poblacion) != len(fitnesses):
        raise ValueError("Población y fitnesses deben tener el mismo tamaño no vacío")
    if tamaño_torneo <= 0:
        raise ValueError("El tamaño del torneo debe ser positivo")

    candidatos = [rng.randrange(len(poblacion)) for _ in range(tamaño_torneo)]
    ganador = min(candidatos, key=lambda indice: fitnesses[indice])
    return poblacion[ganador]


def cruce_ox(
    padre1: Permutacion,
    padre2: Permutacion,
    rng: random.Random,
    probabilidad: float = 0.9,
) -> tuple[Permutacion, Permutacion]:
    """Aplica Order Crossover y devuelve dos hijos válidos."""
    _validar_probabilidad(probabilidad, "La probabilidad de cruce")
    _validar_padres(padre1, padre2)
    if rng.random() >= probabilidad or len(padre1) < 2:
        return padre1, padre2

    inicio, fin = sorted(rng.sample(range(len(padre1)), 2))

    def crear_hijo(origen: Permutacion, relleno: Permutacion) -> Permutacion:
        hijo = [None] * len(origen)
        hijo[inicio:fin] = origen[inicio:fin]
        segmento = set(origen[inicio:fin])
        restantes = [trabajo for trabajo in relleno if trabajo not in segmento]
        posiciones = list(range(fin, len(hijo))) + list(range(0, inicio))
        for posicion, trabajo in zip(posiciones, restantes):
            hijo[posicion] = trabajo
        return tuple(hijo)  # type: ignore[arg-type]

    return crear_hijo(padre1, padre2), crear_hijo(padre2, padre1)


def mutacion_intercambio(
    individuo: Permutacion,
    rng: random.Random,
    probabilidad: float = 0.1,
) -> Permutacion:
    """Intercambia dos posiciones del individuo según la probabilidad dada."""
    _validar_probabilidad(probabilidad, "La probabilidad de mutación")
    if rng.random() >= probabilidad or len(individuo) < 2:
        return individuo

    posiciones = rng.sample(range(len(individuo)), 2)
    mutado = list(individuo)
    mutado[posiciones[0]], mutado[posiciones[1]] = (
        mutado[posiciones[1]],
        mutado[posiciones[0]],
    )
    return tuple(mutado)


def algoritmo_genetico(
    tiempos_procesamiento: Sequence[Sequence[int]],
    tamaño_poblacion: int = 50,
    generaciones: int = 100,
    probabilidad_cruce: float = 0.9,
    probabilidad_mutacion: float = 0.1,
    elitismo: int = 1,
    tamaño_torneo: int = 3,
    semilla: int | None = None,
) -> ResultadoGenetico:
    """Ejecuta un AG minimizando el makespan del PFSP."""
    trabajos = _obtener_trabajos(tiempos_procesamiento)
    if tamaño_poblacion <= 0 or generaciones < 0:
        raise ValueError("Población positiva y generaciones no negativas requeridas")
    if elitismo < 0 or elitismo >= tamaño_poblacion:
        raise ValueError("El elitismo debe estar entre 0 y población - 1")
    _validar_probabilidad(probabilidad_cruce, "La probabilidad de cruce")
    _validar_probabilidad(probabilidad_mutacion, "La probabilidad de mutación")

    rng = random.Random(semilla)
    poblacion = generar_poblacion_inicial(trabajos, tamaño_poblacion, rng)
    historial = []
    mejor_global = None
    mejor_fitness = None
    mejor_generacion = 0

    for generacion in range(generaciones + 1):
        fitnesses = [calcular_fitness(tiempos_procesamiento, individuo) for individuo in poblacion]
        orden = sorted(range(len(poblacion)), key=lambda indice: fitnesses[indice])
        mejor_actual = fitnesses[orden[0]]
        historial.append(mejor_actual)
        if mejor_fitness is None or mejor_actual < mejor_fitness:
            mejor_fitness = mejor_actual
            mejor_global = poblacion[orden[0]]
            mejor_generacion = generacion
        if generacion == generaciones:
            break

        siguiente = [poblacion[indice] for indice in orden[:elitismo]]
        while len(siguiente) < tamaño_poblacion:
            padre1 = seleccion_torneo(poblacion, fitnesses, rng, tamaño_torneo)
            padre2 = seleccion_torneo(poblacion, fitnesses, rng, tamaño_torneo)
            hijo1, hijo2 = cruce_ox(padre1, padre2, rng, probabilidad_cruce)
            siguiente.append(mutacion_intercambio(hijo1, rng, probabilidad_mutacion))
            if len(siguiente) < tamaño_poblacion:
                siguiente.append(mutacion_intercambio(hijo2, rng, probabilidad_mutacion))
        poblacion = siguiente

    return ResultadoGenetico(
        mejor_global, mejor_fitness, tuple(historial), mejor_generacion
    )


def busqueda_local_insercion(
    tiempos_procesamiento: Sequence[Sequence[int]],
    individuo: Permutacion,
) -> Permutacion:
    """Mejora una permutación moviendo trabajos a otra posición."""
    actual = individuo
    fitness_actual = calcular_fitness(tiempos_procesamiento, actual)
    mejoro = True
    while mejoro:
        mejoro = False
        for origen in range(len(actual)):
            for destino in range(len(actual)):
                if origen == destino:
                    continue
                candidato = list(actual)
                trabajo = candidato.pop(origen)
                candidato.insert(destino, trabajo)
                candidato = tuple(candidato)
                fitness_candidato = calcular_fitness(tiempos_procesamiento, candidato)
                if fitness_candidato < fitness_actual:
                    actual = candidato
                    fitness_actual = fitness_candidato
                    mejoro = True
                    break
            if mejoro:
                break
    return actual


def algoritmo_memetico(
    tiempos_procesamiento: Sequence[Sequence[int]],
    tamaño_poblacion: int = 50,
    generaciones: int = 100,
    probabilidad_cruce: float = 0.9,
    probabilidad_mutacion: float = 0.1,
    elitismo: int = 1,
    tamaño_torneo: int = 3,
    semilla: int | None = None,
    frecuencia_busqueda: int = 1,
) -> ResultadoMemetico:
    """Combina evolución genética con búsqueda local periódica."""
    trabajos = _obtener_trabajos(tiempos_procesamiento)
    if tamaño_poblacion <= 0 or generaciones < 0:
        raise ValueError("Población positiva y generaciones no negativas requeridas")
    if elitismo < 0 or elitismo >= tamaño_poblacion:
        raise ValueError("El elitismo debe estar entre 0 y población - 1")
    _validar_probabilidad(probabilidad_cruce, "La probabilidad de cruce")
    _validar_probabilidad(probabilidad_mutacion, "La probabilidad de mutación")
    if frecuencia_busqueda <= 0:
        raise ValueError("La frecuencia de búsqueda debe ser positiva")

    rng = random.Random(semilla)
    poblacion = generar_poblacion_inicial(trabajos, tamaño_poblacion, rng)
    historial = []
    mejor_global = None
    mejor_fitness = None
    mejor_generacion = 0
    for generacion in range(generaciones + 1):
        fitnesses = [calcular_fitness(tiempos_procesamiento, individuo) for individuo in poblacion]
        orden = sorted(range(len(poblacion)), key=lambda indice: fitnesses[indice])
        mejor_actual = fitnesses[orden[0]]
        historial.append(mejor_actual)
        if mejor_fitness is None or mejor_actual < mejor_fitness:
            mejor_fitness = mejor_actual
            mejor_global = poblacion[orden[0]]
            mejor_generacion = generacion
        if generacion == generaciones:
            break

        siguiente = [poblacion[indice] for indice in orden[:elitismo]]
        while len(siguiente) < tamaño_poblacion:
            padre1 = seleccion_torneo(poblacion, fitnesses, rng, tamaño_torneo)
            padre2 = seleccion_torneo(poblacion, fitnesses, rng, tamaño_torneo)
            hijo1, hijo2 = cruce_ox(padre1, padre2, rng, probabilidad_cruce)
            hijos = [mutacion_intercambio(hijo1, rng, probabilidad_mutacion)]
            if len(siguiente) + len(hijos) < tamaño_poblacion:
                hijos.append(mutacion_intercambio(hijo2, rng, probabilidad_mutacion))
            if generacion % frecuencia_busqueda == 0:
                hijos = [busqueda_local_insercion(tiempos_procesamiento, hijo) for hijo in hijos]
            siguiente.extend(hijos)
        poblacion = siguiente

    return ResultadoMemetico(
        mejor_global,
        mejor_fitness,
        tuple(historial),
        mejor_generacion,
    )


def _obtener_trabajos(tiempos_procesamiento: Sequence[Sequence[int]]) -> int:
    if not tiempos_procesamiento or not tiempos_procesamiento[0]:
        raise ValueError("La matriz de tiempos no puede estar vacía")
    trabajos = len(tiempos_procesamiento[0])
    if any(len(fila) != trabajos for fila in tiempos_procesamiento):
        raise ValueError("La matriz de tiempos debe ser rectangular")
    return trabajos


def _validar_probabilidad(probabilidad: float, nombre: str) -> None:
    if not 0 <= probabilidad <= 1:
        raise ValueError(f"{nombre} debe estar entre 0 y 1")


def _validar_padres(padre1: Permutacion, padre2: Permutacion) -> None:
    if (
        len(padre1) != len(padre2)
        or len(set(padre1)) != len(padre1)
        or len(set(padre2)) != len(padre2)
        or set(padre1) != set(padre2)
    ):
        raise ValueError("Los padres deben ser permutaciones del mismo conjunto")