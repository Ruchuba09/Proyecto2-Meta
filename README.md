# Optimización PFSP con metaheurísticas

Proyecto de optimización del problema Permutation Flow Shop Scheduling (PFSP)
mediante algoritmo genético (AG), algoritmo memético (AM) y Ant Colony System
(ACS). Las instancias de prueba se encuentran en `datos/`.

## Requisitos

- Python 3.10 o posterior.
- Dependencias indicadas en `requirements.txt`.

En Windows, desde la carpeta del proyecto:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Ejecución

Los comandos se ejecutan desde la carpeta del proyecto. Cada algoritmo recibe
sus parámetros por línea de comandos, en este orden:

```text
python .\ag.py INSTANCIA SEMILLA POBLACION ITERACIONES PROB_CRUCE PROB_MUTACION SALIDA
python .\am.py INSTANCIA SEMILLA POBLACION ITERACIONES PROB_CRUCE PROB_MUTACION FRECUENCIA_BUSQUEDA SALIDA
python .\acs.py INSTANCIA SEMILLA HORMIGAS ITERACIONES RHO BETA Q0 SALIDA
```

Ejemplos con la instancia pequeña:

```powershell
python .\ag.py .\datos\ins_20_5_00.txt 12345 50 200 0.8 0.1 .\resultados\ag_20.csv
python .\am.py .\datos\ins_20_5_00.txt 12345 50 200 0.8 0.1 10 .\resultados\am_20.csv
python .\acs.py .\datos\ins_20_5_00.txt 12345 50 200 0.1 2.0 0.9 .\resultados\acs_20.csv
```

Para probar los otros tamaños, sustituye la instancia por `ins_50_10_00.txt` o
`ins_100_10_00.txt`. La semilla controla la aleatoriedad y permite repetir una
corrida con los mismos parámetros.

## Resultados

- El archivo indicado por `SALIDA` recibe una fila de resumen por ejecución.
  Incluye el mejor makespan, el RPD, la solución y el tiempo de CPU en segundos.
- Junto al CSV de salida se guarda otro CSV con `iteracion` y
  `mejor_makespan` para cada punto de la convergencia.
- El gráfico de convergencia de esa ejecución se guarda en `graficos/`. No se
  genera una gráfica del tiempo de CPU; ese dato queda en el resumen para la
  comparación de tiempos.

Usa un nombre de salida distinto para cada conjunto de parametros y semilla si
quieres conservar por separado sus curvas. El resumen se agrega al CSV; la curva
y el grafico de una ejecucion con el mismo nombre se reemplazan.

## Estructura principal

- `operadores.py`: lectura de instancias, evaluación del makespan y operadores
  compartidos.
- `algoritmo_genetico.py`: implementaciones del AG y AM.
- `ag.py`, `am.py`, `acs.py`: interfaces de línea de comandos.
- `convergencia.py`: guardado y gráfico de los valores por iteración.
- `datos/`: instancias de prueba.
- `resultados/`: archivos CSV de resumen.
- `graficos/`: graficos de convergencia.
