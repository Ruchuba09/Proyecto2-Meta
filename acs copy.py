import sys
import time
import numpy as np
import pandas as pd
import igraph as ig
import csv
import os


#Tiene parametros de entrada: semilla, tamaño de la colonia, numero iteraciones, tasa de evaporacion, beta,q0, datos de entrada

print('-Numero de semilla: ')
print('-Tamaño de la colonia: ')
print('-Numero de iteraciones: ')
print('-Tasa de evaporacion: ')
print('-Beta: ')
print('-Q0: ')
print('-Datos de entrada: ')
#no se pq estaba aca
# sys.exit(0)

# Capturar el nombre del archivo de salida
archivo_salida = "resultados.csv" 
#lectura de parametros para ejecucion por terminal
if len(sys.argv) > 7:
    semilla = int(sys.argv[1])
    tamColonia = int(sys.argv[2])
    ite = int(sys.argv[3])
    tev = float(sys.argv[4])
    beta = float(sys.argv[5])
    q0 = float(sys.argv[6])
    entrada = sys.argv[7]
    archivo_salida = sys.argv[8]
else:#para testeo rapido
    semilla, tamColonia, ite, tev, beta, q0, entrada = 12345, 10, 50, 0.1, 2.0, 0.9, 'instancia.txt'
col = tamColonia
decimales = 4

tiempo_proceso_ini = time.process_time()
np.random.seed(semilla)

#Num variables aun no tiene valor
'''
matrizDistancias = np.full((numVariables,numVariables),fill_value=-1.0,dtype=float)
for i in range (numVariables-1):
    for j in range(i+1,numVariables):
        matrizDistancias[i,j] = np.sqrt(np.sum(np.square(matrizCoordenadas[i]-matrizCoordenadas[j])))
        matrizDistancias[j,i] = matrizDistancias[i,j]
'''
#generar función de costos
def solucionCalculoCosto(n,s,c):
    aux = c[s[n-1],s[0]]
    for i in range(n-1):
        aux += c[s[i],s[i+1]]
    return aux

#solucionMejorCosto aun no existe
'''
tij0 =1/(numVariables*solucionMejorCosto)
matrizFeromonas = np.full((numVariables,numVariables),fill_value=tij0,dtype=float)
'''

# 1. Leer archivo de coordenadas
try:
    matrizCoordenadas = pd.read_table(entrada, header=None,sep=r'\s+',skiprows=6,skipfooter=2, engine='python')
    matrizCoordenadas = matrizCoordenadas.drop(columns=0).to_numpy()
except:
    print(f"Advertencia: No se pudo leer {entrada}. Usando coordenadas aleatorias (KISS).")
    matrizCoordenadas = np.random.rand(52, 2) * 100

print('matriz de coordenadas: ', matrizCoordenadas.shape)
numVariables = matrizCoordenadas.shape[0]

# 2. Generar matriz de distancias D_ij
matrizDistancias = np.full((numVariables,numVariables),fill_value=-1.0,dtype=float)
for i in range(numVariables-1):
    for j in range(i+1,numVariables):
        matrizDistancias[i,j] = np.sqrt(np.sum(np.square(matrizCoordenadas[i]-matrizCoordenadas[j])))
        matrizDistancias[j,i] = matrizDistancias[i,j]

# 3. Generar matriz de heurística
# div 0
with np.errstate(divide='ignore'):
    matrizHeuristica = np.full_like(matrizDistancias,fill_value=1/matrizDistancias,dtype=float)
np.fill_diagonal(matrizHeuristica,0)

# 4. Generar aleatoriamente 1ra hormiga[cite: 13]
solucionOptima = np.array([0,48,31,44,18,40,7,8,9,42,32,50,10,51,13,12,46,25,26,27,11,24,3,5,14,4,23,47,37,36,39,38,35,34,33,41,43,45,49,30,29,28,22,21,20,19,17,16,15,2,1])
if numVariables < 52: solucionOptima = np.arange(numVariables)
solucionMejor = np.arange(0,numVariables)
np.random.shuffle(solucionMejor)
solucionMejorCosto = solucionCalculoCosto(numVariables,solucionMejor,matrizDistancias)

solucionMejorIteracion = 0

# 6. Generar la matriz de feromonas
tij0 = 1 / (numVariables * solucionMejorCosto)
matrizFeromonas = np.full((numVariables,numVariables), fill_value=tij0, dtype=float)

generacion = 0

while generacion < ite:
    generacion += 1
    print('Generacion: ',generacion)

    # 7. Generar la matriz de la colonia
    colonia = np.full((tamColonia,numVariables),fill_value=-1,dtype=int)

    # 9. Generar mecanismo de memoria
    memoria = np.ones((col,numVariables),dtype=int)

    # 8. Inicializar el primer nodo de cada hormiga al azar
    for k in range(col):
        colonia[k,0] = np.random.randint(numVariables)
        memoria[k][colonia[k,0]] = 0

    # sec2
    for i in range(numVariables-1):
        for k in range(col):
            
            # 1. Seleccionar el siguiente nodo j a partir del nodo i para cada hormiga.
            Tall = memoria[k]*matrizFeromonas[colonia[k,i],:]*np.power(matrizHeuristica[colonia[k,i],:],beta)
            if np.random.rand() < q0:
                j0= np.random.choice(np.where(Tall==Tall.max())[0])
                colonia[k,i+1] = j0
                memoria[k][j0] = 0
            else:
                TxN = Tall
                ruleta = (memoria[k]*TxN)/TxN.sum(where=memoria[k].astype(bool))
                
                for l in range(1,numVariables):
                    ruleta[l] = ruleta[l]+ruleta[l-1]
                rTiro = np.random.rand()
                for j in range(numVariables):
                    if rTiro <= ruleta[j]:
                        colonia[k,i+1] = j
                        memoria[k][j] = 0
                        break
            matrizFeromonas[colonia[k][i],colonia[k,i+1]] = ((1-tev)*matrizFeromonas[colonia[k,i],colonia[k,i+1]]) + (tev*tij0)
            matrizFeromonas[colonia[k][i+1],colonia[k,i]] = matrizFeromonas[colonia[k,i],colonia[k,i+1]]

    for k in range(col):
        matrizFeromonas[colonia[k][-1]][colonia[k][0]] = ((1-tev)*matrizFeromonas[colonia[k][-1],colonia[k][0]]) + (tev*tij0)
        matrizFeromonas[colonia[k][0]][colonia[k][-1]] = matrizFeromonas[colonia[k][-1],colonia[k][0]]

    for k in range(col):
        solucionCosto = solucionCalculoCosto(numVariables,colonia[k],matrizDistancias)
        if solucionCosto < solucionMejorCosto:
            solucionMejorCosto = solucionCosto
            solucionMejor = colonia[k].copy()
            solucionMejorIteracion = generacion
            print('Nueva mejor solucion: Costo', solucionMejorCosto)

    for i in range(numVariables-1):
        for j in range(i+1,numVariables):
            matrizFeromonas[i,j] = (1-tev)*matrizFeromonas[i,j]
            matrizFeromonas[j,i] = matrizFeromonas[i,j]
    for i in range(numVariables-1):
        matrizFeromonas[solucionMejor[i],solucionMejor[i+1]] += tev * (1/solucionMejorCosto)
        matrizFeromonas[solucionMejor[i+1],solucionMejor[i]] = matrizFeromonas[solucionMejor[i],solucionMejor[i+1]]
    matrizFeromonas[solucionMejor[-1],solucionMejor[0]] += tev * (1/solucionMejorCosto)
    matrizFeromonas[solucionMejor[0],solucionMejor[-1]] = matrizFeromonas[solucionMejor[-1],solucionMejor[0]]
#calculo del tiempo
tiempo_cpu = time.process_time() - tiempo_proceso_ini
print(f'Tiempo de ejecución total: {tiempo_cpu:.4f} segundos')


# Crear o añadir al archivo CSV
archivo_existe = os.path.isfile(archivo_salida)

with open(archivo_salida, mode='a', newline='') as f:
    writer = csv.writer(f)
    # Si el archivo no existe, escribimos primero los encabezados
    if not archivo_existe:
        writer.writerow(['Instancia', 'Semilla', 'Hormigas', 'Iteraciones', 'Evaporacion', 'Beta', 'q0', 'Mejor_Costo', 'Tiempo_CPU'])
    
    # Guardamos los resultados de esta corrida
    writer.writerow([entrada, semilla, tamColonia, ite, tev, beta, q0, round(solucionMejorCosto,4), round(tiempo_cpu, 4)])

print(f'Resultados guardados correctamente en: {archivo_salida}')