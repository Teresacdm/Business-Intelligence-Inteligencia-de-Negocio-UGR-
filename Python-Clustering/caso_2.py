# -*- coding: utf-8 -*-
"""
@author: Teresa Cabrera del Moral
"""

'''
Documentación sobre clustering en Python:
    http://scikit-learn.org/stable/modules/clustering.html
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
    http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
'''

import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, DBSCAN, Birch, AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn import cluster
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import silhouette_samples
from sklearn.impute import KNNImputer
from math import floor
from scipy.cluster import hierarchy
import seaborn as sns

def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())

def ScatterMatrix(X, name):
    print("\nGenerando scatter matrix...")
    sns.set()
    variables = list(X)
    variables.remove('cluster')
    X['cluster'] += 1
    sns_plot = sns.pairplot(X, vars=variables, hue="cluster", palette='Set2', 
                    plot_kws={"s": 25}, diag_kind="hist") 
    X['cluster'] -= 1
    #en 'hue' indicamos que la columna 'cluster' define los colores
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
     
    plt.savefig("scatmatrix_case2_"+name+".pdf")
    plt.figure()
    plt.show()
    plt.clf()

def Heatmap(X, name, dataset, labels):
    print("\nGenerando heat-map...")
    cluster_centers = X.groupby("cluster").mean()
    centers = pd.DataFrame(cluster_centers, columns=list(dataset))
    centers_desnormal = centers.copy()
    #se convierten los centros a los rangos originales antes de normalizar
    for var in list(centers):
      centers_desnormal[var] = dataset[var].min()+centers[var]*(dataset[var].max()-dataset[var].min())
    
    plt.figure(figsize=(11, 13))
    sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
    plt.savefig("heatmap_case2_"+name+".pdf")
    plt.show()
    plt.clf()

def KPlot(X, name, k, usadas):
    print("\nGenerando kplot...")
    n_var = len(usadas)
    fig, axes = plt.subplots(k, n_var, sharex='col', figsize=(15,10))
    fig.subplots_adjust(wspace=0.5)
    colors = sns.color_palette(palette="Set2", n_colors=k, desat=None)
    rango=[]

    for i in range(n_var):
      rango.append([X[usadas[i]].min(), X[usadas[i]].max()])
    
    for i in range(k):
      dat_filt = X.loc[X['cluster']==i]
      for j in range(n_var):
          ax = sns.kdeplot(dat_filt[usadas[j]], shade=True, color=colors[i], ax=axes[i,j])
          ax.set_xlim(rango[j][0]-0.05*(rango[j][1]-rango[j][0]), rango[j][1]+0.05*(rango[j][1]-rango[j][0]))
    
    plt.savefig("kdeplot_case2_"+name+".pdf")
    plt.figure()
    plt.show()
    plt.clf()
    
def BoxPlot(X, name, k, usadas):
    print("\nGenerando boxplot...")
    n_var = len(usadas)
    fig, axes = plt.subplots(k, n_var, sharey=True, figsize=(15, 15))
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    colors = sns.color_palette(palette=None, n_colors=k, desat=None)
    
    for i in range(k):
      dat_filt = X.loc[X['cluster']==i]
      for j in range(n_var):
            sns.boxplot(x=dat_filt[usadas[j]], shade=True ,color=colors[i], ax=axes[i, j])
        
    plt.savefig("boxplot_case2_"+name+".pdf")
    plt.figure()
    plt.show()
    plt.clf()
    
def Distrib(X, k, usadas, dataset):
    print("Distribución por variable y cluster")
    plt.figure() 				#creo una figura
    plt.style.use('default') 		#le pongo un estilo
    fig, axes = plt.subplots(k, len(usadas), sharey=True, figsize=(15,15))	#Filas(k)= clusters, Columnas(n_var)= variables
    fig.subplots_adjust(wspace=0, hspace=0)	#Para que se queden las gráficas pegadas
    
    cluster_centers = X.groupby("cluster").mean()
    centers = pd.DataFrame(cluster_centers, columns=list(dataset))
    
    centers_sort=centers
    colors = sns.color_palette(palette='Paired', n_colors=k, desat=None)
    
    for i in range(k):
        c=centers_sort.index[i]
        dat_filt = X.loc[X['cluster']==c]	#Dentro de todos los datos, me quedo los de ese cluster
        for j in range(n_var):
            ax = sns.histplot(x=dat_filt[usadas[j]], label="", color=colors[c], ax=axes[i,j], kde=True) #con kde pinto la curva #Otras opciones son boxplot, o kdplot (útil para superponer, cambiar axes[1,j], y subplots(1, n_var) y algo más de estética)
            ax.set(xlabel=usadas[j] if (i==k-1) else ' ', ylabel='Cluster '+str(c+1) if (j==0) else ' ')
            ax.set(yticklabels=[])
            ax.tick_params(left=False)
            ax.grid(axis='x', linestyle='-', linewidth='0.2', color='gray')
            ax.grid(axis='y', visible=False)
            
    fig.set_size_inches(15, 15)
    fig.savefig("distribucion.pdf")
    
def PlotHist(X, k, usadas):
    n_var = len(usadas)

    sns.set()
    fig, axes = plt.subplots(k, n_var, sharey=True, figsize=(15,15))
    fig.subplots_adjust(wspace=0.007, hspace = 0.04)

    colors = sns.color_palette(palette=None, n_colors=k, desat=None)

    rango = [] # contendrá el mín y el máx para cada variable de todos los clusters
    for j in range(n_var):
        rango.append([X[usadas[j]].min, X[usadas[j]].max])
    
    for i in range(k):
        dat_filt = X.loc[X['cluster']==i]
        for j in range(n_var):
            ax = sns.histplot(x=dat_filt[usadas[j]], color = colors[i], label = "", ax = axes[i,j], kde=True)
            if (i==k-1):
                axes[i,j].set_xlabel(usadas[j])
            else:
                axes[i,j].set_xlabel("")
        
            if (j==0):
                axes[i,j].set_ylabel("Cluster "+str(i))
            else:
                axes[i,j].set_ylabel("")
       
            axes[i,j].set_yticks([])
            axes[i,j].grid(axis='x', linestyle='-', linewidth='0.2', color='gray')
            axes[i,j].grid(axis='y', b=False)
       
            ax.set_xlim(rango[j][0]-0.05*(rango[j][1]-rango[j][0]), rango[j][1]+0.05*(rango[j][1]-rango[j][0]))
            
    plt.show()
    
    plt.clf()


def exAlg(algoritmos, X, etiq, usadas):

  # Listas para almacenar los valores
    nombres = []
    tiempos = []
    numcluster = []
    metricaCH = []
    metricaSC = []
    metricaDB = []
      
    for name,alg in algoritmos:    
        print(name,end='')
        t = time.time()
        cluster_predict = alg.fit_predict(X,subset[peso])    
        tiempo = time.time() - t
        k = len(set(cluster_predict))
        print(": clusters: {:3.0f}, ".format(k),end='')
        print("{:6.2f} segundos".format(tiempo))
      
        # Calculamos los valores de cada métrica
        metric_CH = metrics.calinski_harabasz_score(X, cluster_predict)
        print("\nCalinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')
        #el cálculo de Silhouette puede consumir mucha RAM. 
        #Si son muchos datos, más de 10k, se puede seleccionar una muestra, p.ej., el 20%
        if len(X) > 10000:
          m_sil = 0.2
        else:
          m_sil = 1.0
        metric_SC = metrics.silhouette_score(X, cluster_predict, metric='euclidean', 
                                             sample_size=floor(m_sil*len(X)), random_state=123456)
        print("Silhouette Coefficient: {:.5f}".format(metric_SC))
        sample_silhouette_values = silhouette_samples(X, cluster_predict, metric='euclidean')
        
        # Inicializar una lista para almacenar el coeficiente silhouette promedio por cluster
        silhouette_cluster = []
        # Iterar sobre cada cluster para obtener el coeficiente silhouette promedio
        for i in range(max(cluster_predict) + 1):
            cluster_silhouette_values = sample_silhouette_values[cluster_predict == i]
            cluster_silhouette_avg = np.mean(cluster_silhouette_values)
            silhouette_cluster.append(cluster_silhouette_avg)
            print(f"Cluster {i}: Silhouette Score promedio = {cluster_silhouette_avg}")
        
        
        metric_DB = metrics.davies_bouldin_score(X, cluster_predict)
        print("\nDavies-Bouldin Score: {:.3f}, ".format(metric_DB), end='')
        
        #se convierte la asignación de clusters a DataFrame
        clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])
        #y se añade como columna a X
        X_cluster = pd.concat([X, clusters], axis=1)
         
        print("\nTamaño de cada cluster:\n")
        size = clusters['cluster'].value_counts()
        
        for num,i in size.items():
          print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
        
        nombre = name+str(etiq)
    
        
        # Dibujar el Scatter Matrix
        #ScatterMatrix(X = X_cluster, name = nombre)
        
        # Dibujar el Heatmap
        if name=="K-Means":
            #MDS para K-Means
            '''    
            labels = alg.labels_
            X_labeled = X.copy()  # Crear una copia de tus datos normalizados
            X_labeled['Cluster'] = labels
            mds = MDS(n_components=2, random_state=123456)
            resultados_mds = mds.fit_transform(X_labeled)
            plt.figure(figsize=(10, 8))
            plt.scatter(resultados_mds[:, 0], resultados_mds[:, 1], c=labels, cmap='viridis')
            plt.title('Multidimensional Scaling (MDS) con K-Means de la muestra de edades mayor de 65')
            plt.colorbar(label='Cluster')
            plt.savefig("mds_caso2_kmeans_65.pdf")
            plt.show()
            '''   
        #   Heatmap(X = X_cluster, name = nombre, dataset=X, labels = cluster_predict)
        
        # Dibujamos KdePlot
        #KPlot(X = X_cluster, name = nombre, k = k, usadas = usadas)
        
        # Dibujamos BoxPlot
        #BoxPlot(X = X_cluster, name = nombre, k = k, usadas = usadas)
        
        # Dibujamos distribución por variable y cluster
        #Distrib(X=X_cluster, k = k, usadas = usadas, dataset = X)
        
        if name=='AggCluster':
          #Filtro quitando los elementos (outliers) que caen en clusters muy pequeños en el jerárquico
          min_size = 5
          X_filtrado = X_cluster[X_cluster.groupby('cluster').cluster.transform(len) > min_size]
          k_filtrado = len(set(X_filtrado['cluster']))
          print("De los {:.0f} clusters hay {:.0f} con más de {:.0f} elementos. Del total de {:.0f} elementos, se seleccionan {:.0f}".format(k,k_filtrado,min_size,len(X),len(X_filtrado)))
          X_filtrado = X_filtrado.drop(columns=['cluster'])
          #Dendrograms(X = X_filtrado, name = nombre, path = path)
        
        # Almacenamos los datos para generar la tabla comparativa
        nombres.append(name)   
        tiempos.append(tiempo)
        numcluster.append(len(set(cluster_predict)))
        metricaCH.append(metric_CH)
        metricaSC.append(metric_SC)
        metricaDB.append(metric_DB)
        
        print("\n-------------------------------------------\n")
        
        # Generamos la tabla comparativa  
        resultados = pd.concat([pd.DataFrame(nombres, columns=['Name']), 
                                pd.DataFrame(numcluster, columns=['Num Clusters']), 
                                pd.DataFrame(metricaCH, columns=['CH']), 
                                pd.DataFrame(metricaSC, columns=['SC']), 
                                pd.DataFrame(metricaDB, columns=['DB']),
                                pd.DataFrame(tiempos, columns=['Time'])], axis=1)
        print(resultados)


datos = pd.read_csv('03_Datos_noviembre_2023_num.csv')
peso = 'ponde'

#Se imputan los valores desconocidos por vecinos más cercanos

imputer = KNNImputer(n_neighbors=3)
datos_norm = datos.apply(norm_to_zero_one)
datos_imputados_array = imputer.fit_transform(datos_norm)
datos_norm2 = pd.DataFrame(datos_imputados_array, columns = datos.columns)
datos = datos_norm2

# Seleccionar casos
#subset=datos.loc[(datos['edad_r']==0)] #Edades entre 18 y 24
subset=datos.loc[(datos['edad_r']==1)] #Edades superiores a 65

'''
#Histograma para la pregunta 9_1 en el caso de los jóvenes
subset=datos.loc[(datos['edad_r']==0)]
respuestas_cambio_climatico = subset['p9_1']
# Crear el histograma
plt.figure(figsize=(10, 8))
plt.hist(respuestas_cambio_climatico, bins=10, color='skyblue', edgecolor='black')  # Puedes ajustar el número de bins según sea necesario
plt.xlabel('Respuestas sobre Cambio Climático')
plt.title('Histograma de Respuestas sobre Cambio Climático (18-24 años)')
plt.grid(True)
plt.savefig("histograma_cambio_climatico.pdf")
plt.show()
'''

# Seleccionar variables de interés para clustering
# renombramos las variables por comodidad
subset=subset.rename(columns={
    "p8_1": "cambio climático",
    "p8_2": "conflicto bélico",
    "p8_3": "flujo migratorio",
    "p8_4": "terrorismo internacional",
    "p8_5": "crisis inflación",
    "p8_8": "populismos",
    "p8_9": "pandemias",
    })
usadas = ['cambio climático',
          'conflicto bélico',
          'flujo migratorio', 
          'terrorismo internacional',
          'crisis inflación', 
          'populismos', 
          'pandemias']

n_var = len(usadas)
X = subset[usadas]

# eliminar outliers como aquellos casos fuera de 1.5 veces el rango intercuartil
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
X = X[~((X < (Q1 - 1.5 * IQR)) |(X > (Q3 + 1.5 * IQR))).any(axis=1)]

k_means = KMeans(init='k-means++', n_clusters=2, n_init=5, random_state=123456)
ms = MeanShift(bandwidth=0.526, bin_seeding=True)
ward = AgglomerativeClustering(n_clusters=2, linkage="ward")
db = DBSCAN(eps=0.5, min_samples=10)
brc = Birch(branching_factor=25, n_clusters=2, threshold=0.25, compute_labels=True)


algoritmos = {('K-Means', k_means), ('MeanShift', ms), 
               ('AggCluster', ward), ('DBSCAN', db),('Birch', brc)}
print("\nCaso de estudio 2 , tamaño: "+str(len(X))+"\n")
print("-------------------------------------------\n")
exAlg(algoritmos, X, "caso2", usadas)