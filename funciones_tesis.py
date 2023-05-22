import numpy as np
import osmnx as ox
import networkx as nx
import pandas as pd
import folium as fl
import tkinter as tk
from tkinter import simpledialog
from datetime import timedelta
import ipywidgets as widgets
import io
from ipywidgets import HTML
from IPython.display import display
import base64
from IPython.display import IFrame
import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
from geopy.geocoders import Nominatim
import folium
import time
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QRadioButton, QFileDialog, QProgressBar, QDialog, QMessageBox
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, QCoreApplication
from PyQt5.QtGui import QFont
from geopy.geocoders import Nominatim
from geopy.geocoders import ArcGIS

# geolocator = Nominatim(user_agent="Alejandro Fontalvo IIND")
# # It ccrear grafico de bogota
# graph_area = ("Bogotá, Distrito Capital, Colombia")

# # Create the graph of the area from OSM data. It will download the data and create the graph
# G = ox.graph_from_place(graph_area, network_type='drive')

# # OSM data are sometime incomplete so we use the speed module of osmnx to add missing edge speeds and travel times
# G = ox.add_edge_speeds(G)
# G = ox.add_edge_travel_times(G)

def subir_direcciones():
    widgets.IntSlider()
    from IPython.display import display
    w = widgets.IntSlider()
    uploader = widgets.FileUpload(
        accept='*.csv',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
        multiple=False  # True to accept multiple files upload else False
    )
    display(uploader)
    return(uploader)


def descargar_template():
    df=pd.read_csv(r"Data_clientes_template.csv",delimiter=";")
    template = df.to_csv(index=False)

    #FILE
    filename = 'template.csv'
    b64 = base64.b64encode(template.encode())
    payload = b64.decode()

    #BUTTONS
    html_buttons = '''<html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>
    <body>
    <a download="{filename}" href="data:text/csv;base64,{payload}" download>
    <button class="p-Widget jupyter-widgets jupyter-button widget-button mod-warning">Download File</button>
    </a>
    </body>
    </html>
    '''

    html_button = html_buttons.format(payload=payload,filename=filename)
    display(HTML(html_button))
    
def sacar_lat_lon_clientes(direcciones_clientes,geolocator,G):
    
    puntos=[]
    # for i in direcciones_clientes["Direccion"]:
    #     puntico=geolocator.geocode(i, timeout=None)
    #     print(puntico)
    # puntos.append(pd.DataFrame([puntico.longitude,puntico.latitude]).transpose())
    # crear lista de long lat
    df=pd.DataFrame([0,0]).transpose()
    for i in direcciones_clientes["DIRECCION"]:
        puntico=geolocator.geocode(i, timeout=180)
        df=pd.concat([df,pd.DataFrame([puntico.longitude,puntico.latitude]).transpose()])
    df=df.iloc[1:,].reset_index().drop(["index"],axis=1)
    #df.index=nombres
    df.columns=['lon',"lat"]
    df["nodo"]=[ox.distance.nearest_nodes(G, df.iloc[i,0],df.iloc[i,1]) for i in range(df.shape[0])]
    df.index=direcciones_clientes["ID"]
    return(df)

def crear_matriz_distancias(direcciones_clientes,df,G):
    # crear matriz de distancias
    matriz=pd.DataFrame(np.identity(df.shape[0])*9999999)
    matriz.columns=direcciones_clientes["ID"]
    matriz.index=direcciones_clientes["ID"]


    for i in range(df.shape[0]):
        for j in range(df.shape[0]):
            if i!=j:       
                origin_coordinates = (df.iloc[i,0] ,df.iloc[i,1])
                destination_coordinates = (df.iloc[j,0] ,df.iloc[j,1])
                # # In the graph, get the nodes closest to the points
                origin_node = ox.distance.nearest_nodes(G, origin_coordinates[0],origin_coordinates[1])
                destination_node = ox.distance.nearest_nodes(G, destination_coordinates[0],destination_coordinates[1])
                # # Get the distance in meters
                try:
                    distance_in_meters = nx.shortest_path_length(G, origin_node, destination_node, weight='length')
                    matriz.iloc[i,j]=distance_in_meters
                except:
                    distance_in_meters = "NoDisponible"
                    matriz.iloc[i,j]=distance_in_meters
    return(matriz)

def heuristica_nn(matriz):
    contar=0
    estoy=0
    camino=[estoy]
    mat_auxiliar=matriz.copy()
    while contar<matriz.shape[0]:
        mat_auxiliar.iloc[:,estoy]=99999999
        minimo=np.where(mat_auxiliar.iloc[estoy,:]==min(mat_auxiliar.iloc[estoy,:]))[0][0]
        estoy=minimo
        camino.append(estoy)
        contar=contar+1
    return(camino)

def ruta_nodos(camino,df,G):
    rutas=[]
    for i in range(len(camino)-1):
        shortest_route_by_distance = ox.distance.shortest_path(G, df.iloc[camino[i],2], df.iloc[camino[i+1],2], weight='length')
        rutas.append(shortest_route_by_distance[:-1])
    def flatten(l):
        return [item for sublist in l for item in sublist]
    rutas_pegadas=flatten(rutas)
    return(rutas_pegadas)

def mapa_ruta_auxiliar(rutas_pegadas,df,G,secuencia_con_index):
    m3 = ox.plot_route_folium(G, rutas_pegadas, popup_attribute="length", weight=7)
    # add marker one by one on the map
    secuencia_temp = pd.DataFrame(secuencia_con_index)
    secuencia = secuencia_temp.reset_index()
    inicio = secuencia.loc[0]['ID']
    fin = secuencia.loc[secuencia.shape[0]-1]['ID']
    ids = df.index
    #folium.Marker(Location=[df.iloc[df.shape[0]-1]['lat'], df.iloc[df.shape[0]-1]['lon']], icon = folium.Icon(color = 'red'), popup = ids[df.shape[0]-1]).add_to(m3)
    for i in range(0,secuencia.shape[0]):
        punto = secuencia.loc[i]['ID']
        if punto == inicio or punto == fin:
            folium.Marker(location=[df.loc[punto]['lat'], df.loc[punto]['lon']], icon = folium.Icon(color = "green", icon="play", prefix='fa'), popup=punto).add_to(m3)
        else:
            folium.Marker(
              location=[df.loc[punto]['lat'], df.loc[punto]['lon']], popup=punto
           ).add_to(m3)
    return(m3)


def mapa_ruta(rutas_pegadas,df,G,secuencia_con_index, vuelve):
    if vuelve == "No":
        m3 = ox.plot_route_folium(G, rutas_pegadas, popup_attribute="length", weight=7)
        # add marker one by one on the map
        secuencia_temp = pd.DataFrame(secuencia_con_index)
        secuencia = secuencia_temp.reset_index()
        inicio = secuencia.loc[0]['ID']
        fin = secuencia.loc[secuencia.shape[0]-1]['ID']
        ids = df.index
        #folium.Marker(Location=[df.iloc[df.shape[0]-1]['lat'], df.iloc[df.shape[0]-1]['lon']], icon = folium.Icon(color = 'red'), popup = ids[df.shape[0]-1]).add_to(m3)
        for i in range(0,secuencia.shape[0]):
            punto = secuencia.loc[i]['ID']
            if punto == inicio:
                folium.Marker(location=[df.loc[punto]['lat'], df.loc[punto]['lon']], icon = folium.Icon(color = "green", icon="play", prefix='fa'), popup=punto).add_to(m3)
            elif punto == fin:
                folium.Marker(location=[df.loc[punto]['lat'], df.loc[punto]['lon']], icon = folium.Icon(color = "red",icon="flag-checkered", prefix='fa' ), popup=punto).add_to(m3)
            else:
                folium.Marker(
                  location=[df.loc[punto]['lat'], df.loc[punto]['lon']], popup=punto
               ).add_to(m3)
    else:
        m3 = ox.plot_route_folium(G, rutas_pegadas, popup_attribute="length", weight=7)
        # add marker one by one on the map
        secuencia_temp = pd.DataFrame(secuencia_con_index)
        secuencia = secuencia_temp.reset_index()
        inicio = secuencia.loc[0]['ID']
        fin = secuencia.loc[secuencia.shape[0]-1]['ID']
        ids = df.index
        #folium.Marker(Location=[df.iloc[df.shape[0]-1]['lat'], df.iloc[df.shape[0]-1]['lon']], icon = folium.Icon(color = 'red'), popup = ids[df.shape[0]-1]).add_to(m3)
        for i in range(0,secuencia.shape[0]):
            punto = secuencia.loc[i]['ID']
            if punto == inicio or punto == fin:
                folium.Marker(location=[df.loc[punto]['lat'], df.loc[punto]['lon']], icon = folium.Icon(color = "green", icon="play", prefix='fa'), popup=punto).add_to(m3)
            else:
                folium.Marker(
                  location=[df.loc[punto]['lat'], df.loc[punto]['lon']], popup=punto
               ).add_to(m3)



    return(m3)

def optimizacion(matriz):
    model1 = gp.Model("tesis")
    nodos=list(range(0,matriz.shape[0]))
    arcos=[(i,j) for i in nodos for j in nodos]
    #crear variables de decision
    x=model1.addVars(arcos,vtype=GRB.BINARY,name="xij")
    distancias=np.matrix(matriz)
    k=1
    # crear funcion objetivo
    model1.setObjective(quicksum(distancias[i,j]*x[i,j] for i in nodos for j in nodos),GRB.MINIMIZE)
    #restricciones
    #al menos una vez debo entrar al nodo
    model1.addConstrs(quicksum(x[i,j] for i in nodos)==1 for j in nodos if j!=0)
    #al menos una vez debo salir del nodo
    model1.addConstrs(quicksum(x[i,j] for j in nodos)==1 for i in nodos if i!=0)
    #no me puedo devolver por el mismo arco
    model1.addConstrs(x[i,j]+x[j,i]<=1 for i in nodos for j in nodos)
    #tengo que salir k veces del centro de distribucion
    model1.addConstr(quicksum(x[i,0] for i in nodos)==k)
    model1.addConstr(quicksum(x[0,j] for j in nodos)==k)
    model1.optimize()
    
    import networkx as nx
    #Crear tabla con ruta y camino
    arcosActivos=[i for i in arcos if x[i].x==1]

    grafo=np.zeros((matriz.shape[0],matriz.shape[0]))
    for i in nodos:
        for j in nodos:
            grafo[i,j]=x[i,j].x    
    G = nx.from_numpy_array(grafo)

    hora = []
    for ds in nx.cycle_basis(G):
        hora.append((sum([100/1000*distancias[i,j]*x[i,j].x for i in ds for j in ds])+(len(ds)-1)*90)/60)
    hora    

    rutas =[]
    for ds in nx.cycle_basis(G):
        rutas.append("-".join(map(str, ds)))
    rutas

    

    f = []
    for ds in nx.cycle_basis(G):
        f.append(matriz.iloc[ds,-1].sum())
    f   

    tabla = {'Ruta': rutas, 'Tiempo (horas)': hora, "Fotos":f }
    df = pd.DataFrame(data=tabla)
    df
    return(ds,df)

def todo(direcciones_clientes,metodo,geolocator,G, secuencia, vuelve):
    df=sacar_lat_lon_clientes(direcciones_clientes,geolocator,G)
    matriz=crear_matriz_distancias(direcciones_clientes,df,G)
    camino_ret = []
    if metodo =="o":
        camino_opti,tabla=optimizacion(matriz)
        camino_opti.insert(0,0)
        rutas_pegadas=ruta_nodos(camino_opti,df,G)
        camino_ret = camino_opti
        #print(camino_opti)
    else:
        camino=heuristica_nn_carga(matriz,clientes=direcciones_clientes,capacidad=999999999999)
        rutas_pegadas=ruta_nodos(camino,df,G)
        camino_ret = camino
        #print(camino)
    return(mapa_ruta(rutas_pegadas,df,G,secuencia,vuelve), camino_ret)

def heuristica_nn_carga(matriz,clientes,capacidad=999999999999):
    infactible = 'Infactible'
    if(capacidad >= max(clientes["CARGA"].astype(float))):
        contar=0
        estoy=0
        peso=0
        camino=[estoy]
        mat_auxiliar=matriz.copy()
        mat_auxiliar.iloc[:,0]=99999999
        while contar<matriz.shape[0]:
            minimo_t=np.where(mat_auxiliar.iloc[estoy,:]==min(mat_auxiliar.iloc[estoy,:]))[0][0]
            peso_t=peso+clientes.loc[minimo_t,"CARGA"]
            #print(peso_t)
            if peso_t<capacidad:
                minimo=np.where(mat_auxiliar.iloc[estoy,:]==min(mat_auxiliar.iloc[estoy,:]))[0][0]
                estoy=minimo
                mat_auxiliar.iloc[:,estoy]=99999999
                camino.append(estoy)
                contar=contar+1
                peso=peso+clientes.loc[minimo,"CARGA"]
            else:
                estoy=0
                camino.append(0)
                
                minimo=np.where(mat_auxiliar.iloc[estoy,:]==min(mat_auxiliar.iloc[estoy,:]))[0][0]
                estoy=minimo
                peso=clientes.loc[minimo,"CARGA"]
                #print(peso)
                mat_auxiliar.iloc[:,estoy]=99999999
                camino.append(estoy)
                contar=contar+1
        return(camino)
    else:
        return(infactible)


def acumular_distancia(matriz,secuencia_con_index):
    distancia_total = 0
    secuencia_temp = pd.DataFrame(secuencia_con_index)
    secuencia = secuencia_temp.reset_index()
    n = secuencia.shape[0]

    for i in range(n-1):
        salida = secuencia.loc[i, "ID"]
        llegada = secuencia.loc[i+1, "ID"]
        distancia = matriz.loc[salida,llegada]
        distancia_total += distancia

    return(distancia_total)

