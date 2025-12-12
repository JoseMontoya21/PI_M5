import os  

import pandas as pd


def cargarDatos():

    #1. Ruta absoluta del directorio donde esta el archivo en la carpeta src
    ruta_actual = os.path.dirname(os.path.abspath(__file__))

    #2. sbuir un nivel de carpetas para llegar a la carpeta donde esta la base de datos
    ruta_proyecto = os.path.dirname(ruta_actual)

    #3. construyamos la ruya completa a la base de datos
    ruta_excel = os.path.join(ruta_proyecto,"Base_de_datos.xlsx")

    #4. leemos los datos y los imprimimos
    df = pd.read_excel(ruta_excel)
    print(df)

    return df
if __name__ == "__main__":
    datos = cargarDatos()
    print(datos.head())
    print(datos.columns)

