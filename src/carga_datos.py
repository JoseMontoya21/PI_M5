import os  
import pandas as pd


def cargarDatos():
    #1. Ruta absoluta del directorio donde esta el archivo en la carpeta src
    ruta_actual = os.path.dirname(os.path.abspath(__file__))

    #2. subir un nivel de carpetas para llegar a la carpeta donde esta la base de datos
    ruta_proyecto = os.path.dirname(ruta_actual)

    #3. construyamos la ruta completa a la base de datos
    ruta_excel = os.path.join(ruta_proyecto, "Base_de_datos.xlsx")

    #4. leemos los datos y los imprimimos
    df = pd.read_excel(ruta_excel)
    print(df)

    return df

def cargarDatosLimpios():
    """
    Carga el dataset limpio desde dataset_limpio.csv
    """
    #1. Ruta absoluta del directorio donde esta el archivo en la carpeta src
    ruta_actual = os.path.dirname(os.path.abspath(__file__))

    #2. subir un nivel de carpetas para llegar a la carpeta del proyecto
    ruta_proyecto = os.path.dirname(ruta_actual)

    #3. construyamos la ruta completa al dataset limpio
    ruta_csv_limpio = os.path.join(ruta_proyecto, "dataset_limpio.csv")

    #4. Cargar el dataset limpio
    df = pd.read_csv(ruta_csv_limpio)
    print(f"Dataset limpio cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    return df

if __name__ == "__main__":
    datos = cargarDatos()
    print(datos.head())
    print(datos.columns)
    

