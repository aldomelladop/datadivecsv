"""
Módulo para realizar un Análisis Exploratorio de Datos (EDA) básico en un archivo CSV.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def execute_analysis(file_name: str, normalize_cols=True,
                     check_duplicates=True, show_summary_flag=True) -> pd.DataFrame:
    """
    Ejecuta un análisis exploratorio de datos en un archivo CSV.

    :param file_name: Nombre del archivo CSV a leer.
    :param normalize_cols: Booleano, si se normalizan los nombres de columnas.
    :param check_duplicates: Booleano, si se verifica y elimina duplicados.
    :param show_summary_flag: Booleano, si se muestra un resumen del DataFrame.
    :return: DataFrame de pandas o None en caso de error.
    """
    data_frame = None
    file_found = False

    # Directorios en los que buscar el archivo, local y en la nube
    directories = ['datasets/', 'notebooks/', '']
    current_dir = os.getcwd()

    # Primero, intentar leer usando rutas relativas al directorio actual
    for folder in directories:
        file_path = os.path.join(current_dir, folder, file_name) if folder else file_name
        print(f"Intentando leer desde: {file_path}")  # Mensaje de depuración

        try:
            # Imprime el contenido del directorio si es uno de los esperados
            if folder in ['datasets/', 'notebooks/']:
                print(f"Contenido del directorio {os.path.join(current_dir, folder)}: {os.listdir(os.path.join(current_dir, folder))}")

            # Intento de leer el archivo
            data_frame = pd.read_csv(file_path)
            file_found = True
            print(f"Archivo encontrado en: {file_path}")  # Mensaje de depuración
            break
        except FileNotFoundError:
            print(f"No se encontró el archivo en: {file_path}")  # Mensaje de depuración
            continue

    # Si no se encuentra, intentar leer usando rutas relativas a la raíz
    if not file_found:
        directories = ['/datasets/', '/notebooks/', '']
        for folder in directories:
            file_path = os.path.join(folder, file_name) if folder else file_name
            print(f"Intentando leer desde: {file_path}")  # Mensaje de depuración
            try:
                data_frame = pd.read_csv(file_path)
                file_found = True
                print(f"Archivo encontrado en: {file_path}")  # Mensaje de depuración
                break
            except FileNotFoundError:
                print(f"No se encontró el archivo en: {file_path}")  # Mensaje de depuración
                continue

    # Si después de ambos bucles el archivo no se encontró, retornar None
    if not file_found:
        print("Archivo no encontrado en las rutas especificadas.")
        return None
    
    if normalize_cols:
        data_frame = normalize_column_names(data_frame)

    if check_duplicates:
        data_frame = check_for_duplicates(data_frame)
        if data_frame.empty:
            print("Datos vacíos tras eliminar duplicados.")
            return None

    if show_summary_flag:
        show_summary(data_frame)

    return data_frame


def normalize_column_names(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza los nombres de las columnas de un DataFrame a snake_case.

    :param data_frame: DataFrame de pandas.
    :return: DataFrame con nombres de columnas normalizados.
    """
    # *********** Corregir: Está colocando doble guión bajo ***********
    data_frame.columns = [re.sub(r'(?<!^)(?=[A-Z])', '_', col).lower().replace(' ', '_').replace('__', '_')
                          for col in data_frame.columns]
    return data_frame


def check_for_duplicates(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica y elimina filas duplicadas en un DataFrame.

    :param data_frame: DataFrame de pandas.
    :return: DataFrame sin duplicados.
    """
    if data_frame.duplicated().sum() > 0:
        print("Duplicados encontrados. Eliminando...")
        data_frame.drop_duplicates(inplace=True)
    return data_frame


def show_summary(data_frame: pd.DataFrame):
    """
    Muestra un resumen del DataFrame.

    :param data_frame: DataFrame de pandas.
    """
    print("Primeras 5 filas:")
    print(data_frame.head())
    print("\nÚltimas 5 filas:")
    print(data_frame.tail())
    print("\nMuestra aleatoria de 5 filas:")
    print(data_frame.sample(5))

    print("\nInformación del DataFrame:")
    print(data_frame.info())

    print("\nEstadísticas Descriptivas:")
    print(data_frame.describe())

    print("\nValores Faltantes:")
    print(data_frame.isnull().sum())

    print("\nHistogramas para Variables Numéricas:")
    data_frame.hist(bins=15, figsize=(15, 10))
    plt.show()

    if data_frame.select_dtypes(include=[np.number]).shape[1] > 1:
        print("\nMapa de Calor de Correlación:")
        sns.heatmap(data_frame.corr(), annot=True)
        plt.show()

    print("\nAnálisis de Variables Categóricas:")
    for column in data_frame.select_dtypes(include=['object']).columns:
        print(f"\nDistribución de la variable {column}:")
        print(data_frame[column].value_counts())
        sns.countplot(y=column, data=data_frame)
        plt.show()

if __name__ == "__main__":
    execute_analysis('tu_archivo.csv', normalize_cols=True,
                     check_duplicates=False, show_summary_flag=True)
