# ==============================
# Tarea Corta 2 — Árboles de Decisión — Inteligencia Artificial
# Integrantes: María Felix Mendez Abarca, Cristhian Rivas y Jozafath Perez
# Descripción: Carga y limpieza del dataset adult-census y división manual
#              en entrenamiento y prueba usando solo NumPy.
# ==============================
import numpy as np
import pandas as pd

SEMILLA = 42


def cargar_datos(ruta="data/adult-census.csv"):
    """Lee el CSV y devuelve X (DataFrame) y y (arreglo de NumPy con 0/1).

    Limpieza:
    - Las etiquetas y todas las columnas de texto vienen con un espacio al inicio
      (' <=50K', ' Private'); skipinitialspace lo quita al leer.
    - La clase se pasa a 0/1: >50K = 1, <=50K = 0.
    - Los '?' (dato faltante en workclass, occupation y native-country) se dejan
      como una categoría más: que falte el dato también es información.

    Las categóricas se dejan como texto limpio y ArbolDecision las codifica: en cada
    nodo busca el mejor grupo de categorías para cada lado. Con validación cruzada
    eso dio mejor F1 que codificarlas aquí como números (ordinal) o como columnas
    0/1 (one-hot); ver la sección Datos del notebook.
    """
    df = pd.read_csv(ruta, skipinitialspace=True)
    y = (df.pop("class") == ">50K").astype(int).to_numpy()
    return df, y


def filas(X, idx):
    """X[idx] tanto para arreglos de NumPy como para DataFrames."""
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def train_test_split_manual(X, y, test_size=0.2, semilla=SEMILLA):
    """Divide en entrenamiento y prueba barajando los índices con NumPy.

    Devuelve X_train, X_test, y_train, y_test. Con la misma semilla la división
    es siempre la misma.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1")
    idx = np.random.default_rng(semilla).permutation(len(y))
    n_test = int(round(len(y) * test_size))
    test, train = idx[:n_test], idx[n_test:]
    return filas(X, train), filas(X, test), y[train], y[test]
