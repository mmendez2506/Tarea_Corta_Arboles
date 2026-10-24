# ==============================
# Tarea Corta 2 — Árboles de Decisión — Inteligencia Artificial
# Integrantes: María Felix Mendez Abarca, Cristhian Rivas y Jozafath Perez
# Descripción: Entrena y evalúa 10 combinaciones de hiperparámetros del
#              árbol de decisión usando validación cruzada.
# ==============================
import time

import numpy as np
import pandas as pd

from src.arbol import ArbolDecision
from src.datos import cargar_datos, train_test_split_manual
from src.evaluacion import accuracy, validacion_cruzada

# Se varía un hiperparámetro a la vez para poder atribuirle el efecto:
#   M1-M6  profundidad (de muy simple a sin límite), con gini y sin freno por muestras
#   M7-M8  min_split_samples con profundidad 10 fija
#   M9-M10 los mismos parámetros que M3 y M7, pero con entropía, para comparar criterios
COMBINACIONES = [
    {"max_depth": 2,    "min_split_samples": 2,   "criterio": "gini"},      # M1
    {"max_depth": 4,    "min_split_samples": 2,   "criterio": "gini"},      # M2
    {"max_depth": 6,    "min_split_samples": 2,   "criterio": "gini"},      # M3
    {"max_depth": 10,   "min_split_samples": 2,   "criterio": "gini"},      # M4
    {"max_depth": 15,   "min_split_samples": 2,   "criterio": "gini"},      # M5
    {"max_depth": None, "min_split_samples": 2,   "criterio": "gini"},      # M6
    {"max_depth": 10,   "min_split_samples": 50,  "criterio": "gini"},      # M7
    {"max_depth": 10,   "min_split_samples": 300, "criterio": "gini"},      # M8
    {"max_depth": 6,    "min_split_samples": 2,   "criterio": "entropia"},  # M9
    {"max_depth": 10,   "min_split_samples": 50,  "criterio": "entropia"},  # M10
]


def correr_experimentos(X_train, y_train, k=5):
    """Una fila por combinación: métricas de validación cruzada (media y desv),
    accuracy en entrenamiento y tamaño del árbol entrenado con todo el train."""
    filas = []
    for i, params in enumerate(COMBINACIONES, 1):
        inicio = time.perf_counter()
        cv = validacion_cruzada(ArbolDecision, params, X_train, y_train, k=k)
        # Árbol con todo el train: para medir sobreajuste (train vs CV) y complejidad
        completo = ArbolDecision(**params)
        completo.fit(X_train, y_train)
        filas.append({"modelo": f"M{i}", **params, **cv,
                      "accuracy_train": accuracy(y_train, completo.predict(X_train)),
                      "profundidad_real": completo.profundidad(),
                      "hojas": completo.cantidad_hojas(),
                      "tiempo_s": round(time.perf_counter() - inicio, 1)})
        print(f"M{i:<3} {params}  F1={cv['f1_media']:.3f} ± {cv['f1_desv']:.3f}")
    tabla = pd.DataFrame(filas)
    tabla["max_depth"] = tabla["max_depth"].astype("Int64")  # None se ve como <NA>
    return tabla


def escoger_mejor(tabla, k=5, metrica="f1"):
    """Regla de un error estándar: entre los modelos cuya métrica media está a menos
    de un error estándar (desv / sqrt(k)) del mejor, se queda con el más simple
    (menos hojas). Devuelve la fila; sus hiperparámetros son COMBINACIONES[fila.name]."""
    top = tabla.loc[tabla[f"{metrica}_media"].idxmax()]
    limite = top[f"{metrica}_media"] - top[f"{metrica}_desv"] / np.sqrt(k)
    candidatos = tabla[tabla[f"{metrica}_media"] >= limite]
    return candidatos.sort_values("hojas").iloc[0]


if __name__ == "__main__":
    X, y = cargar_datos()
    X_train, X_test, y_train, y_test = train_test_split_manual(X, y)
    tabla = correr_experimentos(X_train, y_train)
    pd.set_option("display.width", 200)
    print(tabla.round(3).to_string(index=False))
    mejor = escoger_mejor(tabla)
    print(f"\nMejor modelo: {mejor['modelo']} {COMBINACIONES[mejor.name]}")
