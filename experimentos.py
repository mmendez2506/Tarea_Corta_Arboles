# ==============================
# Tarea Corta 2 — Árboles de Decisión — Inteligencia Artificial
# Integrantes: María Felix Mendez Abarca, Cristhian Rivas y Jozafath Perez
# Descripción: Entrena y evalúa 10 combinaciones de hiperparámetros del
#              árbol de decisión usando validación cruzada.
# ==============================
from src.arbol import ArbolDecision
from src.datos import cargar_datos, train_test_split_manual
from src.evaluacion import validacion_cruzada

COMBINACIONES = [
    # {"max_depth": 3, "min_split_samples": 2, "criterio": "gini"},
]

if __name__ == "__main__":
    pass
