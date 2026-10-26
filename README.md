# Tarea Corta 2 — Árboles de Decisión

**Curso:** Inteligencia Artificial — TEC  
**Integrantes:** María Felix Mendez Abarca, Cristhian Rivas y Jozafath Perez

Este proyecto usa el dataset `adult-census` para predecir si una persona gana más de 50 mil dólares al año. El árbol de decisión está hecho desde cero con NumPy. El taller incluye validación cruzada y la comparación de 10 combinaciones de hiperparámetros.

## Archivos del proyecto

- `src/arbol.py`: implementación del árbol, los nodos, Gini y entropía.
- `src/datos.py`: carga de datos y división en entrenamiento y prueba.
- `src/evaluacion.py`: métricas y validación cruzada.
- `experimentos.py`: combinaciones de hiperparámetros que se van a comparar.
- `taller.ipynb`: resultados y análisis.
- `tests/test_arbol.py`: pruebas del árbol.
- `tests/test_evaluacion.py`: pruebas de datos, métricas y validación cruzada.
- `data/adult-census.csv`: datos del taller.

## Cómo usar el árbol

```python
from src.arbol import ArbolDecision

modelo = ArbolDecision(max_depth=6, min_split_samples=20, criterio="gini")
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
```

El árbol acepta datos numéricos y columnas de texto en un DataFrame. Las etiquetas pueden ser números o texto. Si usas un DataFrame, las columnas pueden llegar en otro orden al predecir, pero deben estar todas las que se usaron para entrenar.

Los parámetros principales son:

- `max_depth`: profundidad máxima del árbol. Usa `None` para no poner un límite.
- `min_split_samples`: mínimo de muestras que necesita un nodo para poder dividirse.
- `criterio`: medida de impureza, `"gini"` o `"entropia"`.

Para revisar el modelo puedes usar `resumen()`, `importancia_variables()` e `imprimir(max_niveles=3)`.

## Datos y evaluación

```python
from src.datos import cargar_datos, train_test_split_manual
from src.evaluacion import validacion_cruzada

X, y = cargar_datos()
X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2)
cv = validacion_cruzada(ArbolDecision, {"max_depth": 10, "min_split_samples": 300, "criterio": "gini"},
                        X_train, y_train, k=5)
```

`validacion_cruzada` devuelve la media y la desviación estándar de accuracy, precision, recall y F1, calculadas solo con NumPy.

## Experimentos

`python experimentos.py` corre las 10 combinaciones con validación cruzada de 5 folds e imprime la tabla y el modelo escogido. El modelo final es `max_depth=10`, `min_split_samples=300` con gini, con F1 de 0.667 en validación y en prueba.

## Pruebas

Desde la carpeta principal del proyecto, ejecuta:

```bash
python -m tests.test_arbol
python -m tests.test_evaluacion
```
