# Sistema de Detección de Fraude con XAI

## Descripción general

Este proyecto implementa un sistema de detección de fraude utilizando modelos basados en árboles y técnicas de explicabilidad (XAI).
El objetivo es predecir si una transacción es fraudulenta y explicar qué características influyen en la decisión del modelo.

---

## Estructura del proyecto

### fraud_xai.py

Contiene la implementación desde cero de:

- Árbol de decisión
- Random Forest
- Gradient Boosting
- Ensamble de modelos
- Explicaciones tipo SHAP basadas en árboles

### main.py

Se encarga de:

- Cargar y preprocesar el dataset
- Entrenar los modelos (Con depth 6 y 30 trees tarda alrededor de 50 min)
- Evaluar las predicciones
- Mostrar explicaciones por transacción

### creditcard.csv

Dataset de transacciones financieras con etiquetas de fraude (`1`) y no fraude (`0`).

---

## Modelos implementados

### Árbol de Decisión

Modelo base que divide los datos utilizando la métrica Gini para minimizar la impureza en cada nodo.

### Random Forest

Conjunto de árboles entrenados sobre subconjuntos aleatorios del dataset.
La predicción final se obtiene promediando las salidas de todos los árboles.

### Gradient Boosting

Entrena árboles de forma secuencial.
Cada nuevo árbol aprende a corregir los errores (residuos) de los árboles anteriores, ajustando gradualmente la predicción.

### Ensamble de modelos

Las predicciones del Random Forest y Gradient Boosting se combinan mediante un promedio simple:
