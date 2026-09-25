# FBJ Internet — Herramienta de decisión de capacidad y velocidad

Prototipo y despliegue de la solución del proyecto de Prácticas Profesionalizantes.

**ITSE** · Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial
Claramunt, Federico · Molina, Javier

---

## Qué es

Una herramienta de decisión para el titular de FBJ Internet, un proveedor de servicios de Internet
local que revende capacidad de dos antenas Starlink a 60 clientes. La herramienta estima la
velocidad efectiva que va a recibir un cliente y, a partir de ahí, cuántos clientes activos admite
cada nodo antes de degradar el servicio.

El modelo de estimación está adentro. Las pantallas están organizadas alrededor de decisiones:

| Pantalla | Pregunta que responde |
|---|---|
| 1 · Panel del nodo | ¿Cómo está operando cada antena y contra qué límite? |
| 2 · Cartera por zona | ¿Dónde se concentra el problema? |
| 3 · Simulador de alta | ¿Qué velocidad va a recibir el cliente que estoy por instalar? |
| 4 · Capacidad del nodo | ¿Cuántos clientes más aguanta y qué cuesta cada uno? |
| 5 · Decisión de inversión | ¿Amplío el satélite o mejoro el tramo final? |
| 6 · Datos y límites | ¿Qué sostiene esta herramienta y qué no? |

## Qué NO es

No lee datos en vivo de la red. **No es un sistema de monitoreo**: es una herramienta de decisión
construida sobre 90 días de análisis (01/06/2026 al 29/08/2026). Los datos son de origen simulado,
construidos a partir de los parámetros operativos y económicos relevados con la entidad. El detalle
está en la pantalla «Datos y límites» de la propia aplicación.

## Cómo correrlo

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abre en `http://localhost:8501`.

## Estructura

```
app.py                      aplicación (6 pantallas)
modelo_metricas.json        métricas, curva de saturación, umbrales y efectos del modelo
entrenar_modelo_fbj.py      script de entrenamiento (reproducibilidad)
data/
  mediciones_procesado.csv  8.067 mediciones · 52 clientes · 90 días · 4 franjas
  clientes_procesado.csv    padrón de 60 clientes sin datos personales
```

La aplicación entrena el modelo al arrancar, desde `data/mediciones_procesado.csv` y con el mismo
procedimiento de `entrenar_modelo_fbj.py`. Así el resultado es reproducible y no depende de la
versión exacta de las librerías para deserializar un archivo binario.

## El modelo

- **Objetivo:** estimar `velocidad_efectiva_mbps` por cliente y franja horaria.
- **Variables:** distancia al nodo, señal medida, clientes activos en el nodo, modalidad de conexión,
  nodo, franja horaria y condiciones del día (lluvia, fin de semana, feriado).
- **Enfoque:** Random Forest, comparado contra una regresión lineal regularizada como línea base
  interpretable.
- **Evaluación:** partición temporal (días 1 a 60 para entrenar, 61 a 90 para evaluar), para medir el
  desempeño sobre días que el modelo no vio.

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Ridge (lineal regularizada) | 3,11 Mbps | 3,85 | 0,751 |
| **Random Forest** | **2,76 Mbps** | **3,49** | **0,795** |

### Decisiones de análisis

- **El consumo por franja se excluyó del modelo.** Correlaciona 0,954 con la velocidad, pero no la
  explica: el consumo se calcula a partir de la velocidad lograda. Usarlo como predictor inflaría
  el desempeño en 0,07 de R² sin aportar información sobre la red.
- **Las variables predictoras se limitaron a las que la empresa conoce antes de instalar** o puede
  observar al momento de la medición. Se excluyeron las consecuencias del servicio medido.
- **La señal de los clientes por cable no se imputa:** es un nulo estructural.
- **No se eliminó ningún registro** de los conjuntos originales.

## Datos personales

El conjunto de trabajo identifica a los clientes por código. Nombre y apellido se excluyeron del
padrón analizado (Ley 25.326 de Protección de Datos Personales). Este repositorio puede ser público
sin exponer datos de ninguna persona.
