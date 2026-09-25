"""
FBJ Internet — Herramienta de decisión de capacidad y velocidad
Prácticas Profesionalizantes · ITSE · Claramunt, Federico · Molina, Javier

Qué es: una herramienta de decisión para el titular de FBJ Internet. Estima la velocidad
efectiva que va a recibir un cliente y, a partir de ahí, cuántos clientes activos admite cada
nodo. El modelo está adentro; las pantallas están organizadas alrededor de decisiones.

Qué NO es: no lee datos en vivo de la red. Trabaja sobre 90 días de análisis (01/06 a 29/08/2026).
Los datos son de origen simulado, construidos a partir de parámetros operativos relevados con la
entidad. Ver la pantalla «Datos y límites».
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).parent
NUM = ["distancia_m", "senal_dbm", "usuarios_activos_antena"]
CAT = ["tipo_conexion", "id_antena", "franja", "lluvia", "es_fin_de_semana", "es_feriado"]
OBJ = "velocidad_efectiva_mbps"
FRANJAS = ["Madrugada", "Mañana", "Tarde", "Noche"]
SIN_SENAL = -100          # centinela para los clientes por cable: no tienen enlace inalámbrico
PRECIO, MARGEN, COSTO_ANTENA = 20000, 18500, 65000
COSTOS_FIJOS = 200000 + 2 * COSTO_ANTENA

st.set_page_config(page_title="FBJ Internet · Velocidad y capacidad", page_icon="📶", layout="wide")


# --------------------------------------------------------------------------- datos y modelo
@st.cache_data
def cargar_datos():
    med = pd.read_csv(BASE / "data" / "mediciones_procesado.csv", parse_dates=["fecha"])
    med["senal_dbm"] = med.senal_dbm.fillna(SIN_SENAL)
    cli = pd.read_csv(BASE / "data" / "clientes_procesado.csv")
    return med, cli


@st.cache_data
def cargar_meta():
    return json.loads((BASE / "modelo_metricas.json").read_text(encoding="utf-8"))


@st.cache_resource
def entrenar(_med):
    """Entrena al arrancar desde el mismo conjunto y con el mismo procedimiento del script de
    entrenamiento. Evita depender de una versión exacta de las librerías para deserializar."""
    def armar():
        return Pipeline([
            ("pre", ColumnTransformer([
                ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), NUM),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])),
            ("m", RandomForestRegressor(n_estimators=400, min_samples_leaf=5, random_state=42))])
    modelo = armar().fit(_med[NUM + CAT], _med[OBJ])
    lineal = armar().set_params(m=Ridge(alpha=1.0, random_state=42)).fit(_med[NUM + CAT], _med[OBJ])
    return {"Random Forest": modelo, "Ridge (lineal regularizada)": lineal}


def predecir(modelos, fila):
    """Devuelve la velocidad estimada por el modelo de mejor desempeño."""
    return float(modelos["Random Forest"].predict(pd.DataFrame([fila]))[0])


def fila_de_prediccion(tipo, nodo, distancia, senal, activos, franja, lluvia, finde, feriado):
    return {"distancia_m": distancia,
            "senal_dbm": SIN_SENAL if tipo == "Cable" else senal,
            "usuarios_activos_antena": activos,
            "tipo_conexion": tipo, "id_antena": nodo, "franja": franja,
            "lluvia": lluvia, "es_fin_de_semana": finde, "es_feriado": feriado}


def umbral_para(curva, objetivo, tipo=None):
    """Máximo de clientes activos simultáneos por nodo que sostiene la velocidad objetivo."""
    col = "mediana" if tipo is None else ("cable" if tipo == "Cable" else "antena")
    ok = [c["activos"] for c in curva if c.get(col) is not None and c[col] >= objetivo]
    return max(ok) if ok else 0


med, cli = cargar_datos()
meta = cargar_meta()
modelos = entrenar(med)
mae = meta["metricas"]["Random Forest"]["mae"]
r2 = meta["metricas"]["Random Forest"]["r2"]
curva = meta["curva"]

# --------------------------------------------------------------------------- barra lateral
st.sidebar.title("📶 FBJ Internet")
st.sidebar.caption("Herramienta de decisión · estimación de velocidad y capacidad por nodo")
pantalla = st.sidebar.radio("Pantalla", [
    "1 · Panel del nodo",
    "2 · Cartera por zona",
    "3 · Simulador de alta",
    "4 · Capacidad del nodo",
    "5 · Decisión de inversión",
    "6 · Datos y límites",
])
st.sidebar.divider()
st.sidebar.metric("Error del modelo", f"± {mae} Mbps", help=f"Error absoluto medio sobre días no usados en el entrenamiento (R² {r2}).")
st.sidebar.caption("Claramunt, Federico · Molina, Javier\n\nITSE · Prácticas Profesionalizantes")

# =========================================================================== 1 panel del nodo
if pantalla.startswith("1"):
    st.title("Panel del nodo")
    st.caption("Estado observado de cada antena en la ventana analizada (01/06 al 29/08/2026).")

    c1, c2, c3 = st.columns(3)
    nodo = c1.selectbox("Antena satelital", ["A", "B"], help="FBJ opera dos antenas Starlink, con 30 clientes cada una.")
    franja = c2.selectbox("Franja horaria", FRANJAS, index=3)
    objetivo = c3.slider("Velocidad mínima aceptable (Mbps)", 15, 30, 20,
                         help="El valor por debajo del cual el cliente no está recibiendo lo ofrecido.")

    d = med[(med.id_antena == nodo) & (med.franja == franja)]
    act_max, act_med = int(d.usuarios_activos_antena.max()), float(d.usuarios_activos_antena.median())
    umbral = umbral_para(curva, objetivo)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Máximo de clientes activos a la vez", act_max, help="Máximo observado en esta franja: 25 clientes están asociados al nodo, pero no todos usan la red al mismo tiempo.")
    k2.metric("Velocidad mediana del nodo", f"{d[OBJ].median():.1f} Mbps", delta=f"{d[OBJ].median()-objetivo:+.1f} vs objetivo", delta_color="normal")
    k3.metric("Velocidad mediana por cable", f"{d[d.tipo_conexion=='Cable'][OBJ].median():.1f} Mbps")
    k4.metric("Velocidad mediana por antena", f"{d[d.tipo_conexion=='Antena'][OBJ].median():.1f} Mbps")

    bajo = 100 * (d[OBJ] < objetivo).mean()
    if act_max <= umbral - 2:
        st.success(f"**Con margen.** El nodo opera con {act_max} clientes activos simultáneos y el límite para sostener {objetivo} Mbps es de {umbral}. El {bajo:.0f}% de las mediciones de esta franja queda por debajo del objetivo.")
    elif act_max <= umbral:
        st.warning(f"**Al límite.** El nodo alcanza {act_max} clientes activos simultáneos y el límite para sostener {objetivo} Mbps es de {umbral}. El {bajo:.0f}% de las mediciones queda por debajo del objetivo.")
    else:
        st.error(f"**Saturado.** El nodo llega a {act_max} clientes activos simultáneos y el límite para sostener {objetivo} Mbps es de {umbral}. El {bajo:.0f}% de las mediciones de esta franja queda por debajo del objetivo.")

    st.subheader("Curva de saturación")
    st.caption("Velocidad mediana según cuántos clientes estén transfiriendo datos al mismo tiempo en el nodo. "
               "Hasta 13 activos la velocidad se mantiene; a partir de 14 empieza a caer.")
    cv = pd.DataFrame(curva).set_index("activos")[["mediana", "cable", "antena"]]
    cv.columns = ["Todos los clientes", "Clientes por cable", "Clientes por antena"]
    st.line_chart(cv)
    st.caption(f"Límite de admisión según velocidad objetivo: **hasta {umbral} clientes activos simultáneos** "
               f"({umbral_para(curva, objetivo, 'Cable')} si el nodo se mira solo por cable, "
               f"{umbral_para(curva, objetivo, 'Antena')} si se mira solo por antena).")

    st.divider()
    g1, g2 = st.columns(2)
    g1.metric("Carga de la antena satelital (máx.)", f"{d.carga_antena_pct.max():.1f} %")
    g2.metric("Carga media", f"{d.carga_antena_pct.mean():.1f} %")
    st.info("**La antena satelital no es el límite.** Su capacidad de 400 Mbps nunca se usa más del 54,2%. "
            "Las antenas quedan al 54% y los clientes igual reciben la mitad de lo ofrecido: la pérdida está "
            "en el tramo final que llega al domicilio, no en el servicio satelital.")

# =========================================================================== 2 cartera
elif pantalla.startswith("2"):
    st.title("Cartera por zona")
    st.caption("Dónde se concentra el problema. Se muestran zonas y códigos de cliente, nunca datos personales.")

    j = med.merge(cli[["id_cliente", "zona", "tipo_conexion", "distancia_m"]], on="id_cliente",
                  how="left", suffixes=("", "_cli"))
    obj = st.slider("Velocidad mínima aceptable (Mbps)", 15, 30, 20)
    g = j.groupby("zona").agg(mediciones=(OBJ, "size"),
                              velocidad_mediana=(OBJ, "median"),
                              bajo_objetivo=(OBJ, lambda s: 100 * (s < obj).mean()),
                              distancia_media=("distancia_m", "mean")).round(1)
    g["clientes"] = j.groupby("zona").id_cliente.nunique()
    g["inalambricos_%"] = j.groupby("zona").tipo_conexion.apply(lambda s: round(100 * (s == "Antena").mean())).round(0)
    g = g.sort_values("velocidad_mediana")
    st.dataframe(g.rename(columns={"velocidad_mediana": "velocidad mediana (Mbps)", "bajo_objetivo": "% bajo el objetivo",
                                   "distancia_media": "distancia media (m)", "inalambricos_%": "% clientes por antena"}),
                 width="stretch")
    st.bar_chart(g["velocidad_mediana"])
    peor = g.index[0]
    st.warning(f"**La zona con peor desempeño es {peor}**, con {g.loc[peor,'velocidad_mediana']:.1f} Mbps de mediana y "
               f"{g.loc[peor,'bajo_objetivo']:.0f}% de las mediciones por debajo del objetivo, sobre "
               f"{int(g.loc[peor,'clientes'])} clientes. La distancia media de esa zona es de "
               f"{g.loc[peor,'distancia_media']:.0f} m al nodo, contra {g['distancia_media'].min():.0f} m de la mejor.")
    st.caption("El criterio de priorización es operativo: la zona que menos recibe y a la que más cuesta llegar "
               "es la que primero hay que intervenir.")

# =========================================================================== 3 simulador de alta
elif pantalla.startswith("3"):
    st.title("Simulador de alta")
    st.caption("Antes de instalar: qué velocidad va a recibir este cliente. Es el criterio que hoy no existe.")

    c1, c2, c3 = st.columns(3)
    tipo = c1.selectbox("Modalidad de conexión", ["Antena", "Cable"],
                        help="Cable llega en mejores condiciones; la antena inalámbrica depende de la distancia y de la señal.")
    nodo = c2.selectbox("Nodo al que se conectaría", ["A", "B"])
    franja = c3.selectbox("Franja de mayor uso", FRANJAS, index=3)

    c4, c5, c6 = st.columns(3)
    distancia = c4.slider("Distancia al nodo (metros)", 40, 900, 250, step=10)
    senal = c5.slider("Señal medida en el domicilio (dBm)", -85, -50, -70, disabled=(tipo == "Cable"),
                      help="Con antena: medir en el domicilio antes de instalar. Con cable no aplica.")
    activos = c6.slider("Clientes activos en el nodo en hora pico", 1, 25, 13)

    c7, c8, c9, c10 = st.columns(4)
    lluvia = c7.checkbox("Día de lluvia")
    finde = c8.checkbox("Fin de semana")
    feriado = c9.checkbox("Feriado")
    objetivo = c10.slider("Velocidad mínima aceptable (Mbps)", 15, 30, 20)

    fila = fila_de_prediccion(tipo, nodo, distancia, senal, activos, franja,
                              int(lluvia), int(finde), int(feriado))
    est = predecir(modelos, fila)
    est = min(est, 35.0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Velocidad estimada", f"{est:.1f} Mbps", delta=f"{est-objetivo:+.1f} vs objetivo",
              delta_color="normal")
    m2.metric("Rango probable", f"{max(2, est-mae):.1f} – {est+mae:.1f} Mbps",
              help=f"El modelo tiene un error absoluto medio de {mae} Mbps.")
    lim = umbral_para(curva, objetivo, tipo)
    m3.metric("Activos que admite el nodo", f"{lim}" if lim else "ninguno",
              help=f"Máximo de clientes activos simultáneos para sostener {objetivo} Mbps en clientes por {tipo.lower()}.")

    if est >= objetivo:
        st.success(f"**Cumple.** Con estos datos, el cliente recibiría unos {est:.1f} Mbps, por encima del "
                   f"objetivo de {objetivo} Mbps.")
    elif est >= objetivo - mae:
        st.warning(f"**Duda.** La estimación ({est:.1f} Mbps) está dentro del margen de error del objetivo "
                   f"({objetivo} Mbps). Conviene medir la señal en el domicilio antes de confirmar la instalación.")
    else:
        st.error(f"**No cumple.** Con estos datos el cliente recibiría unos {est:.1f} Mbps, por debajo del "
                 f"objetivo de {objetivo} Mbps. La instalación quedaría por debajo de lo ofrecido.")

    if activos > lim and lim:
        st.error(f"Además, el nodo ya tendría {activos} clientes activos simultáneos y aguanta {lim} para "
                 f"sostener {objetivo} Mbps. La velocidad estimada ya descuenta esa carga.")
    st.info("**Los 400 metros son el punto de quiebre.** Hasta ahí el servicio entrega entre 27 y 30 Mbps; "
            "a partir de esa distancia cae a 16 Mbps y el 84% de las mediciones queda por debajo de 20 Mbps. "
            "El mismo cliente a 300 m y a 700 m son dos ventas distintas.")

# =========================================================================== 4 capacidad
elif pantalla.startswith("4"):
    st.title("Capacidad del nodo")
    st.caption("Cuántos clientes más admite cada nodo, y qué cuesta en velocidad cada cliente de más.")

    objetivo = st.slider("Velocidad mínima aceptable (Mbps)", 15, 30, 20)
    activos = st.slider("Clientes activos simultáneos en el nodo (hora pico)", 1, 22, 13)
    idx = max(c["activos"] for c in curva if c["activos"] <= activos)
    fila = next(c for c in curva if c["activos"] == idx)
    lim = umbral_para(curva, objetivo)

    m1, m2, m3 = st.columns(3)
    m1.metric("Velocidad mediana del nodo", f"{fila['mediana']:.1f} Mbps")
    m2.metric("Clientes por cable", f"{fila['cable']:.1f} Mbps" if fila["cable"] else "—")
    m3.metric("Clientes por antena", f"{fila['antena']:.1f} Mbps" if fila["antena"] else "—")

    if activos > lim:
        st.error(f"Con {activos} clientes activos simultáneos el nodo está por encima del límite de {lim} "
                 f"para sostener {objetivo} Mbps.")
    else:
        margen = lim - activos
        st.success(f"El nodo admite hasta **{lim} clientes activos simultáneos** para sostener {objetivo} Mbps. "
                   f"Con {activos} activos queda un margen de {margen} clientes simultáneos más.")

    st.divider()
    st.subheader("Cuánto cuesta cada cliente de más")
    efect = meta["efectos_inalambricos"]
    st.markdown(f"Cada cliente activo adicional en el mismo nodo cuesta alrededor de "
                f"**{abs(efect['activos']):.2f} Mbps** de velocidad mediana, y el efecto se acumula sobre los "
                f"clientes que ya están conectados: no se reparte entre todos, lo pagan todos.")
    st.markdown(f"En los clientes inalámbricos, además, cada 100 metros de distancia al nodo cuestan "
                f"**{abs(efect['distancia_m_por_100m']):.2f} Mbps**, y cada 10 dBm de mejor señal suman "
                f"**{efect['senal_dbm']*10:.1f} Mbps**.")
    st.caption("Efectos estimados dentro del grupo de clientes inalámbricos, controlando por carga del nodo, "
               "franja horaria y lluvia. La distancia y la señal miden la misma condición física del enlace: "
               "no deben sumarse entre sí.")

    st.subheader("Toda la curva")
    cv = pd.DataFrame(curva)[["activos", "n", "mediana", "cable", "antena"]]
    cv.columns = ["Activos simultáneos", "Mediciones", "Mediana (Mbps)", "Por cable (Mbps)", "Por antena (Mbps)"]
    st.dataframe(cv, width="stretch", hide_index=True)

# =========================================================================== 5 inversión
elif pantalla.startswith("5"):
    st.title("Decisión de inversión")
    st.caption("Dónde rinde más la próxima inversión: ampliar el satélite o mejorar el tramo final.")

    st.subheader("Lo que cuesta cada opción")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sumar una tercera antena Starlink**")
        st.metric("Costo mensual", f"$ {COSTO_ANTENA:,.0f}".replace(",", "."))
        st.metric("Costo anual", f"$ {COSTO_ANTENA*12:,.0f}".replace(",", "."))
        st.metric("Clientes necesarios solo para pagarla", f"{(COSTO_ANTENA/MARGEN):.1f}")
        st.caption("Se paga haya o no clientes nuevos, porque es un costo fijo mensual.")
    with c2:
        st.markdown("**Intervenir el tramo final de los clientes inalámbricos**")
        st.metric("Alcance", f"{int((cli.tipo_conexion=='Antena').sum())} clientes", help="Los clientes conectados por antena inalámbrica, que son los que reciben menos velocidad.")
        st.metric("Velocidad mediana que reciben hoy", f"{med[med.tipo_conexion=='Antena'][OBJ].median():.1f} Mbps")
        st.metric("Velocidad mediana de los clientes por cable", f"{med[med.tipo_conexion=='Cable'][OBJ].median():.1f} Mbps")
        st.caption("Reorientar antenas, acercar el nodo o pasar el cliente a cable tiene un costo por cliente, no un costo fijo creciente.")

    st.divider()
    st.subheader("La evidencia")
    e1, e2, e3 = st.columns(3)
    e1.metric("Uso máximo del satélite", f"{med.carga_antena_pct.max():.1f} %", help="La capacidad de 400 Mbps por antena nunca se usa más de la mitad.")
    e2.metric("Diferencia cable vs antena", f"{med[med.tipo_conexion=='Cable'][OBJ].median()-med[med.tipo_conexion=='Antena'][OBJ].median():.1f} Mbps")
    e3.metric("Mediciones bajo 20 Mbps", f"{100*(med[OBJ]<20).mean():.0f} %")

    st.error("**Ampliar el satélite no mueve el problema.** Si la antena está al 54% de uso y los clientes "
             "igual reciben la mitad de lo ofrecido, agregar capacidad no cambia la velocidad que llega al "
             "domicilio: el límite está en el tramo final. Antes de comprar la tercera antena conviene "
             "intervenir los clientes inalámbricos alejados y volver a medir.")

    st.divider()
    st.subheader("Capacidad remanente de la instalación actual")
    objetivo = st.slider("Velocidad mínima aceptable (Mbps)", 15, 30, 20)
    lim = umbral_para(curva, objetivo)
    actual = int(med.usuarios_activos_antena.max())
    r1, r2, r3 = st.columns(3)
    r1.metric("Clientes activos simultáneos hoy (máx.)", actual)
    r2.metric("Límite del nodo", lim)
    r3.metric("Margen", lim - actual, delta_color="off")
    if lim - actual > 0:
        st.info(f"Queda margen para {lim-actual} clientes activos simultáneos más por nodo antes de cruzar el "
                f"objetivo de {objetivo} Mbps. Ojo: el límite no es una cantidad de clientes contratados sino de "
                f"clientes usando la red al mismo tiempo, y eso depende de la hora.")
    else:
        st.warning("Con los clientes activos actuales el nodo ya está en el límite del objetivo elegido. "
                   "Bajar el objetivo a un valor realista o intervenir el tramo final son las salidas.")

    st.divider()
    st.subheader("El negocio en números")
    n1, n2, n3 = st.columns(3)
    n1.metric("Abono mensual", f"$ {PRECIO:,.0f}".replace(",", "."))
    n2.metric("Margen por cliente", f"$ {MARGEN:,.0f}".replace(",", "."))
    n3.metric("Punto de equilibrio", f"{COSTOS_FIJOS/MARGEN:.1f} clientes")
    st.caption(f"Con {len(cli)} clientes el margen mensual es del orden de "
               f"$ {len(cli)*MARGEN-COSTOS_FIJOS:,.0f}".replace(",", ".") +
               ". Cada baja por mal servicio cuesta el margen de ese cliente todos los meses.")

# =========================================================================== 6 datos y límites
else:
    st.title("Datos y límites")
    st.caption("Qué sostiene esta herramienta y qué no. Se lee antes de tomar cualquier decisión con ella.")

    st.subheader("Origen de los datos")
    st.warning("**Los datos son de origen simulado**, construidos a partir de los parámetros operativos y "
               "económicos relevados con el titular de FBJ Internet: el precio único del plan ($20.000), la "
               "velocidad ofrecida (30 a 40 Mbps), las dos antenas Starlink con plan Residencial Estándar "
               "($65.000 mensuales cada una), la capacidad efectiva medida por antena (400 Mbps), los 60 "
               "clientes, el reparto 30/30 entre las antenas, la modalidad de conexión (40 por antena "
               "inalámbrica y 20 por cable) y los dos puntos de acceso.")
    st.markdown("La empresa **no tiene registros históricos de velocidad efectiva ni de utilización de sus "
                "nodos**: nunca midió esos datos. Por eso el conjunto se construye y se declara de este modo. "
                "Los resultados describen el comportamiento del modelo construido sobre los parámetros "
                "relevados, y serán válidos para la red real en la medida en que esos parámetros la representen.")

    st.subheader("Desempeño del modelo")
    d1, d2, d3 = st.columns(3)
    d1.metric("Error absoluto medio", f"{mae} Mbps", help="Sobre días que el modelo no vio durante el entrenamiento.")
    d2.metric("R²", f"{r2}")
    d3.metric("Mediciones usadas", f"{meta['n_entrenamiento']:,}".replace(",", "."))
    st.caption(f"Entrenado con los días 1 a 60 y evaluado con los días 61 a 90 de la ventana "
               f"{meta['ventana'][0]} a {meta['ventana'][1]}. La partición es temporal, no aleatoria: "
               f"el modelo se evalúa sobre días que no vio, que es la situación real de uso.")

    st.subheader("Límites que hay que tener presentes")
    st.markdown(f"""
- **El objetivo está censurado en 35 Mbps.** El 14,2% de las mediciones alcanza el techo del plan. En ese
  extremo el modelo no puede distinguir entre un cliente que recibe exactamente lo ofrecido y otro que
  recibiría más si el plan lo permitiera: ahí el error no mide una falla del modelo.
- **El efecto de la concurrencia es una cota superior.** La velocidad se modeló repartiendo la capacidad
  del nodo entre los clientes activos, pero la demanda medida nunca supera el 54% de esa capacidad. En
  una red real, una capacidad que sobra no debería limitar a nadie: el impacto real de la carga
  probablemente sea menor y el límite de admisión podría ser más alto.
- **La ventana es de 90 días**, sin historia anterior. No permite analizar estacionalidad.
- **8 de los 60 clientes no tienen mediciones** (suspendidos y bajas previas a la ventana).
- **Esta herramienta no lee datos en vivo.** No es un sistema de monitoreo: es una herramienta de
  decisión construida sobre 90 días de análisis. Para monitorear haría falta instrumentar la red.
""")

    st.subheader("Decisiones de análisis que sostienen los resultados")
    st.markdown("""
- **El consumo por franja se excluyó del modelo.** Correlaciona 0,954 con la velocidad, pero no la
  explica: el consumo se calcula a partir de la velocidad lograda. Usarlo como variable predictora
  inflaría el desempeño del modelo en 0,07 de R² sin aportar información sobre la red.
- **Las variables predictoras son las que la empresa conoce antes de instalar** o puede observar al
  momento de la medición: distancia, señal, modalidad de conexión, nodo, franja, lluvia y condiciones
  del día. Se excluyeron las consecuencias del servicio medido.
- **La señal de los clientes por cable no se imputa.** Es un nulo estructural: no tienen enlace
  inalámbrico y no hay valor que inventar.
- **No se eliminó ningún registro.** Los valores atípicos de consumo corresponden a hogares de uso
  intensivo, que son justamente los que presionan la capacidad.
- **El conjunto de trabajo no tiene datos personales.** Nombre y apellido se excluyeron del padrón
  analizado (Ley 25.326).
""")

    st.divider()
    st.caption("FBJ Internet · Herramienta de decisión · Claramunt, Federico · Molina, Javier · "
               "ITSE · Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial")
