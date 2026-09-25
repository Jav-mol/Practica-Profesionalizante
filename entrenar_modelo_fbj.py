#!/usr/bin/env python3
"""Etapa 5 — Entrena el modelo de estimación de velocidad efectiva y lo serializa para la app.

Decisiones de modelado (justificadas en el documento de la Etapa 5):
  - Se excluye consumo_gb_franja: contiene al objetivo en su fórmula (fuga de información).
  - Las variables predictoras se limitan a las que la empresa conoce antes de instalar o puede
    observar en el momento de la medición. Se excluyen cortes, minutos_sin_servicio, horas_conexion
    y sesiones: son consecuencia del servicio medido, no condiciones de la instalación.
  - Split temporal (días 1-60 / 61-90) para reportar las métricas honestas.
  - El modelo que se publica se reentrena con los 90 días completos.
Uso: /opt/data/venvs/pp/bin/python entrenar_modelo_fbj.py
"""
import json
from pathlib import Path
import joblib
import pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# El script corre tanto en el repositorio del proyecto como dentro del repositorio de la app:
# se busca el conjunto procesado en las ubicaciones posibles en lugar de fijar una ruta absoluta.
AQUI = Path(__file__).resolve().parent
CANDIDATOS = [AQUI.parent / "datos_procesados", AQUI / "data",
              AQUI.parent.parent / "datos_procesados",
              Path("/opt/data/practica-profesionalizante/FBJ/datos_procesados")]
PROC = next((p for p in CANDIDATOS if (p / "mediciones_procesado.csv").exists()), None)
if PROC is None:
    raise SystemExit("no se encontró mediciones_procesado.csv en ninguna ubicación conocida")
APP = AQUI if (AQUI / "app.py").exists() else AQUI.parent / "streamlit"
APP.mkdir(parents=True, exist_ok=True)
BASE = PROC.parent

NUM = ["distancia_m", "senal_dbm", "usuarios_activos_antena"]
CAT = ["tipo_conexion", "id_antena", "franja", "lluvia", "es_fin_de_semana", "es_feriado"]
OBJ = "velocidad_efectiva_mbps"

m = pd.read_csv(f"{PROC}/mediciones_procesado.csv", parse_dates=["fecha"])
# nulo estructural: los clientes por cable no tienen señal. Centinela "no aplica", nunca la mediana.
m["senal_dbm"] = m.senal_dbm.fillna(-100)

def armar(modelo_num, modelo_cat):
    return Pipeline([
        ("pre", ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), modelo_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), modelo_cat)])),
        ("m", None)])

print("=" * 74); print("ENTRENAMIENTO DEL MODELO · FBJ Internet"); print("=" * 74)
print(f"conjunto: {len(m)} mediciones · {m.id_cliente.nunique()} clientes · {m.fecha.nunique()} días")
print(f"variables predictoras ({len(NUM)+len(CAT)}): {NUM + CAT}")

corte = m.fecha.min() + pd.Timedelta(days=60)
tr, te = m.fecha < corte, m.fecha >= corte
print(f"\nsplit temporal: {tr.sum()} para entrenar (días 1-60) · {te.sum()} para evaluar (días 61-90)")

modelos = {
    "Ridge (lineal regularizada)": Ridge(alpha=1.0, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=5, random_state=42),
}
metricas, pipes = {}, {}
for nombre, est in modelos.items():
    pipe = armar(NUM, CAT); pipe.set_params(m=est)
    pipe.fit(m.loc[tr, NUM + CAT], m.loc[tr, OBJ])
    pr = pipe.predict(m.loc[te, NUM + CAT])
    y = m.loc[te, OBJ]
    metricas[nombre] = dict(mae=round(float(mean_absolute_error(y, pr)), 2),
                            rmse=round(float(mean_squared_error(y, pr) ** .5), 2),
                            r2=round(float(r2_score(y, pr)), 3))
    pipes[nombre] = pipe
    print(f"  {nombre:30} MAE {metricas[nombre]['mae']:5.2f} Mbps · RMSE {metricas[nombre]['rmse']:5.2f} · R² {metricas[nombre]['r2']:.3f}")

mejor = "Random Forest"
# ---- interpretación de efectos con el modelo lineal sobre datos de prueba
lin = pipes["Ridge (lineal regularizada)"]
pre = lin.named_steps["pre"]; nombres = list(pre.get_feature_names_out())
coef = pd.Series(lin.named_steps["m"].coef_, index=nombres).sort_values()
esc = pre.named_transformers_["num"].named_steps["esc"].scale_
print("\nefecto de las variables numéricas (Mbps por unidad, en el modelo lineal):")
efectos = {}
for v in NUM:
    i = nombres.index(f"num__{v}")
    efectos[v] = round(float(coef.iloc[i] / esc[NUM.index(v)]), 4)
    if v == "distancia_m": print(f"  {v:26} {efectos[v]*100:+7.2f} Mbps por cada 100 m")
    else: print(f"  {v:26} {efectos[v]:+7.2f} Mbps por unidad")
print("\nvariables categóricas con mayor efecto:")
for k, v in coef.tail(4).items():
    if k.startswith("cat__"): print(f"  {k.replace('cat__',''):32} {v:+6.2f} Mbps")

# La señal es un nulo estructural en los clientes por cable, y dentro del grupo inalámbrico la
# distancia y la señal miden la misma condición física (la calidad del enlace), por lo que sus
# coeficientes se reparten y cambian de signo si se estiman juntos. Se estima el efecto de cada
# una por separado, dentro del grupo inalámbrico, y se declara que no son efectos independientes.
wire = m[(m.tipo_conexion == "Antena") & tr]

def efecto_wire(variable):
    pipe = armar([variable, "usuarios_activos_antena"], ["franja", "lluvia", "id_antena"])
    pipe.set_params(m=Ridge(alpha=1.0, random_state=42))
    pipe.fit(wire[[variable, "usuarios_activos_antena"] + ["franja", "lluvia", "id_antena"]], wire[OBJ])
    esc = pipe.named_steps["pre"].named_transformers_["num"].named_steps["esc"].scale_
    c = pd.Series(pipe.named_steps["m"].coef_, index=pipe.named_steps["pre"].get_feature_names_out())
    return round(float(c[f"num__{variable}"] / esc[0]), 4), round(float(c["num__usuarios_activos_antena"] / esc[1]), 4)

ef_dist, ef_act_d = efecto_wire("distancia_m")
ef_sen, ef_act_s = efecto_wire("senal_dbm")
efectos_wire = {"distancia_m_por_100m": round(ef_dist * 100, 2), "senal_dbm": ef_sen,
                "activos": round((ef_act_d + ef_act_s) / 2, 3),
                "activos_con_distancia": ef_act_d, "activos_con_senal": ef_act_s}
print("\nefectos dentro del grupo inalámbrico (cada variable estimada por separado):")
print(f"  distancia   {efectos_wire['distancia_m_por_100m']:+6.2f} Mbps por cada 100 m")
print(f"  señal       {efectos_wire['senal_dbm']:+6.3f} Mbps por dBm  ({efectos_wire['senal_dbm']*10:+.1f} Mbps por cada 10 dBm)")
print(f"  activos     {efectos_wire['activos']:+6.2f} Mbps por cliente activo más")

imp = pd.Series(pipes[mejor].named_steps["m"].feature_importances_,
                index=pipes[mejor].named_steps["pre"].get_feature_names_out()).sort_values(ascending=False)
print(f"\nimportancia de variables ({mejor}):")
for k, v in imp.head(6).items(): print(f"  {k:38} {v:.3f}")

# ---------------------------------------------------------------- modelo publicado
pub = {}
for nombre, est in modelos.items():
    p = armar(NUM, CAT); p.set_params(m=est)
    p.fit(m[NUM + CAT], m[OBJ])          # reentrenado con los 90 días
    pub[nombre] = p
print(f"\nmodelos publicados reentrenados con las {len(m)} mediciones completas")

# ---------------------------------------------------------------- regla de admisión
cur = m.groupby("usuarios_activos_antena").agg(n=(OBJ, "size"), mediana=(OBJ, "median")).reset_index()
por_tipo = m.pivot_table(index="usuarios_activos_antena", columns="tipo_conexion",
                         values=OBJ, aggfunc="median").round(1)
cur = cur.merge(por_tipo, on="usuarios_activos_antena", how="left")
cur = cur[cur.n >= 20]
curva = [{"activos": int(r.usuarios_activos_antena), "n": int(r.n), "mediana": round(float(r.mediana), 1),
          "cable": None if pd.isna(r.Cable) else round(float(r.Cable), 1),
          "antena": None if pd.isna(r.Antena) else round(float(r.Antena), 1)} for r in cur.itertuples()]
umbrales = {}
for objetivo in (15, 18, 20, 25, 30):
    ok = [c["activos"] for c in curva if c["mediana"] >= objetivo]
    umbrales[objetivo] = max(ok) if ok else None
print("\nregla de admisión (máximo de clientes activos simultáneos por nodo para sostener cada velocidad objetivo):")
for k, v in umbrales.items(): print(f"  objetivo {k:2} Mbps -> hasta {v} activos")

# ---------------------------------------------------------------- salida
joblib.dump({"modelos": pub, "num": NUM, "cat": CAT, "objetivo": OBJ,
             "metricas": metricas, "mejor": mejor, "efectos": efectos,
             "importancias": {k: round(float(v), 4) for k, v in imp.items()},
             "curva": curva, "umbrales": umbrales, "efectos_inalambricos": efectos_wire,
             "n_entrenamiento": int(len(m)), "ventana": [str(m.fecha.min().date()), str(m.fecha.max().date())]},
            f"{APP}/modelo_fbj.joblib")
with open(f"{APP}/modelo_metricas.json", "w", encoding="utf-8") as fh:
    json.dump({"metricas": metricas, "efectos": efectos, "efectos_inalambricos": efectos_wire, "umbrales": umbrales, "curva": curva,
               "importancias": {k: round(float(v), 4) for k, v in imp.items()},
               "n_entrenamiento": int(len(m)), "ventana": [str(m.fecha.min().date()), str(m.fecha.max().date())],
               "variables_num": NUM, "variables_cat": CAT},
              fh, ensure_ascii=False, indent=1)
print(f"\nguardado: streamlit/modelo_fbj.joblib y streamlit/modelo_metricas.json")
