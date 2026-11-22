# BatteryGuard AI
Proyecto final de Fundamentos de la IA (LITIID-73), UT Cancún.

## URLs en producción

- **Aplicación web (Frontend en Vercel):** https://batteryguard-ai-front.vercel.app
- **API Backend + IA (Render):** https://batteryguard-ai.onrender.com
- **Documentación interactiva (Swagger):** https://batteryguard-ai.onrender.com/docs
- **Estado del servicio:** https://batteryguard-ai.onrender.com/health
- **Base de datos:** PostgreSQL 17 en Supabase

> Nota: En el plan gratuito de Render el servicio entra en reposo tras 15 minutos de inactividad y tarda ~35–45 segundos en responder la primera solicitud.

---

## Versiones exactas de Python y frameworks utilizados

| Componente | Versión exacta | Uso |
| :--- | :--- | :--- |
| **Python** | `3.13.14` *(entrenamiento original documentado en `3.15.2`)* | Entorno local y despliegue en Render |
| **FastAPI** | `0.165.0` | Framework REST API y validación de esquemas |
| **Uvicorn** | `0.54.0` | Servidor ASGI para desarrollo y producción |
| **SQLAlchemy** | `2.1.9` | ORM (SQLite local / PostgreSQL en producción) |
| **psycopg[binary]** | `3.3.9` | Driver PostgreSQL para Supabase |
| **PyJWT** | `2.15.1` | Autenticación mediante tokens JWT (`HS256`) |
| **scikit-learn** | `1.9.1` | Pipeline de preprocesamiento y Regresión Logística |
| **pandas** | `3.1.1` | Manipulación de datos e inferencia |
| **numpy** | `2.5.4` | Cálculo numérico |
| **joblib** | `1.6.0` | Serialización y carga de `models/modelo_candidato.joblib` |
| **imbalanced-learn** | `0.14.2` | Experimentación con balanceo (`SMOTENC`) en entrenamiento |

---

## Dependencias requeridas (`requirements.txt`)

Todas las dependencias exactas se encuentran fijadas en [`requirements.txt`](requirements.txt):

```text
numpy==2.5.4
pandas==3.1.1
scikit-learn==1.9.1
joblib==1.6.0
imbalanced-learn==0.14.2
fastapi==0.165.0
uvicorn==0.54.0
sqlalchemy==2.1.9
PyJWT==2.15.1
psycopg[binary]==3.3.9
```

Instalación recomendada en entorno virtual:

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Variables de entorno necesarias (`.env.example`)

Consulta la plantilla completa en [`.env.example`](.env.example).

| Variable | Requerida | Valor por defecto / ejemplo | Descripción |
| :--- | :--- | :--- | :--- |
| `PYTHON_VERSION` | Producción (Render) | `3.13.14` | Fija la versión de Python en Render |
| `DATABASE_URL` | Producción *(opcional en local)* | `sqlite:///batteryguard.db` (si se omite) | URL PostgreSQL Session Pooler de Supabase (puerto `5432`) |
| `JWT_SECRET_KEY` | Producción *(opcional en local)* | Clave aleatoria larga | Secreto para firmar tokens JWT |
| `JWT_EXPIRE_HOURS` | No | `24` | Vigencia del token JWT en horas |
| `CORS_ORIGINS` | Producción | `https://batteryguard-ai-front.vercel.app,http://localhost:5173` | Orígenes permitidos separados por coma |
| `ADMIN_EMAIL` | No | `admin@batteryguard.ai` | Correo del administrador creado al inicializar BD vacía |
| `ADMIN_PASSWORD` | No | `admin1234` | Contraseña del administrador inicial |

---

## Comandos exactos de arranque

### 1. Arrancar la API del backend (local)

```powershell
# Windows (PowerShell) desde la raíz del repositorio
.\.venv\Scripts\uvicorn.exe src.api.app:app --reload --host 127.0.0.1 --port 8000
```

```bash
# Linux / macOS (con entorno virtual activo)
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### 2. Arrancar la API en producción (Render)

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn src.api.app:app --host 0.0.0.0 --port $PORT`

### 3. Reproducir el pipeline de Machine Learning localmente

```bash
pip install -r requirements.txt
python src/data/preprocess.py
python src/models/train.py --data data/processed/ev_battery_failure_clean.csv --output .
python src/models/predict.py
python src/models/ablation.py --data data/processed/ev_battery_failure_clean.csv --output .
```

---

## Credenciales de prueba y control de roles (RBAC)

El sistema maneja **2 roles** (`Administrador` y `Empleado`) autenticados por JWT (`Authorization: Bearer <token>`) con contraseñas hasheadas (`PBKDF2-SHA256`).

- **Cuenta Administrador inicial (seed automático):**
  - **Correo:** `admin@batteryguard.ai`
  - **Contraseña:** `admin1234`
- **Flujo para probar rol `Empleado`:**
  1. Inicia sesión como **Administrador** y crea un usuario con rol `Empleado` en **Agregar empleado** (`POST /employees`).
  2. Si el Administrador restablece la contraseña (`POST /employees/{id}/reset-password`), el sistema asigna la contraseña temporal **`cambiame1234`** y activa `mustChangePassword: true`, bloqueando el acceso hasta definir una nueva contraseña personal.
  3. El rol `Empleado` puede consultar vehículos, ejecutar simulaciones IA, consultar historial y cambiar su propia contraseña, pero recibe **HTTP 403** si intenta gestionar empleados, dar de alta vehículos o editar umbrales BMS.

### Resumen de endpoints implementados (`src/api/app.py`)

| Método | Endpoint | Acceso | Descripción |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Público | Estado del servicio y modelo activo |
| `POST` | `/predict` | Público | Inferencia directa según contrato base |
| `GET` | `/presets` | Público | Lista de vehículos con perfil BMS base |
| `POST` | `/auth/login` | Público | Inicio de sesión (`{ email, password }`) |
| `GET` | `/auth/me` | Autenticado | Sesión del usuario actual |
| `POST` | `/auth/change-password` | Autenticado | Cambio obligatorio o voluntario de contraseña |
| `GET` / `POST` | `/employees` | Solo Administrador | Listar y crear empleados |
| `POST` | `/employees/{id}/reset-password` | Solo Administrador | Asigna `cambiame1234` y `mustChangePassword: true` |
| `GET` | `/vehicles` | Autenticado | Lista vehículos registrados en BD |
| `POST` | `/vehicles` | Solo Administrador | Alta de vehículo eléctrico (placa única) |
| `PATCH` | `/vehicles/{id}/status` | Autenticado | Actualiza estado en taller (`En ingreso`, `En diagnóstico`, `En carga`, `Listo`) |
| `POST` | `/simulate` | Autenticado | Simulación BMS + inferencia IA + guardado en historial |
| `GET` | `/evaluations` | Autenticado | Historial de simulaciones (opcional `?vehicleId=`) |
| `GET` | `/settings/thresholds` | Autenticado | Consulta umbrales BMS |
| `PUT` | `/settings/thresholds` | Solo Administrador | Actualiza umbrales BMS |

---

## Contrato de API para integración (`POST /predict`)

URL local: `http://127.0.0.1:8000/predict` (Swagger: `http://127.0.0.1:8000/docs`). El cuerpo requiere un `preset_id` que exista en `models/presets.json`; `overrides` es opcional.

```json
{
  "preset_id": "EV109069",
  "overrides": {
    "cell_voltage_avg": 3.649,
    "cell_temperature_max": 60.0,
    "terrain_type": "Flat"
  }
}
```

Validaciones de overrides:

- `cell_voltage_avg`: número entre 3.06 y 3.85 V.
- `cell_temperature_max`: número entre −4.95 y 86.98 °C.
- `terrain_type`: `Flat`, `Coastal`, `Mountainous`, `Hilly`, `Desert` o `Desconocido`.
- No se admiten otras claves en `overrides`. Las entradas inválidas responden HTTP 422.

Ejemplo real observado en Swagger para esa solicitud (la latencia varía entre ejecuciones):

```json
{
  "prediccion": 0,
  "etiqueta": "Sin falla detectada",
  "score_riesgo": 0.103261,
  "nivel_riesgo": "bajo",
  "recomendaciones": [
    "Sin alerta del clasificador; mantener el plan de seguimiento preventivo."
  ],
  "rsl_meses": null,
  "tiempo_ms": 113.397,
  "modelo": "Logistic Regression regularized C=0.1",
  "umbral_clasificacion": 0.318373
}
```

`score_riesgo` es el score producido por el clasificador; no se ha validado como probabilidad calibrada.

---

## Ablación de variables

Para registrar en `results/experiments.csv` la comparación entre las 60 features y la variante conservadora de 52 features, ejecuta:

```bash
python src/models/ablation.py --data data/processed/ev_battery_failure_clean.csv --output .
```

Ambos escenarios usan split estratificado 80/20, semilla 42, validación cruzada de cinco folds y umbral elegido por F1 máximo en predicciones OOF dentro de train. El test no se usa para elegir el umbral.

## Limitaciones conocidas

- El dataset es sintético; la etiqueta `battery_failure` se genera con factores simulados. La ficha no publica la fórmula exacta ni los factores usados.
- El modelo clasifica la etiqueta sintética del dataset. No se ha validado que anticipe fallas futuras en vehículos o baterías reales.
- El modelo elegido es una Regresión Logística regularizada (`C=0.1`). En test obtuvo F1 positivo 0.7131, precisión 0.6731, recall 0.7581 y ROC-AUC 0.9711. No alcanzó la meta anterior F1 >0.85.
- `score_riesgo` no es una probabilidad calibrada. La salida es binaria (`bajo`/`alto`); no hay estimador RSL y `rsl_meses` queda en `null`.
- La ficha de Kaggle describe licencia CC BY-SA 4.0. Fuente: [EV Battery Health Prediction Dataset (20K)](https://www.kaggle.com/datasets/srisyra02/ev-battery-health-prediction-dataset-20k), owner Srimathy Sivanessan. Al compartir el dataset, conserva la atribución y la licencia.

## Datos
- Archivo: `data/raw/ev_battery_failure_prediction_Dataset.csv`; procesado: `data/processed/ev_battery_failure_clean.csv`.
- Volumen medido: 20,000 filas × 70 columnas. Target `battery_failure`: 18,616 clase 0 (93.08%) y 1,384 clase 1 (6.92%).
