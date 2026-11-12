from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "modelo_candidato.joblib"
SCHEMA_PATH = ROOT / "models" / "schema.json"
PRESETS_PATH = ROOT / "models" / "presets.json"

_MODEL = joblib.load(MODEL_PATH)
_SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
_PRESETS = json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
_ALLOWED_OVERRIDES = {"cell_voltage_avg", "cell_temperature_max", "terrain_type"}
_TERRAIN = {"Flat", "Coastal", "Mountainous", "Hilly", "Desert", "Desconocido"}

def predecir(datos_entrada: dict) -> dict:
    """Infere usando un preset real de train y overrides validados."""
    started = time.perf_counter()
    if not isinstance(datos_entrada, dict):
        raise TypeError("datos_entrada debe ser un diccionario")
    preset_id = str(datos_entrada.get("preset_id", ""))
    if preset_id not in _PRESETS:
        raise ValueError(f"Preset desconocido: {preset_id}")
    row = dict(_PRESETS[preset_id])
    received_overrides = datos_entrada.get("overrides", {})
    if not isinstance(received_overrides, dict):
        raise TypeError("overrides debe ser un diccionario")
    overrides = dict(received_overrides)
    unexpected = set(overrides) - _ALLOWED_OVERRIDES
    if unexpected:
        raise ValueError(f"Overrides no permitidos: {sorted(unexpected)}")

    if "cell_voltage_avg" in overrides:
        try:
            value = float(overrides["cell_voltage_avg"])
        except (TypeError, ValueError):
            raise ValueError("cell_voltage_avg debe ser un número entre 3.06 y 3.85 V") from None
        if not 3.06 <= value <= 3.85:
            raise ValueError("cell_voltage_avg debe estar entre 3.06 y 3.85 V")
        overrides["cell_voltage_avg"] = value
    if "cell_temperature_max" in overrides:
        try:
            value = float(overrides["cell_temperature_max"])
        except (TypeError, ValueError):
            raise ValueError("cell_temperature_max debe ser un número entre -4.95 y 86.98 °C") from None
        if not -4.95 <= value <= 86.98:
            raise ValueError("cell_temperature_max debe estar entre -4.95 y 86.98 °C")
        overrides["cell_temperature_max"] = value
    if "terrain_type" in overrides:
        if overrides["terrain_type"] not in _TERRAIN:
            raise ValueError(f"terrain_type debe pertenecer a {sorted(_TERRAIN)}")

    row.update(overrides)
    features = _SCHEMA["features"]
    missing = [col for col in features if col not in row]
    if missing:
        raise ValueError("Faltan predictores: " + ", ".join(missing))
    frame = pd.DataFrame([{col: row[col] for col in features}], columns=features)
    class_scores = _MODEL.predict_proba(frame)[0]
    positive_index = list(_MODEL.classes_).index(1)
    risk_score = float(class_scores[positive_index])
    threshold = float(_SCHEMA["threshold"])
    prediction = int(risk_score >= threshold)
    recommendation = (
        "Priorizar revisión técnica de la batería; el resultado es apoyo a la decisión."
        if prediction == 1 else
        "Sin alerta del clasificador; mantener el plan de seguimiento preventivo."
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "prediccion": prediction,
        "etiqueta": "Riesgo de falla" if prediction == 1 else "Sin falla detectada",
        "score_riesgo": round(risk_score, 6),
        "nivel_riesgo": "alto" if prediction == 1 else "bajo",
        "recomendaciones": [recommendation],
        "rsl_meses": None,
        "tiempo_ms": round(elapsed_ms, 3),
        "modelo": _SCHEMA["selected_model"],
        "umbral_clasificacion": round(threshold, 6),
    }

if __name__ == "__main__":
    preset_id = "EV109069"
    if preset_id not in _PRESETS:
        raise ValueError(f"El preset de demostración no existe: {preset_id}")
    demo = {
        "preset_id": preset_id,
        "overrides": {
            "cell_voltage_avg": 3.649,
            "cell_temperature_max": 60.0,
            "terrain_type": "Flat",
        },
    }
    print(json.dumps({"entrada": demo, "salida": predecir(demo)}, ensure_ascii=False, indent=2))