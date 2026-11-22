from __future__ import annotations

import json
import time

import pandas as pd

from src.api.database import BmsThreshold, Vehicle
from src.models.predict import _MODEL, _SCHEMA


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _map_ui_chemistry_to_model(battery_type: str | None) -> str:
    if not battery_type:
        return "NMC"
    upper = battery_type.upper()
    if "LFP" in upper:
        return "LFP"
    if "NCA" in upper:
        return "NCA"
    return "NMC"


def build_default_preset_for_vehicle(vehicle: Vehicle) -> dict:
    pack_v = 760.0 if "800" in (vehicle.pack_architecture or "") else 360.0
    cycles = max(0, int(vehicle.cycle_count or 0))
    est_soh = round(_clamp(98.0 - cycles * 0.011, 55.0, 99.0), 1)
    est_res = round(_clamp(0.16 + cycles * 0.00018, 0.13, 0.52), 2)
    cap = float(vehicle.battery_capacity_kwh or 75.0)

    return {
        "vehicle_brand": vehicle.brand,
        "vehicle_model": vehicle.model,
        "vehicle_type": "SUV",
        "manufacturing_year": float(vehicle.year),
        "battery_manufacturer": "Desconocido",
        "battery_chemistry": _map_ui_chemistry_to_model(vehicle.battery_type),
        "battery_capacity_kwh": cap,
        "drive_type": "AWD",
        "fleet_or_private": "Private",
        "cycle_count": float(cycles),
        "battery_health_percent": est_soh,
        "state_of_charge": 72.0,
        "depth_of_discharge": 35.0,
        "state_of_health": est_soh,
        "cell_voltage_avg": 3.62,
        "cell_voltage_std": 0.008,
        "pack_voltage": pack_v,
        "cell_temperature_avg": 24.0,
        "cell_temperature_max": 32.0,
        "internal_resistance": est_res,
        "charge_efficiency": 96.0,
        "discharge_efficiency": 95.0,
        "remaining_capacity": round(cap * est_soh / 100.0, 2),
        "capacity_loss_percent": round(100.0 - est_soh, 2),
        "average_charge_power_kw": 45.0,
        "terrain_type": "Flat",
        "voltage_imbalance": 12.0,
        "temperature_variance": 8.0,
        "sensor_fault_count": 0.0,
        "BMS_warning_count": 0.0,
        "abnormal_voltage_events": 0.0,
        "previous_faults": 0.0,
    }


def parse_vehicle_preset(vehicle: Vehicle | None) -> dict:
    if vehicle is None:
        return {}
    if vehicle.preset_json:
        try:
            data = json.loads(vehicle.preset_json)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return build_default_preset_for_vehicle(vehicle)


def evaluate_bms_simulation(
    *,
    vehicle: Vehicle | None,
    battery_capacity_kwh: float,
    pack_voltage: float,
    cell_voltage_avg: float,
    state_of_charge: float,
    depth_of_discharge: float,
    cell_temperature_max: float,
    internal_resistance: float,
    cycle_count: int,
    thresholds: BmsThreshold,
) -> dict:
    """
    Capa externa de simulación BMS + inferencia en solo lectura con el modelo entrenado.
    No modifica ni reentrena modelo_candidato.joblib.
    """
    started = time.perf_counter()
    features = _SCHEMA["features"]
    base_preset = parse_vehicle_preset(vehicle)

    # 1. Estimación física auxiliar (algoritmo externo) de SoH, capacidad y eficiencia
    effective_capacity = (
        float(battery_capacity_kwh)
        if battery_capacity_kwh > 0
        else float(vehicle.battery_capacity_kwh if vehicle else base_preset.get("battery_capacity_kwh") or 75.0)
    )

    cycle_penalty = (cycle_count / 2000.0) * 20.0
    resistance_penalty = max(0.0, internal_resistance - 0.13) / (0.52 - 0.13) * 18.0
    thermal_penalty = max(0.0, cell_temperature_max - 40.0) * 0.35
    voltage_penalty = max(0.0, 3.30 - cell_voltage_avg) * 28.0 + max(0.0, cell_voltage_avg - 3.80) * 15.0
    dod_penalty = max(0.0, depth_of_discharge - 75.0) * 0.12

    state_of_health = round(
        _clamp(100.0 - cycle_penalty - resistance_penalty - thermal_penalty - voltage_penalty - dod_penalty, 35.0, 100.0),
        1,
    )
    capacity_loss_percent = round(max(0.0, 100.0 - state_of_health), 1)
    remaining_capacity = round(effective_capacity * (state_of_health / 100.0), 1)
    charge_efficiency = round(
        _clamp(
            98.4
            - max(0.0, internal_resistance - 0.13) * 18.5
            - max(0.0, cell_temperature_max - 45.0) * 0.18
            - max(0.0, 15.0 - cell_temperature_max) * 0.08,
            75.0,
            99.0,
        ),
        1,
    )

    # 2. Ensamblado de las 60 features en solo lectura para el clasificador entrenado
    row = {col: base_preset.get(col, None) for col in features}
    if vehicle is not None:
        row["vehicle_brand"] = row.get("vehicle_brand") or vehicle.brand
        row["vehicle_model"] = row.get("vehicle_model") or vehicle.model
        row["manufacturing_year"] = row.get("manufacturing_year") or float(vehicle.year)
        row["battery_chemistry"] = row.get("battery_chemistry") or _map_ui_chemistry_to_model(vehicle.battery_type)

    row["battery_capacity_kwh"] = effective_capacity
    row["pack_voltage"] = float(pack_voltage) if pack_voltage > 0 else row.get("pack_voltage")
    row["cell_voltage_avg"] = float(cell_voltage_avg)
    row["state_of_charge"] = float(state_of_charge)
    row["depth_of_discharge"] = float(depth_of_discharge)
    row["cell_temperature_max"] = float(cell_temperature_max)
    row["cell_temperature_avg"] = round(max(0.0, float(cell_temperature_max) - 8.5), 2)
    row["internal_resistance"] = float(internal_resistance)
    row["cycle_count"] = float(cycle_count)
    row["state_of_health"] = state_of_health
    row["battery_health_percent"] = state_of_health
    row["remaining_capacity"] = remaining_capacity
    row["capacity_loss_percent"] = capacity_loss_percent
    row["charge_efficiency"] = charge_efficiency

    frame = pd.DataFrame([{col: row[col] for col in features}], columns=features)
    class_scores = _MODEL.predict_proba(frame)[0]
    positive_index = list(_MODEL.classes_).index(1)
    score_riesgo = float(class_scores[positive_index])
    threshold = float(_SCHEMA["threshold"])
    prediccion = int(score_riesgo >= threshold)
    risk_percent = round(score_riesgo * 100.0, 1)

    # 3. Evaluación frente a los umbrales configurables del BMS + recomendaciones
    breaches: list[str] = []
    recommendations: list[str] = []

    if thresholds.cell_temperature_max > 0 and cell_temperature_max >= thresholds.cell_temperature_max:
        breaches.append(f"temperatura ({cell_temperature_max:.0f} °C >= {thresholds.cell_temperature_max:.0f} °C)")
        recommendations.append(
            f"Revisar el sistema de gestión térmica y refrigeración: la celda alcanzó {cell_temperature_max:.0f} °C (umbral BMS: {thresholds.cell_temperature_max:.0f} °C)."
        )
    elif cell_temperature_max >= 50.0:
        recommendations.append(
            "Limitar temporalmente la carga rápida DC para reducir estrés térmico en las celdas."
        )

    if thresholds.min_state_of_health > 0 and state_of_health < thresholds.min_state_of_health:
        breaches.append(f"SoH ({state_of_health:.1f}% < {thresholds.min_state_of_health:.0f}%)")
        recommendations.append(
            f"Programar diagnóstico de capacidad y balanceo de módulos: la salud estimada ({state_of_health:.1f}%) está por debajo del mínimo configurado ({thresholds.min_state_of_health:.0f}%)."
        )

    if thresholds.internal_resistance > 0 and internal_resistance >= thresholds.internal_resistance:
        breaches.append(f"resistencia interna ({internal_resistance:.2f} Ω >= {thresholds.internal_resistance:.2f} Ω)")
        recommendations.append(
            f"Inspeccionar envejecimiento de celdas y conectores de alto voltaje: resistencia interna elevada ({internal_resistance:.2f} Ω)."
        )

    if cell_voltage_avg <= 3.25:
        recommendations.append(
            "Voltaje promedio por celda en zona baja; evitar descargas profundas y verificar balanceo del BMS."
        )
    elif cell_voltage_avg >= 3.80:
        recommendations.append(
            "Voltaje promedio cercano al límite superior; validar calibración de corte de carga del BMS."
        )

    if depth_of_discharge >= 80.0:
        recommendations.append(
            "Reducir la profundidad de descarga (mantener SoC operativo entre 20% y 80%) para prolongar la vida útil."
        )

    if cycle_count >= 1400:
        recommendations.append(
            f"El paquete acumula {cycle_count} ciclos; programar inspección preventiva de degradación."
        )

    alert = bool(prediccion == 1 or len(breaches) > 0)

    if prediccion == 1:
        status_title = "Riesgo de falla detectado"
        status_message = (
            f"La IA clasificó el escenario con riesgo alto ({risk_percent:.1f}%, umbral {threshold * 100:.1f}%)."
        )
        if breaches:
            status_message += " Umbrales BMS superados: " + ", ".join(breaches) + "."
        recommendations.insert(
            0,
            "Priorizar revisión técnica del paquete de batería; el diagnóstico de IA es apoyo a la decisión.",
        )
    elif breaches:
        status_title = "Alerta de umbral BMS"
        status_message = (
            f"El clasificador IA marca riesgo controlado ({risk_percent:.1f}%), pero se superaron umbrales del taller: "
            + ", ".join(breaches)
            + "."
        )
    else:
        status_title = "Operación estable"
        status_message = (
            f"Sin alerta de falla del clasificador IA (riesgo {risk_percent:.1f}%) y parámetros dentro de umbrales BMS."
        )
        if not recommendations:
            recommendations.append(
                "Sin alerta del clasificador; mantener el plan de seguimiento preventivo y ciclos de carga moderados."
            )

    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)

    return {
        "state_of_health": state_of_health,
        "remaining_capacity": remaining_capacity,
        "charge_efficiency": charge_efficiency,
        "capacity_loss_percent": capacity_loss_percent,
        "risk": risk_percent,
        "alert": alert,
        "statusTitle": status_title,
        "statusMessage": status_message,
        "recommendations": recommendations,
        "prediccion": prediccion,
        "score_riesgo": round(score_riesgo, 6),
        "modelo": _SCHEMA["selected_model"],
        "umbral_clasificacion": round(threshold, 6),
        "tiempo_ms": elapsed_ms,
    }
