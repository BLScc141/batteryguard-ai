from __future__ import annotations

import json
import math
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session

from src.api.auth import (
    TEMP_RESET_PASSWORD,
    create_access_token,
    get_current_user,
    hash_password,
    require_admin,
    require_password_updated,
    serialize_user,
    verify_password,
)
from src.api.database import BmsThreshold, Evaluation, User, Vehicle, get_db, init_db
from src.api.inference import (
    build_default_preset_for_vehicle,
    evaluate_bms_simulation,
    parse_vehicle_preset,
)
from src.models.predict import _PRESETS, _SCHEMA, predecir

TerrainType = Literal[
    "Flat",
    "Coastal",
    "Mountainous",
    "Hilly",
    "Desert",
    "Desconocido",
]

RoleType = Literal["Administrador", "Empleado"]
WorkshopStatusType = Literal["En ingreso", "En diagnóstico", "En carga", "Listo"]

DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def _get_cors_origins() -> list[str]:
    env_origins = os.getenv("CORS_ORIGINS", "").strip()
    if not env_origins:
        return DEFAULT_CORS_ORIGINS
    return [origin.strip() for origin in env_origins.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db(hash_password, _PRESETS)
    yield


# Ensure DB is also initialized when imported directly (e.g. tests)
init_db(hash_password, _PRESETS)


# ---------- Schemas contrato original README (/predict) ----------


class OverridesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cell_voltage_avg: float | None = Field(
        default=None,
        ge=3.06,
        le=3.85,
        description="Voltaje promedio de celda en voltios (3.06 a 3.85 V).",
    )
    cell_temperature_max: float | None = Field(
        default=None,
        ge=-4.95,
        le=86.98,
        description="Temperatura máxima de celda en °C (-4.95 a 86.98 °C).",
    )
    terrain_type: TerrainType | None = Field(
        default=None,
        description="Tipo de terreno.",
    )

    @field_validator("cell_voltage_avg", "cell_temperature_max", mode="before")
    @classmethod
    def _reject_bool_and_null_number(cls, value):
        if value is None:
            raise ValueError("El valor no puede ser null; omite la clave si no deseas sobrescribirla.")
        if isinstance(value, bool):
            raise ValueError("Debe ser un valor numérico, no booleano.")
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError("Debe ser un número finito.")
        return value

    @field_validator("terrain_type", mode="before")
    @classmethod
    def _reject_null_terrain(cls, value):
        if value is None:
            raise ValueError("terrain_type no puede ser null; omite la clave si no deseas sobrescribirla.")
        return value


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "preset_id": "EV109069",
                "overrides": {
                    "cell_voltage_avg": 3.649,
                    "cell_temperature_max": 60.0,
                    "terrain_type": "Flat",
                },
            }
        },
    )

    preset_id: str = Field(..., description="Identificador de preset existente en models/presets.json.")
    overrides: OverridesInput | None = Field(
        default=None,
        description="Overrides opcionales permitidos para la inferencia.",
    )

    @field_validator("preset_id")
    @classmethod
    def _validate_preset_id(cls, value: str) -> str:
        if value not in _PRESETS:
            raise ValueError(f"Preset desconocido: {value}")
        return value


class PredictResponse(BaseModel):
    prediccion: int
    etiqueta: str
    score_riesgo: float
    nivel_riesgo: Literal["bajo", "alto"]
    recomendaciones: list[str]
    rsl_meses: float | None
    tiempo_ms: float
    modelo: str
    umbral_clasificacion: float


# ---------- Schemas del sistema web (Auth, Empleados, Vehículos, Simulación, Umbrales) ----------


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)


class ChangePasswordRequest(BaseModel):
    currentPassword: str | None = None
    newPassword: str = Field(..., min_length=6)


class CreateEmployeeRequest(BaseModel):
    fullName: str = Field(..., min_length=2, max_length=160)
    email: str = Field(..., min_length=4, max_length=255)
    role: RoleType = "Empleado"
    password: str = Field(..., min_length=6, max_length=128)


class CreateVehicleRequest(BaseModel):
    plate: str = Field(..., min_length=2, max_length=64)
    brand: str = Field(..., min_length=1, max_length=80)
    model: str = Field(..., min_length=1, max_length=120)
    year: int = Field(..., ge=2005, le=2035)
    batteryType: str = Field(..., min_length=1, max_length=64)
    packArchitecture: str = Field(..., min_length=1, max_length=32)
    battery_capacity_kwh: float = Field(..., gt=0)
    cycle_count: int = Field(default=0, ge=0)
    status: WorkshopStatusType = Field(default="En ingreso")


class UpdateVehicleStatusRequest(BaseModel):
    status: WorkshopStatusType


class SimulationRequest(BaseModel):
    vehicleId: int | None = None
    battery_capacity_kwh: float = Field(default=0.0, ge=0)
    pack_voltage: float = Field(..., ge=0, le=1000)
    cell_voltage_avg: float = Field(..., ge=3.06, le=3.85)
    state_of_charge: float = Field(..., ge=0, le=100)
    depth_of_discharge: float = Field(..., ge=0, le=100)
    cell_temperature_max: float = Field(..., ge=-4.95, le=86.98)
    internal_resistance: float = Field(..., ge=0.05, le=1.5)
    cycle_count: int = Field(..., ge=0, le=10000)


class ThresholdsPayload(BaseModel):
    cell_temperature_max: float = Field(..., ge=0, le=120)
    min_state_of_health: float = Field(..., ge=0, le=100)
    voltage_imbalance: float = Field(..., ge=0, le=500)
    internal_resistance: float = Field(..., ge=0, le=2.0)
    BMS_warning_count: int = Field(..., ge=0, le=100)
    refreshInterval: int = Field(..., ge=0, le=3600)


def _serialize_vehicle(vehicle: Vehicle) -> dict:
    return {
        "id": vehicle.id,
        "plate": vehicle.plate,
        "brand": vehicle.brand,
        "model": vehicle.model,
        "year": vehicle.year,
        "batteryType": vehicle.battery_type,
        "packArchitecture": vehicle.pack_architecture,
        "battery_capacity_kwh": vehicle.battery_capacity_kwh,
        "cycle_count": vehicle.cycle_count,
        "status": vehicle.status,
        "presetId": vehicle.preset_id,
        "preset": parse_vehicle_preset(vehicle),
    }


def _serialize_evaluation(item: Evaluation) -> dict:
    created = item.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    local_time = created.astimezone().strftime("%H:%M")
    try:
        recs = json.loads(item.recommendations_json)
    except json.JSONDecodeError:
        recs = []

    return {
        "id": item.id,
        "date": local_time,
        "createdAt": created.isoformat(),
        "vehicleId": item.vehicle_id,
        "vehicleLabel": item.vehicle_label,
        "battery_capacity_kwh": item.battery_capacity_kwh,
        "pack_voltage": item.pack_voltage,
        "cell_voltage_avg": item.cell_voltage_avg,
        "state_of_charge": item.state_of_charge,
        "depth_of_discharge": item.depth_of_discharge,
        "cell_temperature_max": item.cell_temperature_max,
        "internal_resistance": item.internal_resistance,
        "cycle_count": item.cycle_count,
        "state_of_health": item.state_of_health,
        "remaining_capacity": item.remaining_capacity,
        "charge_efficiency": item.charge_efficiency,
        "capacity_loss_percent": item.capacity_loss_percent,
        "risk": item.risk,
        "alert": bool(item.alert),
        "status": item.status_title,
        "statusTitle": item.status_title,
        "statusMessage": item.status_message,
        "recommendations": recs,
    }


def _serialize_thresholds(row: BmsThreshold) -> dict:
    return {
        "cell_temperature_max": row.cell_temperature_max,
        "min_state_of_health": row.min_state_of_health,
        "voltage_imbalance": row.voltage_imbalance,
        "internal_resistance": row.internal_resistance,
        "BMS_warning_count": row.bms_warning_count,
        "refreshInterval": row.refresh_interval,
    }


def _get_or_create_thresholds(db: Session) -> BmsThreshold:
    row = db.query(BmsThreshold).filter(BmsThreshold.id == 1).first()
    if row is None:
        row = BmsThreshold(id=1)
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


app = FastAPI(
    title="BatteryGuard AI API",
    description="API de inferencia IA, gestión de vehículos EV, roles y umbrales BMS.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Health y contrato original /predict ----------


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "modelo": _SCHEMA["selected_model"],
        "umbral_clasificacion": round(float(_SCHEMA["threshold"]), 6),
        "presets_disponibles": len(_PRESETS),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict:
    payload: dict = {"preset_id": request.preset_id}
    if request.overrides is not None:
        payload["overrides"] = request.overrides.model_dump(exclude_unset=True)
    try:
        return predecir(payload)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


# ---------- Autenticación y contraseñas ----------


@app.post("/auth/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> dict:
    normalized_email = payload.email.strip().lower()
    user = db.query(User).filter(User.email == normalized_email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Correo o contraseña incorrectos.",
        )

    token = create_access_token(user)
    return {
        "token": token,
        "user": serialize_user(user),
    }


@app.get("/auth/me")
def get_me(current_user: User = Depends(get_current_user)) -> dict:
    return {"user": serialize_user(current_user)}


@app.post("/auth/change-password")
def change_password(
    payload: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    if current_user.must_change_password:
        if payload.newPassword == TEMP_RESET_PASSWORD:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La nueva contraseña debe ser diferente a la contraseña temporal.",
            )
    else:
        if not payload.currentPassword:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Debes ingresar tu contraseña actual.",
            )
        if not verify_password(payload.currentPassword, current_user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La contraseña actual no es correcta.",
            )
        if payload.newPassword == payload.currentPassword:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La nueva contraseña debe ser diferente a la actual.",
            )

    current_user.password_hash = hash_password(payload.newPassword)
    current_user.must_change_password = False
    current_user.status = "Activo"
    db.commit()
    db.refresh(current_user)

    return {
        "message": "Contraseña actualizada correctamente.",
        "user": serialize_user(current_user),
    }


# ---------- Empleados (solo Administrador) ----------


@app.get("/employees")
def list_employees(
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> list[dict]:
    users = db.query(User).order_by(User.created_at.desc(), User.id.desc()).all()
    return [serialize_user(u) for u in users]


@app.post("/employees", status_code=status.HTTP_201_CREATED)
def create_employee(
    payload: CreateEmployeeRequest,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict:
    normalized_email = payload.email.strip().lower()
    if "@" not in normalized_email:
        raise HTTPException(status_code=422, detail="Correo electrónico inválido.")

    existing = db.query(User).filter(User.email == normalized_email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ya existe un usuario registrado con ese correo.",
        )

    user = User(
        full_name=payload.fullName.strip(),
        email=normalized_email,
        role=payload.role,
        password_hash=hash_password(payload.password),
        status="Activo",
        must_change_password=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return serialize_user(user)


@app.post("/employees/{employee_id}/reset-password")
def reset_employee_password(
    employee_id: int,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict:
    user = db.query(User).filter(User.id == employee_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Empleado no encontrado.")

    user.password_hash = hash_password(TEMP_RESET_PASSWORD)
    user.must_change_password = True
    db.commit()
    db.refresh(user)
    return {
        "message": "Contraseña restablecida correctamente.",
        "temporaryPassword": TEMP_RESET_PASSWORD,
        "employee": serialize_user(user),
    }


# ---------- Vehículos y Presets ----------


@app.get("/presets")
def list_presets(db: Session = Depends(get_db)) -> dict:
    vehicles = db.query(Vehicle).order_by(Vehicle.id.asc()).all()
    return {
        "presets": [_serialize_vehicle(v) for v in vehicles],
    }


@app.get("/vehicles")
def list_vehicles(
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict]:
    vehicles = db.query(Vehicle).order_by(Vehicle.id.desc()).all()
    return [_serialize_vehicle(v) for v in vehicles]


@app.post("/vehicles", status_code=status.HTTP_201_CREATED)
def create_vehicle(
    payload: CreateVehicleRequest,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict:
    plate = payload.plate.strip().upper()
    existing = db.query(Vehicle).filter(Vehicle.plate == plate).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ya existe un vehículo registrado con esa placa. Puedes repetir marca y modelo, pero la placa debe ser única.",
        )

    vehicle = Vehicle(
        plate=plate,
        brand=payload.brand.strip(),
        model=payload.model.strip(),
        year=payload.year,
        battery_type=payload.batteryType.strip(),
        pack_architecture=payload.packArchitecture.strip(),
        battery_capacity_kwh=round(float(payload.battery_capacity_kwh), 2),
        cycle_count=int(payload.cycle_count),
        status=payload.status,
    )
    vehicle.preset_json = json.dumps(build_default_preset_for_vehicle(vehicle), ensure_ascii=False)
    db.add(vehicle)
    db.commit()
    db.refresh(vehicle)
    return _serialize_vehicle(vehicle)


@app.patch("/vehicles/{vehicle_id}/status")
def update_vehicle_status(
    vehicle_id: int,
    payload: UpdateVehicleStatusRequest,
    _user: User = Depends(require_password_updated),
    db: Session = Depends(get_db),
) -> dict:
    vehicle = db.query(Vehicle).filter(Vehicle.id == vehicle_id).first()
    if vehicle is None:
        raise HTTPException(status_code=404, detail="Vehículo no encontrado.")

    vehicle.status = payload.status
    db.commit()
    db.refresh(vehicle)
    return _serialize_vehicle(vehicle)


# ---------- Simulación IA y Evaluaciones ----------


@app.post("/simulate")
def simulate_battery(
    payload: SimulationRequest,
    current_user: User = Depends(require_password_updated),
    db: Session = Depends(get_db),
) -> dict:
    vehicle = None
    if payload.vehicleId is not None:
        vehicle = db.query(Vehicle).filter(Vehicle.id == payload.vehicleId).first()
        if vehicle is None:
            raise HTTPException(status_code=404, detail="Vehículo no encontrado.")

    thresholds = _get_or_create_thresholds(db)
    result = evaluate_bms_simulation(
        vehicle=vehicle,
        battery_capacity_kwh=payload.battery_capacity_kwh,
        pack_voltage=payload.pack_voltage,
        cell_voltage_avg=payload.cell_voltage_avg,
        state_of_charge=payload.state_of_charge,
        depth_of_discharge=payload.depth_of_discharge,
        cell_temperature_max=payload.cell_temperature_max,
        internal_resistance=payload.internal_resistance,
        cycle_count=payload.cycle_count,
        thresholds=thresholds,
    )

    vehicle_label = (
        f"{vehicle.brand} {vehicle.model} ({vehicle.plate})"
        if vehicle is not None
        else "Sin vehículo seleccionado"
    )

    evaluation = Evaluation(
        vehicle_id=vehicle.id if vehicle else None,
        vehicle_label=vehicle_label,
        user_id=current_user.id,
        battery_capacity_kwh=(
            payload.battery_capacity_kwh
            if payload.battery_capacity_kwh > 0
            else (vehicle.battery_capacity_kwh if vehicle else 0.0)
        ),
        pack_voltage=payload.pack_voltage,
        cell_voltage_avg=payload.cell_voltage_avg,
        state_of_charge=payload.state_of_charge,
        depth_of_discharge=payload.depth_of_discharge,
        cell_temperature_max=payload.cell_temperature_max,
        internal_resistance=payload.internal_resistance,
        cycle_count=payload.cycle_count,
        state_of_health=result["state_of_health"],
        remaining_capacity=result["remaining_capacity"],
        charge_efficiency=result["charge_efficiency"],
        capacity_loss_percent=result["capacity_loss_percent"],
        risk=result["risk"],
        prediccion=result["prediccion"],
        score_riesgo=result["score_riesgo"],
        alert=result["alert"],
        status_title=result["statusTitle"],
        status_message=result["statusMessage"],
        recommendations_json=json.dumps(result["recommendations"], ensure_ascii=False),
    )
    db.add(evaluation)
    if vehicle is not None:
        vehicle.status = "En diagnóstico" if result["alert"] else "Listo"
        vehicle.cycle_count = int(payload.cycle_count)
    db.commit()
    db.refresh(evaluation)
    if vehicle is not None:
        db.refresh(vehicle)

    return {
        **result,
        "evaluation": _serialize_evaluation(evaluation),
        "vehicle": _serialize_vehicle(vehicle) if vehicle is not None else None,
    }


@app.get("/evaluations")
def list_evaluations(
    vehicleId: int | None = Query(default=None),
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict]:
    query = db.query(Evaluation)
    if vehicleId is not None:
        query = query.filter(Evaluation.vehicle_id == vehicleId)
    rows = query.order_by(Evaluation.created_at.desc(), Evaluation.id.desc()).limit(200).all()
    return [_serialize_evaluation(item) for item in rows]


# ---------- Umbrales BMS (lectura para todos, escritura solo Administrador) ----------


@app.get("/settings/thresholds")
def get_thresholds(
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    return _serialize_thresholds(_get_or_create_thresholds(db))


@app.put("/settings/thresholds")
def update_thresholds(
    payload: ThresholdsPayload,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict:
    row = _get_or_create_thresholds(db)
    row.cell_temperature_max = float(payload.cell_temperature_max)
    row.min_state_of_health = float(payload.min_state_of_health)
    row.voltage_imbalance = float(payload.voltage_imbalance)
    row.internal_resistance = float(payload.internal_resistance)
    row.bms_warning_count = int(payload.BMS_warning_count)
    row.refresh_interval = int(payload.refreshInterval)
    row.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(row)
    return _serialize_thresholds(row)
