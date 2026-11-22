from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQLITE_PATH = ROOT / "batteryguard.db"


def _resolve_database_url() -> str:
    raw = os.getenv("DATABASE_URL", "").strip()
    if not raw:
        return f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"
    if raw.startswith("postgres://"):
        return "postgresql+psycopg://" + raw[len("postgres://") :]
    if raw.startswith("postgresql://") and "+psycopg" not in raw:
        return "postgresql+psycopg://" + raw[len("postgresql://") :]
    return raw


DATABASE_URL = _resolve_database_url()
IS_SQLITE = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if IS_SQLITE else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    full_name: Mapped[str] = mapped_column(String(160), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="Empleado")
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="Activo")
    must_change_password: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class Vehicle(Base):
    __tablename__ = "vehicles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plate: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    brand: Mapped[str] = mapped_column(String(80), nullable=False)
    model: Mapped[str] = mapped_column(String(120), nullable=False)
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    battery_type: Mapped[str] = mapped_column(String(64), nullable=False)
    pack_architecture: Mapped[str] = mapped_column(String(32), nullable=False)
    battery_capacity_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    cycle_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(48), nullable=False, default="En ingreso")
    preset_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    preset_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    vehicle_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("vehicles.id", ondelete="SET NULL"), nullable=True, index=True
    )
    vehicle_label: Mapped[str] = mapped_column(String(200), nullable=False)
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    battery_capacity_kwh: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pack_voltage: Mapped[float] = mapped_column(Float, nullable=False)
    cell_voltage_avg: Mapped[float] = mapped_column(Float, nullable=False)
    state_of_charge: Mapped[float] = mapped_column(Float, nullable=False)
    depth_of_discharge: Mapped[float] = mapped_column(Float, nullable=False)
    cell_temperature_max: Mapped[float] = mapped_column(Float, nullable=False)
    internal_resistance: Mapped[float] = mapped_column(Float, nullable=False)
    cycle_count: Mapped[int] = mapped_column(Integer, nullable=False)
    state_of_health: Mapped[float] = mapped_column(Float, nullable=False)
    remaining_capacity: Mapped[float] = mapped_column(Float, nullable=False)
    charge_efficiency: Mapped[float] = mapped_column(Float, nullable=False)
    capacity_loss_percent: Mapped[float] = mapped_column(Float, nullable=False)
    risk: Mapped[float] = mapped_column(Float, nullable=False)
    prediccion: Mapped[int] = mapped_column(Integer, nullable=False)
    score_riesgo: Mapped[float] = mapped_column(Float, nullable=False)
    alert: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status_title: Mapped[str] = mapped_column(String(120), nullable=False)
    status_message: Mapped[str] = mapped_column(Text, nullable=False)
    recommendations_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class BmsThreshold(Base):
    __tablename__ = "bms_thresholds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cell_temperature_max: Mapped[float] = mapped_column(Float, nullable=False, default=55.0)
    min_state_of_health: Mapped[float] = mapped_column(Float, nullable=False, default=80.0)
    voltage_imbalance: Mapped[float] = mapped_column(Float, nullable=False, default=30.0)
    internal_resistance: Mapped[float] = mapped_column(Float, nullable=False, default=0.45)
    bms_warning_count: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    refresh_interval: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _map_chemistry_to_ui(chemistry: str | None) -> str:
    if not chemistry:
        return "Li-ion (NMC)"
    upper = str(chemistry).upper()
    if upper == "LFP":
        return "LFP (LiFePO4)"
    if upper == "NCA":
        return "NCA"
    if upper == "NMC":
        return "Li-ion (NMC)"
    return "Li-ion (NMC)"


def _safe_float(value, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int) -> int:
    try:
        if value is None:
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def seed_initial_data(db: Session, hash_password_fn, presets: dict) -> None:
    admin_email = os.getenv("ADMIN_EMAIL", "admin@batteryguard.ai").strip().lower()
    admin_password = os.getenv("ADMIN_PASSWORD", "admin1234")

    existing_admin = db.query(User).filter(User.email == admin_email).first()
    if not existing_admin:
        db.add(
            User(
                full_name="Administrador BatteryGuard",
                email=admin_email,
                role="Administrador",
                password_hash=hash_password_fn(admin_password),
                status="Activo",
                must_change_password=False,
            )
        )

    existing_thresholds = db.query(BmsThreshold).filter(BmsThreshold.id == 1).first()
    if not existing_thresholds:
        db.add(
            BmsThreshold(
                id=1,
                cell_temperature_max=55.0,
                min_state_of_health=80.0,
                voltage_imbalance=30.0,
                internal_resistance=0.45,
                bms_warning_count=2,
                refresh_interval=30,
            )
        )

    if db.query(Vehicle).count() == 0:
        statuses = ["En diagnóstico", "Listo", "En carga", "En ingreso"]
        for idx, (preset_id, raw) in enumerate(presets.items()):
            brand = raw.get("vehicle_brand") or "EV"
            model = raw.get("vehicle_model") or "Modelo EV"
            if model == "Desconocido":
                model = "SUV Eléctrico"
            year = _safe_int(raw.get("manufacturing_year"), 2022)
            pack_voltage = _safe_float(raw.get("pack_voltage"), 350.0)
            architecture = "800 V" if pack_voltage >= 600 else "400 V"
            capacity = round(_safe_float(raw.get("battery_capacity_kwh"), 75.0), 2)
            cycles = _safe_int(raw.get("cycle_count"), 0)
            generated_plate = f"EV-{101 + idx:03d}-MX"

            db.add(
                Vehicle(
                    plate=generated_plate,
                    brand=str(brand),
                    model=str(model),
                    year=year,
                    battery_type=_map_chemistry_to_ui(raw.get("battery_chemistry")),
                    pack_architecture=architecture,
                    battery_capacity_kwh=capacity,
                    cycle_count=cycles,
                    status=statuses[idx % len(statuses)],
                    preset_id=preset_id,
                    preset_json=json.dumps(raw, ensure_ascii=False),
                )
            )
    else:
        # Migrar placas antiguas que tenían el preset_id (ej. EV109069) a formato placa EV-XXX-MX
        existing_plates = {
            row[0].upper() for row in db.query(Vehicle.plate).all() if row[0]
        }
        seeded_rows = (
            db.query(Vehicle)
            .filter(Vehicle.preset_id.isnot(None))
            .order_by(Vehicle.id.asc())
            .all()
        )
        next_num = 101
        for vehicle in seeded_rows:
            if vehicle.plate == vehicle.preset_id:
                while f"EV-{next_num:03d}-MX" in existing_plates:
                    next_num += 1
                new_plate = f"EV-{next_num:03d}-MX"
                existing_plates.add(new_plate)
                old_plate = vehicle.plate
                vehicle.plate = new_plate
                # Actualizar etiqueta en evaluaciones previas si existían
                prev_evals = (
                    db.query(Evaluation)
                    .filter(Evaluation.vehicle_id == vehicle.id)
                    .all()
                )
                for ev in prev_evals:
                    if old_plate in ev.vehicle_label:
                        ev.vehicle_label = ev.vehicle_label.replace(old_plate, new_plate)
                next_num += 1

    db.commit()


def init_db(hash_password_fn, presets: dict) -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        seed_initial_data(db, hash_password_fn, presets)
