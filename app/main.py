"""
Employee Attendance & Analytics API

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB  (a local .env is loaded for convenience; real environment variables win)

All application code lives in this one file. Sections:
  1. Configuration and database
  2. Time and rounding helpers   (rules R1 - R5, R8)
  3. Serialisation               (BSON datetime <-> epoch milliseconds)
  4. Request models              (input validation)
  5. Startup: indexes
  6. Endpoints
"""
import logging
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Any, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from pydantic import AfterValidator, BaseModel, Field, model_validator
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import DuplicateKeyError, OperationFailure, PyMongoError

log = logging.getLogger("attendance")

# --------------------------------------------------------------------------- #
# 1. Configuration and database
# --------------------------------------------------------------------------- #
load_dotenv(override=False)  # a local .env never overrides real environment variables

MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB = os.environ.get("MONGO_DB")
if not MONGO_URI or not MONGO_DB:
    raise RuntimeError("MONGO_URI and MONGO_DB must be set (environment variables or .env)")

# tz_aware=True: PyMongo returns timezone-aware UTC datetimes instead of naive ones.
client = MongoClient(MONGO_URI, tz_aware=True, serverSelectionTimeoutMS=5000)
db = client[MONGO_DB]

# --------------------------------------------------------------------------- #
# 2. Time and rounding helpers
# --------------------------------------------------------------------------- #
UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

MS_MIN = 100_000_000_000      # below this the value looks like seconds -> 422
MS_MAX = 4_102_444_800_000    # 2100-01-01

PRESENCE_STATUSES = ("PRESENT", "WFH", "ON_DUTY")
ALL_STATUSES = ("PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY")

GRACE_SECONDS = 10 * 60        # R2
MIN_OVERTIME_MINUTES = 30      # R3
HALF_DAY_BELOW_HOURS = 4.5     # R5


def now_utc() -> datetime:
    """Current instant, truncated to whole seconds (R1)."""
    return datetime.now(UTC).replace(microsecond=0)


def ms_to_dt(ms: int) -> datetime:
    """Epoch milliseconds -> aware UTC datetime, truncated to whole seconds (R1)."""
    return EPOCH + timedelta(seconds=ms // 1000)


def dt_to_ms(dt: Optional[datetime]) -> Optional[int]:
    """Datetime -> epoch milliseconds, using integer arithmetic only (no float rounding).
    A naive datetime is treated as UTC, which is what BSON dates always are."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = dt - EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1000 + delta.microseconds // 1000


def parse_hhmm(value: str) -> time:
    hours, minutes = value.split(":")
    return time(int(hours), int(minutes))


def is_overnight(emp: dict) -> bool:
    """R1: shift_end <= shift_start means the shift crosses midnight. HH:MM strings compare correctly."""
    return emp["shift_end"] <= emp["shift_start"]


def ist_at(day: date, hhmm: str) -> datetime:
    """The instant of clock time hhmm (IST) on calendar day `day`."""
    return datetime.combine(day, parse_hhmm(hhmm), tzinfo=IST)


def attendance_date(punch_in: datetime, emp: dict) -> date:
    """R1: the IST calendar date of the punch-in; for an overnight shift, a punch-in earlier than
    shift_end belongs to the shift that started the previous day."""
    local = punch_in.astimezone(IST)
    day = local.date()
    if is_overnight(emp) and local.time() < parse_hhmm(emp["shift_end"]):
        day -= timedelta(days=1)
    return day


def shift_start_at(day: date, emp: dict) -> datetime:
    return ist_at(day, emp["shift_start"])


def shift_end_at(day: date, emp: dict) -> datetime:
    """R1: an overnight shift ends on the next calendar day."""
    end_day = day + timedelta(days=1) if is_overnight(emp) else day
    return ist_at(end_day, emp["shift_end"])


def compute_late_minutes(punch_in: datetime, day: date, emp: dict) -> int:
    """R2: late only if strictly more than 10:00 after shift start; then whole minutes since shift start."""
    seconds = int((punch_in - shift_start_at(day, emp)).total_seconds())
    return seconds // 60 if seconds > GRACE_SECONDS else 0


def compute_overtime(punch_out: datetime, day: date, emp: dict) -> int:
    """R3: whole minutes after shift end, counted only when that is >= 30."""
    minutes = int((punch_out - shift_end_at(day, emp)).total_seconds()) // 60
    return minutes if minutes >= MIN_OVERTIME_MINUTES else 0


def round_half_up(value: Optional[float], places: int = 2) -> Optional[float]:
    """R4/R8: decimal half-up rounding (Python's round() is banker's rounding on binary floats)."""
    if value is None:
        return None
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: (out - in) seconds / 3600, rounded to 2 decimals half-up. Exact Decimal maths."""
    seconds = int((punch_out - punch_in).total_seconds())
    hours = (Decimal(seconds) / Decimal(3600)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(hours)


def derive_fields(status: str, punch_in: Optional[datetime], punch_out: Optional[datetime],
                  day: date, emp: dict) -> dict:
    """The four derived fields for a record, per the table in DATA_MODEL.md section 3."""
    if status not in PRESENCE_STATUSES or punch_in is None:
        return {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    fields = {
        "work_hours": None,
        "late_minutes": compute_late_minutes(punch_in, day, emp),
        "overtime_minutes": 0,
        "half_day": False,
    }
    if punch_out is not None:
        hours = compute_work_hours(punch_in, punch_out)
        fields["work_hours"] = hours
        fields["overtime_minutes"] = compute_overtime(punch_out, day, emp)
        fields["half_day"] = hours < HALF_DAY_BELOW_HOURS  # R5 uses the rounded value
    return fields


# --------------------------------------------------------------------------- #
# 3. Serialisation: what leaves the API
# --------------------------------------------------------------------------- #
def _ms_or_value(value: Any) -> Any:
    return dt_to_ms(value) if isinstance(value, datetime) else value


def serialize_employee(doc: dict) -> dict:
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": dt_to_ms(doc.get("created_at")),
    }


def serialize_history(entries: Optional[list]) -> list:
    out = []
    for entry in entries or []:
        changes = {
            field: {"from": _ms_or_value(change.get("from")), "to": _ms_or_value(change.get("to"))}
            for field, change in (entry.get("changes") or {}).items()
        }
        out.append({
            "at": dt_to_ms(entry.get("at")),
            "by": entry.get("by"),
            "reason": entry.get("reason"),
            "changes": changes,
        })
    return out


def serialize_record(doc: dict) -> dict:
    """Missing legacy fields get their documented defaults (DATA_MODEL.md section 3)."""
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": dt_to_ms(doc.get("punch_in")),
        "punch_out": dt_to_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes") or 0,
        "overtime_minutes": doc.get("overtime_minutes") or 0,
        "half_day": bool(doc.get("half_day", False)),
        "history": serialize_history(doc.get("history")),
    }


# --------------------------------------------------------------------------- #
# 4. Request models and reusable validated types
# --------------------------------------------------------------------------- #
DATE_PATTERN = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"   # [0-9], not \d: \d also matches non-ASCII digits
HHMM_PATTERN = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"


def _real_date(value: str) -> str:
    date.fromisoformat(value)  # ValueError (-> 422) for e.g. 2026-02-30
    return value


IsoDate = Annotated[str, Field(pattern=DATE_PATTERN), AfterValidator(_real_date)]
HHMM = Annotated[str, Field(pattern=HHMM_PATTERN)]
# strict=True: rejects strings ("1783312500000"), floats (1.7e12) and booleans with 422
EpochMs = Annotated[int, Field(strict=True, ge=MS_MIN, le=MS_MAX)]
Status = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
PresenceStatus = Literal["PRESENT", "WFH", "ON_DUTY"]
Page = Annotated[int, Query(ge=1)]
PageSize = Annotated[int, Query(ge=1, le=100)]


class EmployeeCreate(BaseModel):
    emp_code: Annotated[str, Field(pattern=r"^EMP[0-9]{4,6}$")]
    name: Annotated[str, Field(min_length=1, max_length=100)]
    email: Annotated[str, Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")]
    department: Annotated[str, Field(min_length=1, max_length=50)]
    shift_start: HHMM = "09:30"
    shift_end: HHMM = "18:30"
    joined_on: IsoDate

    @model_validator(mode="after")
    def _shift_not_empty(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInRequest(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None
    status: PresenceStatus = "PRESENT"


def validation_error(loc: tuple, msg: str, value: Any = None) -> RequestValidationError:
    """A 422 in FastAPI's default shape, for checks that involve more than one parameter."""
    return RequestValidationError([{"type": "value_error", "loc": loc, "msg": msg, "input": value}])


# --------------------------------------------------------------------------- #
# 5. Startup: indexes (idempotent - create_index is a no-op when the index already exists)
# --------------------------------------------------------------------------- #
INDEXES = {
    "employees": [
        # Uniqueness of the business key; makes concurrent duplicate creates race-safe (one 201, rest 409).
        ([("emp_code", ASCENDING)], {"unique": True, "name": "uniq_emp_code"}),
        # GET /employees?department=...: equality on department, sorted by emp_code.
        ([("department", ASCENDING), ("emp_code", ASCENDING)], {"name": "department_emp_code"}),
    ],
    "attendance_logs": [
        # One record per (emp_code, date): makes concurrent punch-ins race-safe and serves every
        # per-employee lookup (punch-out, PATCH, an employee's date range).
        ([("emp_code", ASCENDING), ("date", ASCENDING)], {"unique": True, "name": "uniq_emp_date"}),
        # GET /attendance without emp_code: date range + sort (date desc, emp_code asc) with no in-memory sort.
        ([("date", DESCENDING), ("emp_code", ASCENDING)], {"name": "date_desc_emp_code"}),
        # GET /attendance?status=...: equality on status, then the same sort.
        ([("status", ASCENDING), ("date", DESCENDING), ("emp_code", ASCENDING)], {"name": "status_date_emp_code"}),
    ],
}


def ensure_indexes() -> None:
    for collection, specs in INDEXES.items():
        for keys, options in specs:
            try:
                db[collection].create_index(keys, **options)
            except OperationFailure as exc:  # e.g. an index with the same keys but other options exists
                log.error("could not create index %s on %s: %s", options.get("name"), collection, exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_indexes()
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# 6. Endpoints
# --------------------------------------------------------------------------- #
def get_employee_or_404(emp_code: str) -> dict:
    emp = db.employees.find_one({"emp_code": emp_code}, {"_id": 0})
    if emp is None:
        raise HTTPException(404, "employee not found")
    return emp


@app.get("/health")
def health():
    try:
        client.admin.command("ping")
    except PyMongoError:
        raise HTTPException(503, "database unavailable")
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeCreate):
    doc = body.model_dump()
    doc["created_at"] = now_utc()
    try:
        db.employees.insert_one(doc)  # the unique index decides; no check-then-insert race
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return serialize_employee(doc)


@app.get("/employees")
def list_employees(department: Optional[str] = None, page: Page = 1, page_size: PageSize = 20):
    query = {} if department is None else {"department": department}
    total = db.employees.count_documents(query)
    cursor = (
        db.employees.find(query, {"_id": 0})
        .sort("emp_code", ASCENDING)
        .skip((page - 1) * page_size)
        .limit(page_size)
    )
    return {"items": [serialize_employee(d) for d in cursor], "total": total, "page": page,
            "page_size": page_size}


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInRequest):
    emp = get_employee_or_404(body.emp_code)
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else now_utc()
    day = attendance_date(ts, emp)
    doc = {
        "emp_code": body.emp_code,
        "date": day.isoformat(),
        "status": body.status,
        "punch_in": ts,
        "punch_out": None,
        **derive_fields(body.status, ts, None, day, emp),
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(doc)  # unique (emp_code, date): exactly one concurrent insert wins
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return serialize_record(doc)


def build_attendance_query(emp_code: Optional[str], date_from: Optional[str], date_to: Optional[str],
                           status: Optional[str]) -> dict:
    if date_from and date_to and date_from > date_to:
        raise validation_error(("query", "date_from"), "date_from must be on or before date_to", date_from)
    query: dict = {}
    if emp_code is not None:
        query["emp_code"] = emp_code
    if date_from or date_to:
        query["date"] = {}
        if date_from:
            query["date"]["$gte"] = date_from
        if date_to:
            query["date"]["$lte"] = date_to
    if status is not None:
        query["status"] = status
    return query


ATTENDANCE_SORT = [("date", DESCENDING), ("emp_code", ASCENDING)]


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[IsoDate] = None,
    date_to: Optional[IsoDate] = None,
    status: Optional[Status] = None,
    page: Page = 1,
    page_size: PageSize = 20,
):
    query = build_attendance_query(emp_code, date_from, date_to, status)
    total = db.attendance_logs.count_documents(query)
    cursor = (
        db.attendance_logs.find(query, {"_id": 0})
        .sort(ATTENDANCE_SORT)
        .skip((page - 1) * page_size)
        .limit(page_size)
    )
    return {"items": [serialize_record(d) for d in cursor], "total": total, "page": page,
            "page_size": page_size}


# --------------------------------------------------------------------------- #
# TODO (Part B) - next steps:
#   POST  /attendance/punch-out
#   PATCH /attendance/{emp_code}/{date}
#   GET   /analytics/employees/{emp_code}/monthly
#   GET   /analytics/departments/summary
#   GET   /analytics/leaderboard/late
#   GET   /analytics/departments/{department}/trend
#   GET   /admin/explain/{endpoint}
# --------------------------------------------------------------------------- #