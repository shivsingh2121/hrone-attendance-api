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
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Any, Literal, Optional

from bson import json_util
from bson.decimal128 import Decimal128
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from pydantic import AfterValidator, BaseModel, Field, model_validator
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
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


class PunchOutRequest(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None


class RegularizeRequest(BaseModel):
    """Only these fields can be changed. Derived fields sent by a client are ignored (extra keys are dropped).
    The changeable fields default to None = "not supplied"; an explicit null is rejected with 422."""
    status: Status = None
    punch_in: EpochMs = None
    punch_out: EpochMs = None
    reason: Annotated[str, Field(min_length=5, max_length=200)]
    regularized_by: Annotated[str, Field(min_length=1, max_length=50)]


Month = Annotated[str, Query(pattern=r"^[0-9]{4}-(0[1-9]|1[0-2])$")]
Limit = Annotated[int, Query(ge=1, le=50)]


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
        # GET /employees?department=... (equality + sort by emp_code); the trend and the
        # department-filtered summary start from this match.
        ([("department", ASCENDING), ("emp_code", ASCENDING)], {"name": "department_emp_code"}),
        # Department summary without a department filter: employees with joined_on <= end of month (R9).
        ([("joined_on", ASCENDING)], {"name": "joined_on"}),
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


MAX_SHIFT = timedelta(hours=24)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutRequest):
    emp = get_employee_or_404(body.emp_code)
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else now_utc()
    # The most recent record whose punch_in is at or before the punch-out. Records are one per attendance date
    # and punch_in always lies on its own attendance date, so "latest date" = "latest punch_in".
    # Served by the unique (emp_code, date) index; works across midnight for overnight shifts.
    record = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$ne": None, "$lte": ts}},
        sort=[("date", DESCENDING)],
    )
    if record is None:
        raise HTTPException(404, "no punch-in found")
    if record.get("punch_out") is not None:
        raise HTTPException(409, "already punched out")
    if ts <= record["punch_in"]:
        raise validation_error(("body", "punched_at"), "punch-out must be after punch-in", body.punched_at)
    if ts - record["punch_in"] > MAX_SHIFT:
        raise validation_error(("body", "punched_at"), "punch-out is more than 24 hours after punch-in",
                               body.punched_at)

    day = date.fromisoformat(record["date"])
    derived = derive_fields(record["status"], record["punch_in"], ts, day, emp)
    # Conditional update: only succeeds while punch_out is still null, so of two simultaneous
    # punch-outs exactly one matches; the other gets None back and answers 409.
    updated = db.attendance_logs.find_one_and_update(
        {"_id": record["_id"], "punch_out": None},
        {"$set": {"punch_out": ts, **derived}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "already punched out")
    return serialize_record(updated)


AUDITED_FIELDS = ("status", "punch_in", "punch_out", "work_hours", "late_minutes", "overtime_minutes", "half_day")


def current_values(record: dict) -> dict:
    """The stored values of the audited fields, with the documented defaults for missing ones."""
    return {
        "status": record["status"],
        "punch_in": record.get("punch_in"),
        "punch_out": record.get("punch_out"),
        "work_hours": record.get("work_hours"),
        "late_minutes": record.get("late_minutes") or 0,
        "overtime_minutes": record.get("overtime_minutes") or 0,
        "half_day": bool(record.get("half_day", False)),
    }


@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(emp_code: str, date: IsoDate, body: RegularizeRequest):
    emp = get_employee_or_404(emp_code)
    record = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if record is None:
        raise HTTPException(404, "no attendance record for that date")

    supplied = body.model_fields_set
    day = datetime.strptime(date, "%Y-%m-%d").date()
    old = current_values(record)
    status = body.status if "status" in supplied else old["status"]

    if status not in PRESENCE_STATUSES:
        # ABSENT / LEAVE clear both punch times; supplying one with them is contradictory.
        if "punch_in" in supplied or "punch_out" in supplied:
            raise validation_error(("body", "status"), f"{status} clears punch times; do not send punch_in/punch_out")
        new_in, new_out = None, None
    else:
        new_in = ms_to_dt(body.punch_in) if "punch_in" in supplied else old["punch_in"]
        new_out = ms_to_dt(body.punch_out) if "punch_out" in supplied else old["punch_out"]
        if new_in is None:
            raise validation_error(("body", "punch_in"), "a presence status requires a punch_in")
        if "punch_in" in supplied and attendance_date(new_in, emp) != day:
            raise validation_error(("body", "punch_in"), "punch_in must stay on the record's attendance date",
                                   body.punch_in)
        if new_out is not None and not (new_in < new_out <= new_in + MAX_SHIFT):
            raise validation_error(("body", "punch_out"), "punch_out must be after punch_in and within 24 hours",
                                   body.punch_out)

    new = {"status": status, "punch_in": new_in, "punch_out": new_out,
           **derive_fields(status, new_in, new_out, day, emp)}
    changes = {f: {"from": old[f], "to": new[f]} for f in AUDITED_FIELDS if old[f] != new[f]}
    if not changes:
        raise validation_error(("body",), "the request changes nothing")

    entry = {"at": now_utc(), "by": body.regularized_by, "reason": body.reason, "changes": changes}
    # Optimistic concurrency: the update only applies if the record is still exactly as we read it
    # (same punch times, status and number of history entries). A concurrent edit makes it miss -> 409,
    # so no history entry is ever silently lost.
    history_len = len(record.get("history") or [])
    unchanged = {
        "_id": record["_id"],
        "status": record["status"],
        "punch_in": record.get("punch_in"),
        "punch_out": record.get("punch_out"),
    }
    if history_len:
        unchanged["history"] = {"$size": history_len}
    else:  # legacy records may have no history field at all
        unchanged["$or"] = [{"history": None}, {"history": {"$size": 0}}]
    updated = db.attendance_logs.find_one_and_update(
        unchanged,
        {"$set": new, "$push": {"history": entry}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "record was modified concurrently; reload and retry")
    return serialize_record(updated)


# --------------------------------------------------------------------------- #
# 7. Analytics: MongoDB aggregation pipelines
#    Every pipeline is built by a function, so the endpoint and /admin/explain run the exact same thing.
#    Rounding: MongoDB's $round is half-to-even, so exact Decimal128 values come back from the database
#    and the final half-up rounding (R8) of each reported number happens here.
# --------------------------------------------------------------------------- #
PRESENCE = list(PRESENCE_STATUSES)
MAX_TREND_DAYS = 92


def month_bounds(month: str) -> tuple[str, str]:
    """'2026-07' -> ('2026-07-01', '2026-07-31'). Plain strings, comparable with the stored `date`."""
    year, mon = map(int, month.split("-"))
    first = date(year, mon, 1)
    last = (first.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    return first.isoformat(), last.isoformat()


def count_weekdays(start: date, end: date) -> int:
    """R7: Monday-Friday between start and end inclusive (calendar arithmetic, no data involved)."""
    return sum(1 for i in range((end - start).days + 1) if (start + timedelta(days=i)).weekday() < 5)


def to_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, Decimal128):
        return value.to_decimal()
    return Decimal(str(value))


def report(value: Any, places: int = 2) -> Optional[float]:
    """R8: half-up rounding of a reported number (2 decimals, rates 4)."""
    dec = to_decimal(value)
    if dec is None:
        return None
    return float(dec.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


# ---- reusable aggregation expressions (prefix is "$" for the root document, "$logs." for a joined log) ----
def is_weekday_expr(date_field: str) -> dict:
    """R7: the record's attendance date falls on Mon-Fri ($isoDayOfWeek: 1 = Monday ... 7 = Sunday)."""
    day = {"$dateFromString": {"dateString": {"$ifNull": [date_field, "1970-01-01"]}, "format": "%Y-%m-%d"}}
    return {"$lte": [{"$isoDayOfWeek": day}, 5]}


def is_present_expr(prefix: str) -> dict:
    """R6: PRESENT, WFH and ON_DUTY count as present."""
    return {"$in": [{"$ifNull": [f"{prefix}status", ""]}, PRESENCE]}


def present_value_expr(prefix: str) -> dict:
    """R5/R6: a present record is worth 1 day, a half day 0.5, anything else 0 (missing half_day = false)."""
    return {"$cond": [is_present_expr(prefix), {"$cond": [{"$eq": [f"{prefix}half_day", True]}, 0.5, 1]}, 0]}


def is_late_expr(prefix: str) -> dict:
    return {"$gt": [{"$ifNull": [f"{prefix}late_minutes", 0]}, 0]}


def half_up_expr(value: dict, places: int) -> dict:
    """Half-up rounding inside the pipeline on Decimal128 (for non-negative values): floor(x * 10^p + 0.5) / 10^p."""
    scale = Decimal128(str(10 ** places))
    return {"$divide": [{"$floor": {"$add": [{"$multiply": [value, scale]}, Decimal128("0.5")]}}, scale]}


def logs_lookup(date_from: str, date_to: str, fields: tuple) -> dict:
    """Join each employee to their logs in [date_from, date_to]. localField/foreignField + the date range
    become one condition on the unique (emp_code, date) index."""
    return {"$lookup": {
        "from": "attendance_logs",
        "localField": "emp_code",
        "foreignField": "emp_code",
        "pipeline": [
            {"$match": {"date": {"$gte": date_from, "$lte": date_to}}},
            {"$project": {"_id": 0, **{f: 1 for f in fields}}},
        ],
        "as": "logs",
    }}


# ---- pipeline builders ----
def employee_monthly_pipeline(emp_code: str, month: str) -> list:
    start, end = month_bounds(month)
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": start, "$lte": end}}},
        {"$group": {
            "_id": None,
            # R7: only Mon-Fri records count towards present_days; weekend records still add late/overtime.
            "present_days": {"$sum": {"$cond": [is_weekday_expr("$date"), present_value_expr("$"), 0]}},
            "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
            "late_count": {"$sum": {"$cond": [is_late_expr("$"), 1, 0]}},
            "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
            "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
        }},
    ]


def department_summary_pipeline(month: str, department: Optional[str]) -> list:
    start, end = month_bounds(month)
    match: dict = {"joined_on": {"$lte": end}}          # R9: people who join later do not count
    if department is not None:
        match["department"] = department
    return [
        # Start from employees (not logs), so people with zero logs are part of the headcount.
        {"$match": match},
        logs_lookup(start, end, ("date", "status", "half_day", "work_hours", "late_minutes")),
        # One row per (employee, log); an employee without logs keeps exactly one row with no `logs`.
        {"$unwind": {"path": "$logs", "preserveNullAndEmptyArrays": True}},
        {"$group": {
            "_id": "$department",
            "members": {"$addToSet": "$emp_code"},
            "present_days": {"$sum": {"$cond": [is_weekday_expr("$logs.date"), present_value_expr("$logs."), 0]}},
            # Mean over records (not over per-employee means); Decimal128 keeps it exact. null when no values.
            "avg_work_hours": {"$avg": {"$cond": [
                {"$and": [is_present_expr("$logs."), {"$isNumber": "$logs.work_hours"}]},
                {"$toDecimal": "$logs.work_hours"},
                None,
            ]}},
            "late_count": {"$sum": {"$cond": [is_late_expr("$logs."), 1, 0]}},
            "total_late_minutes": {"$sum": {"$cond": [is_late_expr("$logs."), "$logs.late_minutes", 0]}},
            "leave_count": {"$sum": {"$cond": [{"$eq": ["$logs.status", "LEAVE"]}, 1, 0]}},
            "on_duty_count": {"$sum": {"$cond": [{"$eq": ["$logs.status", "ON_DUTY"]}, 1, 0]}},
        }},
        {"$project": {
            "_id": 0, "department": "$_id", "headcount": {"$size": "$members"}, "present_days": 1,
            "avg_work_hours": 1, "late_count": 1, "total_late_minutes": 1, "leave_count": 1, "on_duty_count": 1,
        }},
        {"$sort": {"department": 1}},
    ]


def late_leaderboard_pipeline(month: str, limit: int, department: Optional[str]) -> list:
    start, end = month_bounds(month)
    pipeline = [
        {"$match": {"date": {"$gte": start, "$lte": end}, "late_minutes": {"$gt": 0}}},
        {"$group": {"_id": "$emp_code", "total_late_minutes": {"$sum": "$late_minutes"}, "late_count": {"$sum": 1}}},
        {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "emp"}},
        {"$unwind": "$emp"},  # drops logs whose employee does not exist
    ]
    if department is not None:
        pipeline.append({"$match": {"emp.department": department}})  # rank inside the department only
    pipeline += [
        # Standard competition ranking: ties share a rank and the next rank is skipped (1, 2, 2, 4).
        {"$setWindowFields": {"sortBy": {"total_late_minutes": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}},   # limit applies after ranking: ties at the cutoff are all kept
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {"$project": {
            "_id": 0, "rank": 1, "emp_code": "$_id", "name": "$emp.name", "department": "$emp.department",
            "total_late_minutes": 1, "late_count": 1,
        }},
    ]
    return pipeline


def department_trend_pipeline(department: str, date_from: str, date_to: str) -> list:
    lower = datetime.combine(date.fromisoformat(date_from), time(), tzinfo=UTC)
    upper = datetime.combine(date.fromisoformat(date_to) + timedelta(days=1), time(), tzinfo=UTC)  # exclusive
    return [
        {"$match": {"department": department}},
        logs_lookup(date_from, date_to, ("date", "status", "half_day", "late_minutes")),
        # Turn every employee into "events" on calendar days:
        #   a zero event on `from` (guarantees at least one document for $densify),
        #   a join event (+1 headcount) on max(joined_on, from) if they joined by `to` (R9),
        #   one event per log carrying its presence value (half day 0.5) and lateness.
        {"$project": {"_id": 0, "events": {"$concatArrays": [
            [{"date": date_from, "joins": 0, "present": 0, "late": 0}],
            {"$cond": [{"$lte": ["$joined_on", date_to]},
                       [{"date": {"$max": ["$joined_on", date_from]}, "joins": 1, "present": 0, "late": 0}],
                       []]},
            {"$map": {"input": "$logs", "as": "l", "in": {
                "date": "$$l.date", "joins": 0,
                "present": present_value_expr("$$l."),
                "late": {"$cond": [is_late_expr("$$l."), 1, 0]},
            }}},
        ]}}},
        {"$unwind": "$events"},
        {"$group": {"_id": "$events.date", "joins": {"$sum": "$events.joins"},
                    "present": {"$sum": "$events.present"}, "late": {"$sum": "$events.late"}}},
        {"$project": {"_id": 0, "day": {"$dateFromString": {"dateString": "$_id", "format": "%Y-%m-%d"}},
                      "joins": 1, "present": 1, "late": 1}},
        # Gap filling: exactly one document per calendar day in [from, to], weekends and empty days included.
        {"$densify": {"field": "day", "range": {"step": 1, "unit": "day", "bounds": [lower, upper]}}},
        {"$set": {"joins": {"$ifNull": ["$joins", 0]}, "present": {"$ifNull": ["$present", 0]},
                  "late": {"$ifNull": ["$late", 0]}}},
        # headcount on a day = running total of join events up to that day
        {"$setWindowFields": {"sortBy": {"day": 1}, "output": {
            "headcount": {"$sum": "$joins", "window": {"documents": ["unbounded", "current"]}}}}},
        {"$set": {"is_working_day": {"$lte": [{"$isoDayOfWeek": "$day"}, 5]}}},
        {"$set": {"attendance_rate": {"$cond": [
            {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
            half_up_expr({"$divide": [{"$toDecimal": "$present"}, {"$toDecimal": "$headcount"}]}, 4),
            None,
        ]}}},
        # mean of the non-null rates in the current row and up to 6 rows before it, inside the range only
        {"$setWindowFields": {"sortBy": {"day": 1}, "output": {
            "moving_avg_7d": {"$avg": "$attendance_rate", "window": {"documents": [-6, "current"]}}}}},
        {"$project": {"_id": 0, "date": {"$dateToString": {"date": "$day", "format": "%Y-%m-%d"}},
                      "is_working_day": 1, "headcount": 1, "present_count": "$present", "late_count": "$late",
                      "attendance_rate": 1, "moving_avg_7d": 1}},
        {"$sort": {"date": 1}},
    ]


def check_trend_range(date_from: str, date_to: str) -> None:
    for name, value in (("from", date_from), ("to", date_to)):
        try:
            date.fromisoformat(value)
        except ValueError:
            raise validation_error(("query", name), "not a valid date", value)
    if date_to < date_from:
        raise validation_error(("query", "to"), "to must be on or after from", date_to)
    days = (date.fromisoformat(date_to) - date.fromisoformat(date_from)).days + 1
    if days > MAX_TREND_DAYS:
        raise validation_error(("query", "to"), f"range exceeds {MAX_TREND_DAYS} days", date_to)


# ---- endpoints ----
@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: Month):
    emp = get_employee_or_404(emp_code)
    start, end = month_bounds(month)
    row = next(db.attendance_logs.aggregate(employee_monthly_pipeline(emp_code, month)), {})
    # R7: Mon-Fri of the month, counted from joined_on for a mid-month joiner (0 if they joined later).
    first_day = max(date.fromisoformat(start), date.fromisoformat(emp["joined_on"]))
    working_days = count_weekdays(first_day, date.fromisoformat(end))
    present_days = to_decimal(row.get("present_days", 0))
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": report(present_days),
        "leave_days": row.get("leave_days", 0),
        "late_count": row.get("late_count", 0),
        "total_late_minutes": row.get("total_late_minutes", 0),
        "total_overtime_minutes": row.get("total_overtime_minutes", 0),
        "attendance_pct": None if working_days == 0 else report(present_days * 100 / working_days),
    }


@app.get("/analytics/departments/summary")
def department_summary(month: Month, department: Optional[str] = None):
    items = []
    for row in db.employees.aggregate(department_summary_pipeline(month, department)):
        row["present_days"] = report(row["present_days"])
        row["avg_work_hours"] = report(row.get("avg_work_hours"))
        items.append(row)
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(month: Month, limit: Limit = 10, department: Optional[str] = None):
    items = list(db.attendance_logs.aggregate(late_leaderboard_pipeline(month, limit, department)))
    return {"month": month, "items": items}


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    date_from: Annotated[str, Query(alias="from", pattern=DATE_PATTERN)],
    date_to: Annotated[str, Query(alias="to", pattern=DATE_PATTERN)],
):
    check_trend_range(date_from, date_to)
    if db.employees.count_documents({"department": department}, limit=1) == 0:
        raise HTTPException(404, "unknown department")
    items = []
    for row in db.employees.aggregate(department_trend_pipeline(department, date_from, date_to)):
        row["present_count"] = report(row["present_count"])
        row["attendance_rate"] = report(row.get("attendance_rate"), 4)
        row["moving_avg_7d"] = report(row.get("moving_avg_7d"), 4)
        items.append(row)
    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# 8. Explain: proves the main query of each endpoint uses an index
# --------------------------------------------------------------------------- #
ExplainTarget = Literal["attendance_list", "employee_monthly", "department_summary", "late_leaderboard",
                        "department_trend"]


def require(**params) -> None:
    for name, value in params.items():
        if value is None:
            raise validation_error(("query", name), "Field required")


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: ExplainTarget,
    emp_code: Optional[str] = None,
    month: Annotated[Optional[str], Query(pattern=r"^[0-9]{4}-(0[1-9]|1[0-2])$")] = None,
    department: Optional[str] = None,
    limit: Limit = 10,
    date_from: Optional[IsoDate] = None,
    date_to: Optional[IsoDate] = None,
    status: Optional[Status] = None,
    trend_from: Annotated[Optional[str], Query(alias="from", pattern=DATE_PATTERN)] = None,
    trend_to: Annotated[Optional[str], Query(alias="to", pattern=DATE_PATTERN)] = None,
    page: Page = 1,
    page_size: PageSize = 20,
):
    if endpoint == "attendance_list":
        collection = "attendance_logs"
        command = {
            "find": collection,
            "filter": build_attendance_query(emp_code, date_from, date_to, status),
            "sort": dict(ATTENDANCE_SORT),
            "skip": (page - 1) * page_size,
            "limit": page_size,
            "projection": {"_id": 0},
        }
    else:
        if endpoint == "employee_monthly":
            require(emp_code=emp_code, month=month)
            collection, pipeline = "attendance_logs", employee_monthly_pipeline(emp_code, month)
        elif endpoint == "department_summary":
            require(month=month)
            collection, pipeline = "employees", department_summary_pipeline(month, department)
        elif endpoint == "late_leaderboard":
            require(month=month)
            collection, pipeline = "attendance_logs", late_leaderboard_pipeline(month, limit, department)
        else:  # department_trend
            require(department=department, **{"from": trend_from, "to": trend_to})
            check_trend_range(trend_from, trend_to)
            collection, pipeline = "employees", department_trend_pipeline(department, trend_from, trend_to)
        command = {"aggregate": collection, "pipeline": pipeline, "cursor": {}}
    result = db.command("explain", command, verbosity="executionStats")
    return {"endpoint": endpoint, "collection": collection, "explain": json.loads(json_util.dumps(result))}
