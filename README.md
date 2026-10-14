# Employee Attendance & Analytics API

Built with FastAPI and MongoDB (PyMongo). All the code is in `app/main.py`.

## How to run
```bash
pip install -r requirements.txt
export MONGO_URI="mongodb+srv://..."
export MONGO_DB="attendance_db"
uvicorn app.main:app --port 8000
```
You can also put `MONGO_URI` and `MONGO_DB` in a local `.env` file. Real environment variables always win.

The app creates all indexes when it starts. `GET /health` returns 200 when MongoDB is reachable.
I tested it on MongoDB 8.0 (Atlas). It only uses features that exist in MongoDB 6.0 and newer.

## Good to know
- **Time:** the API uses epoch milliseconds. Inside the app all times are UTC, and dates and shifts are calculated in IST.
- **Rounding:** numbers are rounded half-up using Python's `Decimal`. MongoDB's `$round` does not round half-up, so the final rounding is done in Python.
- **Trend:** a date range longer than 92 days returns 422. The 7-day moving average uses the rounded daily rates.
- **Two requests at the same time:**
  - punch-in: the unique index on `emp_code + date` allows only one record.
  - punch-out: the update only works while `punch_out` is still empty.
  - PATCH: the update only works if nobody changed the record in between; otherwise the second request gets 409, so no history is lost.
- **Explain:** `/admin/explain/{endpoint}` builds the query with the same function as the real endpoint and returns MongoDB's explain output.

## How I tested
I used my own test data with the tricky cases: ties at the limit, weekend records, an employee with no logs, an employee who joined mid-month, night shifts, and logs of an employee who does not exist. I also added about 100,000 random records. I compared every analytics result with a separate Python calculation and checked that the explain output shows IXSCAN and no COLLSCAN.

Nothing from the contract is missing as far as I know.
