"""
Accident Tracker API + map server.

Run:
    pip install fastapi uvicorn "psycopg[binary]"
    export DATABASE_URL="postgresql://user:pass@localhost/accidents"
    uvicorn main:app --reload
Then open http://localhost:8000
"""
import os
from datetime import datetime
from typing import Optional

import psycopg
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://localhost/accidents")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Use a static/ folder if present, otherwise the HTML files can sit next to main.py
STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.isdir(STATIC_DIR):
    STATIC_DIR = BASE_DIR

app = FastAPI(title="Global Accident Tracker", version="0.1")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def db():
    return psycopg.connect(DATABASE_URL)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/incidents")
def incidents(
    bbox: str = Query(..., description="west,south,east,north"),
    accident_type: Optional[str] = None,
    severity: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    limit: int = Query(5000, le=20000),
):
    """Incidents inside a bounding box, as GeoJSON."""
    try:
        w, s, e, n = [float(x) for x in bbox.split(",")]
    except ValueError:
        raise HTTPException(400, "bbox must be west,south,east,north")

    where = [
        "status <> 'rejected'",
        "location && ST_MakeEnvelope(%s, %s, %s, %s, 4326)::geography",
    ]
    params: list = [w, s, e, n]
    if accident_type:
        where.append("accident_type = %s")
        params.append(accident_type)
    if severity:
        where.append("severity = %s")
        params.append(severity)
    if date_from:
        where.append("occurred_at >= %s")
        params.append(date_from)
    if date_to:
        where.append("occurred_at <= %s")
        params.append(date_to)
    params.append(limit)

    sql = f"""
        SELECT incident_id, accident_type, severity, occurred_at,
               vehicles_involved, injuries, fatalities, status, confidence,
               ST_X(location::geometry), ST_Y(location::geometry)
        FROM incidents
        WHERE {' AND '.join(where)}
        ORDER BY occurred_at DESC
        LIMIT %s
    """
    with db() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r[9], r[10]]},
            "properties": {
                "id": r[0],
                "type": r[1],
                "severity": r[2],
                "occurred_at": r[3].isoformat(),
                "vehicles": r[4],
                "injuries": r[5],
                "fatalities": r[6],
                "status": r[7],
                "confidence": r[8],
            },
        }
        for r in rows
    ]
    return {
        "type": "FeatureCollection",
        "features": features,
        "truncated": len(rows) >= limit,
    }


@app.get("/api/stats/monthly")
def monthly(country: Optional[str] = None, accident_type: Optional[str] = None):
    """Incident counts per month and severity, for trend charts."""
    where, params = ["status <> 'rejected'"], []
    if country:
        where.append("country_code = %s")
        params.append(country.upper())
    if accident_type:
        where.append("accident_type = %s")
        params.append(accident_type)

    sql = f"""
        SELECT to_char(date_trunc('month', occurred_at), 'YYYY-MM') AS month,
               severity, COUNT(*)
        FROM incidents
        WHERE {' AND '.join(where)}
        GROUP BY 1, 2
        ORDER BY 1
    """
    with db() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return [{"month": m, "severity": s, "count": c} for m, s, c in cur.fetchall()]


@app.get("/api/coverage")
def coverage():
    """Data coverage per country, so users know where the map is reliable."""
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT country_code, accident_type, year, coverage_level, notes FROM coverage"
        )
        return [
            {"country": a, "type": b, "year": c, "coverage": d, "notes": e}
            for a, b, c, d, e in cur.fetchall()
        ]


# ---------- Analytics ----------

@app.get("/analytics")
def analytics_page():
    return FileResponse(os.path.join(STATIC_DIR, "analytics.html"))


@app.get("/api/stats/hourly")
def hourly(country: Optional[str] = None, accident_type: Optional[str] = None):
    """Incident counts by hour of day and by day of week."""
    where, params = ["status <> 'rejected'"], []
    if country:
        where.append("country_code = %s")
        params.append(country.upper())
    if accident_type:
        where.append("accident_type = %s")
        params.append(accident_type)
    w = " AND ".join(where)
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT EXTRACT(HOUR FROM occurred_at)::int, COUNT(*)
                FROM incidents WHERE {w} GROUP BY 1 ORDER BY 1""", params)
        by_hour = {int(h): c for h, c in cur.fetchall()}
        cur.execute(
            f"""SELECT EXTRACT(DOW FROM occurred_at)::int, COUNT(*)
                FROM incidents WHERE {w} GROUP BY 1 ORDER BY 1""", params)
        by_dow = {int(d): c for d, c in cur.fetchall()}
    return {
        "by_hour": [by_hour.get(h, 0) for h in range(24)],
        "by_weekday": [by_dow.get(d, 0) for d in range(7)],  # 0 = Sunday
    }


@app.get("/api/hotspots")
def hotspots(
    country: Optional[str] = None,
    accident_type: Optional[str] = None,
    cell_size: float = Query(0.02, ge=0.001, le=1.0, description="grid size in degrees (~2 km at 0.02)"),
    min_count: int = Query(5, ge=1),
    limit: int = Query(50, le=500),
):
    """Densest grid cells: a fast, simple hotspot measure."""
    where, params = ["status <> 'rejected'"], [cell_size]
    if country:
        where.append("country_code = %s")
        params.append(country.upper())
    if accident_type:
        where.append("accident_type = %s")
        params.append(accident_type)
    params += [min_count, limit]
    sql = f"""
        SELECT ST_X(cell), ST_Y(cell), cnt, fatal, serious FROM (
            SELECT ST_SnapToGrid(location::geometry, %s) AS cell,
                   COUNT(*) AS cnt,
                   COUNT(*) FILTER (WHERE severity = 'fatal') AS fatal,
                   COUNT(*) FILTER (WHERE severity = 'serious') AS serious
            FROM incidents
            WHERE {' AND '.join(where)}
            GROUP BY cell
            HAVING COUNT(*) >= %s
            ORDER BY cnt DESC
            LIMIT %s
        ) t
    """
    with db() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return [
            {"lng": x, "lat": y, "count": c, "fatal": f, "serious": s}
            for x, y, c, f, s in cur.fetchall()
        ]
