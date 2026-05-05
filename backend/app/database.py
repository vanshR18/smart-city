from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from loguru import logger
from app.config import get_settings

settings = get_settings()

# ── Engine ────────────────────────────────────────────────────────────────────
# pool_pre_ping=True: tests connection health before using it from the pool
# pool_size=10: keep up to 10 connections open
# max_overflow=20: allow up to 20 extra connections under heavy load
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=(settings.app_env == "development"),  # log SQL only in dev
)

# ── Session factory ───────────────────────────────────────────────────────────
# autocommit=False: we control when to commit (safer)
# autoflush=False:  we control when changes hit the DB within a session
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


# ── Base class for all ORM models ─────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Dependency for FastAPI routes ─────────────────────────────────────────────
def get_db():
    """
    Yields a DB session per request, always closes it when done.

    Usage in FastAPI:
        @app.get("/something")
        def route(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Utility functions ─────────────────────────────────────────────────────────
def init_db():
    """
    Creates all tables + enables PostGIS extension.
    Call this once at startup.
    """
    with engine.connect() as conn:
        # PostGIS must be enabled before any geometry columns are created
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        conn.commit()
        logger.info("PostGIS extension enabled.")

    # Create all tables defined in ORM models
    from app.models import events  # noqa: F401 — import so Base sees the models
    Base.metadata.create_all(bind=engine)
    logger.info("All database tables created.")


def check_db_connection() -> bool:
    """Ping the database. Returns True if healthy."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False