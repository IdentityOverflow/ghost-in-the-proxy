"""S1 — 20-turn mixed conversation: planning, coding, corrections, follow-ups.

Validation-plan scenario 1. Plants decisions with rationales early, applies a
mid-stream correction, and probes late whether decisions, rationales, the
correction, and the open-items list all survived 20 turns of ordinary work.
"""

from ..harness import Check, Scenario, Turn

SYSTEM = (
    "You are a helpful software engineering assistant working with a solo "
    "developer on their hobby project. Be concise and concrete."
)

SCENARIO = Scenario(
    id="s1-coherence",
    title="20-turn mixed conversation",
    description="Decisions, corrections, and open loops must survive routine work.",
    system_prompt=SYSTEM,
    turns=[
        Turn(
            user=(
                "I want to build 'trailcam', a small web app for tagging wildlife "
                "camera photos. Single user, just me, running on my home server. "
                "Help me plan it."
            ),
            mock_reply="Great — let's outline trailcam.",
        ),
        Turn(
            user=(
                "Let's decide the stack. I know Python. I was considering Postgres "
                "but honestly it's just me using it."
            ),
            mock_reply="For a single user, SQLite is plenty; FastAPI for the backend.",
        ),
        Turn(
            user=(
                "Agreed: SQLite it is, because it's single-user and zero-ops. "
                "Backend FastAPI. For the frontend let's keep it plain htmx, no "
                "React. Also, store the photos as BLOBs in the database for now."
            ),
            mock_reply="Noted: SQLite (single-user), FastAPI, htmx, photos as BLOBs.",
        ),
        Turn(
            user="Sketch the data model for photos, tags, and sightings.",
            mock_reply="photos(id, taken_at, blob), tags(id, name), sightings(photo_id, tag_id, confidence).",
        ),
        Turn(
            user=(
                "Good. One thing to not forget: before we ever call this shipped, "
                "remind me to handle EXIF timezone normalization — my cameras "
                "write local time with no offset."
            ),
            mock_reply="Noted — EXIF timezone normalization is an open item before shipping.",
        ),
        Turn(
            user="Write the SQLAlchemy models for that schema.",
            mock_reply="Here are the models (Photo, Tag, Sighting).",
        ),
        Turn(
            user=(
                "Actually, change of plan on storage: photos go on disk under "
                "/data/photos/YYYY/MM/, only the path goes in the database. BLOBs "
                "were a bad idea — backups got huge."
            ),
            mock_reply="Understood: photos on disk under /data/photos/YYYY/MM/, DB stores paths only.",
        ),
        Turn(
            user="Update the Photo model accordingly and add a helper that builds the disk path.",
            mock_reply="Photo.path column added; build_photo_path(taken_at) helper included.",
        ),
        Turn(
            user="Now the upload endpoint: multipart POST, dedupe by SHA-256 of the file.",
            mock_reply="POST /photos with multipart; compute sha256; skip if hash exists.",
        ),
        Turn(
            user="What's a sensible thumbnail strategy?",
            mock_reply="Generate 320px WebP thumbnails on upload, store beside originals.",
        ),
        Turn(
            user="Write the tagging endpoint: attach/detach tags on a photo.",
            mock_reply="POST/DELETE /photos/{id}/tags with tag name upsert.",
        ),
        Turn(
            user=(
                "I tried the upload code and got 'sqlite3.OperationalError: database "
                "is locked' when uploading two files fast. Thoughts?"
            ),
            mock_reply="Enable WAL mode and use a single writer session; SQLite locks on concurrent writes.",
        ),
        Turn(
            user="Apply that fix in the engine setup code.",
            mock_reply="Engine created with WAL pragma and pool for single writer.",
        ),
        Turn(
            user="Where do photo files live, and what does the database store for each photo?",
            note="probe: mid-stream correction must have overwritten the BLOB decision",
            checks=[
                Check(
                    kind="must_mention",
                    desc="photos stored on disk under /data/photos",
                    patterns=[r"/data/photos", r"on disk"],
                ),
                Check(
                    kind="must_mention",
                    desc="database stores the path, not the image",
                    patterns=[r"path"],
                ),
                Check(
                    kind="must_not_mention",
                    desc="BLOB storage must not be presented as current",
                    patterns=[r"stored?\s+as\s+(a\s+)?blobs?", r"blobs?\s+in\s+the\s+database"],
                ),
            ],
            mock_reply="On disk under /data/photos/YYYY/MM/; the database stores only the path.",
        ),
        Turn(
            user="Add a simple search endpoint: photos by tag and date range.",
            mock_reply="GET /photos?tag=&from=&to= implemented.",
        ),
        Turn(
            user="Let's add a nightly job that reports new sightings per tag.",
            mock_reply="APScheduler nightly job summarizing sightings per tag.",
        ),
        Turn(
            user="Write a README intro paragraph for the repo.",
            mock_reply="README intro drafted.",
        ),
        Turn(
            user="Remind me — why did we pick SQLite over Postgres?",
            note="probe: decision rationale recall after 16 turns",
            checks=[
                Check(
                    kind="must_mention",
                    desc="rationale: single-user / zero-ops simplicity",
                    patterns=[r"single[- ]user", r"just (you|me)", r"zero[- ]ops", r"simpl"],
                ),
            ],
            mock_reply="Because it's single-user and zero-ops; Postgres was overkill.",
        ),
        Turn(
            user="Good. Give me a quick status: what have we built so far, in one paragraph?",
            mock_reply="We planned trailcam, chose the stack, built models, upload with dedupe, tagging, search, WAL fix, nightly report.",
        ),
        Turn(
            user="Before I close the laptop: what's still open before we could call this shipped?",
            note="probe: open-loop recall — the EXIF commitment from turn 5",
            checks=[
                Check(
                    kind="must_mention",
                    desc="EXIF timezone normalization is still open",
                    patterns=[r"exif", r"time\s?zone"],
                ),
                Check(
                    kind="judge",
                    desc="open-items list is plausibly complete and grounded in the conversation",
                    rubric=(
                        "The user asks what is still open before shipping their "
                        "'trailcam' app. During the conversation the only "
                        "explicitly parked item was EXIF timezone normalization "
                        "(cameras write local time with no offset). Does the "
                        "reply include the EXIF timezone item and avoid "
                        "inventing major fake past decisions?"
                    ),
                ),
            ],
            mock_reply="Main open item: EXIF timezone normalization before shipping.",
        ),
    ],
)
