# FlintAI - Archaeological Artifact Analysis System

A full-stack web application for analyzing, comparing, and reconstructing archaeological artifact fragments.

## Quick Start

### Development
```bash
cd backend
pip install -r requirements.txt       # or: python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
python app.py
```
Open **http://localhost:5000** in your browser.

Or use the run script:
```bash
chmod +x run.sh
./run.sh
```

### Production (Docker)
```bash
SECRET_KEY=$(openssl rand -hex 32) \
JWT_SECRET_KEY=$(openssl rand -hex 32) \
DB_PASSWORD=$(openssl rand -hex 16) \
docker compose up --build -d
```
Or use the convenience script:
```bash
chmod +x deploy.sh && ./deploy.sh
```

---

## Project Structure

```
FlintstonesAI/
├── InstanceSegmentation.py     # Core CV algorithm
├── backend/
│   ├── app.py                 # Flask app (HTML pages + API)
│   ├── models.py              # SQLAlchemy models
│   ├── utils.py               # Auth, image processing
│   ├── config.py              # Environment config
│   ├── requirements.txt
│   ├── templates/             # HTML pages (Jinja2)
│   │   ├── base.html
│   │   ├── login.html
│   │   ├── register.html
│   │   ├── dashboard.html
│   │   ├── upload.html
│   │   ├── browse.html
│   │   ├── artifact.html
│   │   ├── reconstruct.html
│   │   └── collections.html
│   ├── static/
│   │   ├── css/style.css     # Earth-tone theme CSS
│   │   └── js/
│   │       ├── api.js        # API fetch wrapper
│   │       └── app.js        # Auth state + helpers
│   ├── uploads/               # Uploaded artifact images
│   └── static/                # Reconstruction outputs
├── docker-compose.yml         # backend + postgres + redis + nginx
├── Dockerfile                 # Backend Docker image
├── run.sh                     # Development startup
└── deploy.sh                  # Production deployment
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `SECRET_KEY` | Flask secret key | Auto-generated |
| `JWT_SECRET_KEY` | JWT signing key | Auto-generated |
| `DATABASE_URL` | PostgreSQL connection | SQLite (`flintstones.db`) |
| `REDIS_URL` | Redis for rate limiting | In-memory |
| `FLASK_ENV` | `development` or `production` | `development` |

---

## Architecture

- **Backend**: Flask with Jinja2 templates — serves both HTML pages and JSON API
- **Frontend**: Plain HTML + CSS + vanilla JS (no build step required)
- **Database**: SQLite (dev) / PostgreSQL (prod)
- **Image Analysis**: OpenCV via `InstanceSegmentation.py`
- **Auth**: JWT in HttpOnly cookies

## Pages

| Route | Description |
|---|---|
| `/` | Dashboard (or redirect to `/login`) |
| `/login` | Login |
| `/register` | Register account |
| `/upload` | Upload artifact image |
| `/browse` | Search/browse public artifacts |
| `/artifact/<id>` | Artifact detail + find similar |
| `/reconstruct` | Select fragments and assemble |
| `/collections` | Organize artifacts into collections |

## API Endpoints

### Auth
- `POST /api/auth/register` — Create account
- `POST /api/auth/login` — Login (returns JWT)
- `POST /api/auth/logout` — Logout

### Artifacts
- `GET /api/artifacts` — List public artifacts
- `GET /api/artifacts/mine` — List own artifacts
- `GET /api/artifacts/<id>` — Artifact detail
- `POST /api/artifacts` — Upload + analyze image
- `PUT /api/artifacts/<id>` — Update metadata
- `DELETE /api/artifacts/<id>` — Soft delete
- `GET /api/artifacts/<id>/match` — Find similar fragments
- `POST /api/artifacts/compare-batch` — Pairwise comparison

### Reconstruction
- `POST /api/reconstruct` — Reconstruct from N fragments

### Collections
- `GET/POST /api/collections` — List/create collections
- `GET/PUT/DELETE /api/collections/<id>` — Collection operations
- `POST /api/collections/<id>/artifacts/<aid>` — Add artifact
- `DELETE /api/collections/<id>/artifacts/<aid>` — Remove artifact
