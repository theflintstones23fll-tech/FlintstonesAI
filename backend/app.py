import os, sys, json, uuid, traceback
from datetime import datetime, timedelta
from functools import wraps
import cv2
import numpy as np

from flask import (
    Flask, request, jsonify, send_from_directory,
    make_response, render_template
)
from flask_jwt_extended import (
    JWTManager, jwt_required, get_jwt_identity,
    create_access_token, create_refresh_token, decode_token
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from PIL import Image
import bcrypt

# ─── Config ──────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

DATABASE_URL = os.environ.get("DATABASE_URL")
SECRET_KEY = os.environ.get("SECRET_KEY") or os.urandom(32).hex()
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY") or os.urandom(32).hex()

# ─── Database ────────────────────────────────────────────────────────────────

db = SQLAlchemy()

def generate_uuid():
    return str(uuid.uuid4())


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    institution = db.Column(db.String(200))
    bio = db.Column(db.Text)
    role = db.Column(db.String(20), default="archaeologist")
    is_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    artifacts = db.relationship("Artifact", back_populates="owner", lazy="dynamic", cascade="all, delete-orphan")
    collections = db.relationship("Collection", back_populates="owner", lazy="dynamic", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id, "username": self.username, "email": self.email,
            "institution": self.institution, "bio": self.bio, "role": self.role,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Artifact(db.Model):
    __tablename__ = "artifacts"
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    era = db.Column(db.String(100), index=True)
    period = db.Column(db.String(100), index=True)
    region = db.Column(db.String(200))
    material = db.Column(db.String(100))
    width_cm = db.Column(db.Float)
    height_cm = db.Column(db.Float)
    area_cm2 = db.Column(db.Float)
    image_filename = db.Column(db.String(500), nullable=False)
    thumbnail_filename = db.Column(db.String(500))
    dominant_colors = db.Column(db.Text)
    polygon_data = db.Column(db.Text)
    polygon_px = db.Column(db.Text)
    px_per_cm = db.Column(db.Float)
    hausdorff_score = db.Column(db.Float)
    meter_size_cm = db.Column(db.Float, nullable=False, default=8.0)
    meter_polygon_cm = db.Column(db.Text)
    meter_polygon_px = db.Column(db.Text)
    classification_status = db.Column(db.String(20), default="pending")
    is_deleted = db.Column(db.Boolean, default=False)
    owner_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    owner = db.relationship("User", back_populates="artifacts")
    collection_id = db.Column(db.String(36), db.ForeignKey("collections.id"))
    collection = db.relationship("Collection", back_populates="artifacts")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self, include_polygon=False):
        data = {
            "id": self.id, "name": self.name, "description": self.description,
            "era": self.era, "period": self.period, "region": self.region,
            "material": self.material, "width_cm": self.width_cm,
            "height_cm": self.height_cm, "area_cm2": self.area_cm2,
            "image_url": f"/uploads/{self.image_filename}",
            "thumbnail_url": f"/uploads/{self.thumbnail_filename}" if self.thumbnail_filename else None,
            "dominant_colors": self._parse_json(self.dominant_colors),
            "meter_size_cm": self.meter_size_cm,
            "meter_polygon_cm": self._parse_json(self.meter_polygon_cm),
            "classification_status": self.classification_status,
            "owner": self.owner.to_dict() if self.owner else None,
            "collection_id": self.collection_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_polygon:
            data["polygon_data"] = self._parse_json(self.polygon_data)
        return data

    @staticmethod
    def _parse_json(value):
        if not value:
            return None
        try:
            return json.loads(value)
        except Exception:
            return None


class Collection(db.Model):
    __tablename__ = "collections"
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    era = db.Column(db.String(100))
    period = db.Column(db.String(100))
    owner_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    owner = db.relationship("User", back_populates="collections")
    artifacts = db.relationship("Artifact", back_populates="collection", lazy="dynamic", cascade="all, delete-orphan")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self, include_artifacts=False):
        data = {
            "id": self.id, "name": self.name, "description": self.description,
            "era": self.era, "period": self.period,
            "artifact_count": self.artifacts.filter_by(is_deleted=False).count(),
            "owner": self.owner.to_dict() if self.owner else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_artifacts:
            data["artifacts"] = [a.to_dict() for a in self.artifacts.filter_by(is_deleted=False).all()]
        return data


class AuditLog(db.Model):
    __tablename__ = "audit_logs"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"))
    action = db.Column(db.String(100), nullable=False)
    resource = db.Column(db.String(100))
    resource_id = db.Column(db.String(36))
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(500))
    details = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(password, password_hash):
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))

def strong_password(password):
    import re
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain an uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain a lowercase letter."
    if not re.search(r"[0-9]", password):
        return False, "Password must contain a digit."
    return True, ""

def allowed_file(filename):
    ALLOWED = {"png", "jpg", "jpeg", "webp", "bmp"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED

def log_audit(user_id, action, resource=None, resource_id=None, details=None):
    try:
        db.session.add(AuditLog(
            user_id=user_id, action=action, resource=resource, resource_id=resource_id,
            ip_address=request.remote_addr,
            user_agent=str(request.user_agent)[:500] if request.user_agent else None,
            details=details,
        ))
        db.session.commit()
    except Exception:
        pass

def save_upload(file):
    if not file or file.filename == "" or not allowed_file(file.filename):
        return None
    ext = file.filename.rsplit(".", 1)[1].lower()
    safe_name = f"{uuid.uuid4().hex}.{ext}"
    file.save(os.path.join(UPLOAD_DIR, safe_name))
    return safe_name

def create_thumbnail(filename):
    try:
        src = os.path.join(UPLOAD_DIR, filename)
        thumb_name = f"thumb_{filename}"
        dst = os.path.join(UPLOAD_DIR, thumb_name)
        img = Image.open(src)
        img.thumbnail((300, 300), Image.Resampling.LANCZOS)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            bg.save(dst, "JPEG", quality=85)
        else:
            img.save(dst, "JPEG", quality=85)
        return thumb_name
    except Exception:
        return None

def extract_average_rgb(image_path, polygon_px=None):
    try:
        import cv2
        import numpy as np
        img = cv2.imread(image_path)
        if img is None:
            return None
        orig_h, orig_w = img.shape[:2]
        if polygon_px:
            pts = np.array([[int(x), int(y)] for x, y in polygon_px], dtype=np.int32)
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            masked = img.copy()
            masked[mask == 0] = 0
            pixels = img.reshape(-1, 3)[mask.reshape(-1) > 0]
        else:
            pixels = img.reshape(-1, 3)
        if len(pixels) == 0:
            return None
        avg = np.mean(pixels, axis=0)
        return [int(avg[0]), int(avg[1]), int(avg[2])]
    except Exception:
        return None


def add_scale_bar_to_image(img, px_per_cm, meter_cm):
    h, w = img.shape[:2]
    bar_length_cm = meter_cm
    bar_length_px = int(bar_length_cm * px_per_cm)
    if bar_length_px > w - 40:
        bar_length_cm = meter_cm / 2
        bar_length_px = int(bar_length_cm * px_per_cm)
    if bar_length_px > w - 40:
        bar_length_cm = meter_cm / 4
        bar_length_px = int(bar_length_cm * px_per_cm)
    bar_x, bar_y = 20, h - 30
    bar_color = (255, 255, 255)
    bar_thick = max(2, int(h * 0.005))
    cv2.line(img, (bar_x, bar_y), (bar_x + bar_length_px, bar_y), (0, 0, 0), bar_thick + 2)
    cv2.line(img, (bar_x, bar_y), (bar_x + bar_length_px, bar_y), bar_color, bar_thick)
    tick_h = max(4, int(h * 0.015))
    cv2.line(img, (bar_x, bar_y - tick_h), (bar_x, bar_y + tick_h), (0, 0, 0), max(1, bar_thick // 2))
    cv2.line(img, (bar_x + bar_length_px, bar_y - tick_h), (bar_x + bar_length_px, bar_y + tick_h), (0, 0, 0), max(1, bar_thick // 2))
    if bar_length_cm >= 1:
        label = f"{int(bar_length_cm)} cm"
    else:
        label = f"{bar_length_cm:.1f} cm"
    font_scale = max(0.3, min(0.6, h / 600))
    cv2.putText(img, label, (bar_x + 5, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(1, bar_thick // 2))
    return img

def paginate(query, per_page=20, page=None):
    if page is None:
        page = request.args.get("page", 1, type=int)
    return query.paginate(page=page, per_page=min(per_page, 100), error_out=False)


# ─── App Setup ───────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))
app.config["SECRET_KEY"] = SECRET_KEY
app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)
app.config["JWT_REFRESH_TOKEN_EXPIRES"] = timedelta(days=30)
app.config["JWT_TOKEN_LOCATION"] = ["headers", "cookies"]
app.config["JWT_COOKIE_CSRF_PROTECT"] = False
app.config["JWT_COOKIE_SECURE"] = os.environ.get("FLASK_ENV") == "production"
app.config["JWT_COOKIE_SAMESITE"] = "Lax"
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL or f"sqlite:///{os.path.join(BASE_DIR, 'flintstones.db')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

db.init_app(app)
jwt = JWTManager(app)

with app.app_context():
    db.create_all()


@app.after_request
def security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response


# ─── Auth Helpers ─────────────────────────────────────────────────────────────

def _get_current_user():
    token = request.cookies.get("access_token_cookie")
    if not token:
        return None
    try:
        decoded = decode_token(token)
        user_id = decoded.get("sub")
        if user_id:
            return db.session.get(User, user_id)
    except Exception:
        pass
    return None


# ─── Page Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def page_index():
    return render_template("home.html", user=None)

@app.route("/dashboard")
def page_dashboard():
    user = _get_current_user()
    if not user:
        return render_template("login.html", user=None)
    return render_template("dashboard.html", user=user.to_dict())

@app.route("/sponsor")
def page_sponsor():
    return render_template("sponsor.html", user=None)

@app.route("/login")
def page_login():
    user = _get_current_user()
    if user:
        return render_template("dashboard.html", user=user.to_dict())
    return render_template("login.html", user=None)

@app.route("/register")
def page_register():
    user = _get_current_user()
    if user:
        return render_template("dashboard.html", user=user.to_dict())
    return render_template("register.html", user=None)

@app.route("/upload")
def page_upload():
    user = _get_current_user()
    if not user:
        return render_template("login.html", user=None)
    return render_template("upload.html", user=user.to_dict())

@app.route("/browse")
def page_browse():
    user = _get_current_user()
    return render_template("browse.html", user=user.to_dict() if user else None)

@app.route("/artifact/<artifact_id>")
def page_artifact(artifact_id):
    artifact = Artifact.query.filter_by(id=artifact_id, is_deleted=False).first()
    user = _get_current_user()
    return render_template("artifact.html", artifact=artifact, user=user.to_dict() if user else None)

@app.route("/reconstruct")
def page_reconstruct():
    user = _get_current_user()
    if not user:
        return render_template("login.html", user=None)
    return render_template("reconstruct.html", user=user.to_dict())

@app.route("/collections")
def page_collections():
    user = _get_current_user()
    if not user:
        return render_template("login.html", user=None)
    return render_template("collections.html", user=user.to_dict())


# ─── Auth API ─────────────────────────────────────────────────────────────────

@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.get_json() or {}
    identity = (data.get("username") or "").strip()
    password = data.get("password", "")
    if not identity or not password:
        return jsonify({"error": "Username/email and password required."}), 400
    user = User.query.filter((User.username == identity) | (User.email == identity)).first()
    if not user or not verify_password(password, user.password_hash):
        return jsonify({"error": "Invalid credentials."}), 401
    access = create_access_token(identity=user.id)
    refresh = create_refresh_token(identity=user.id)
    log_audit(user.id, "login", "user", user.id)
    return jsonify({"access_token": access, "refresh_token": refresh, "user": user.to_dict()})


@app.route("/login", methods=["POST"])
def page_login_submit():
    identity = (request.form.get("username") or "").strip()
    password = request.form.get("password", "")
    if not identity or not password:
        return render_template("login.html", user=None, error="Username and password required.")
    user = User.query.filter((User.username == identity) | (User.email == identity)).first()
    if not user or not verify_password(password, user.password_hash):
        return render_template("login.html", user=None, error="Invalid credentials.")
    access = create_access_token(identity=user.id)
    log_audit(user.id, "login", "user", user.id)
    resp = make_response(render_template("dashboard.html", user=user.to_dict()))
    resp.set_cookie("access_token_cookie", access, max_age=60 * 60, httponly=False, secure=False, samesite="Lax")
    resp.set_cookie("flintai_token", access, max_age=60 * 60, httponly=False, secure=False, samesite="Lax", path="/")
    return resp


@app.route("/api/auth/register", methods=["POST"])
def api_register():
    data = request.get_json() or {}
    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password", "")
    institution = (data.get("institution") or "").strip()
    if not username or len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters."}), 400
    if not email or "@" not in email:
        return jsonify({"error": "A valid email is required."}), 400
    valid, msg = strong_password(password)
    if not valid:
        return jsonify({"error": msg}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already taken."}), 409
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered."}), 409
    user = User(username=username, email=email, password_hash=hash_password(password),
                institution=institution, role="archaeologist")
    db.session.add(user)
    db.session.commit()
    log_audit(user.id, "register", "user", user.id)
    return jsonify({"message": "Account created. Welcome, archaeologist!", "user": user.to_dict()}), 201


@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    resp = make_response(jsonify({"message": "Logged out."}))
    resp.delete_cookie("access_token_cookie")
    resp.delete_cookie("flintai_token")
    return resp


@app.route("/api/auth/refresh", methods=["POST"])
@jwt_required(refresh=True)
def api_refresh():
    return jsonify({"access_token": create_access_token(identity=get_jwt_identity())})


# ─── User API ─────────────────────────────────────────────────────────────────

@app.route("/api/users/me", methods=["GET"])
@jwt_required()
def get_me():
    user = db.session.get(User, get_jwt_identity())
    if not user:
        return jsonify({"error": "User not found."}), 404
    return jsonify(user.to_dict())

@app.route("/api/users/me", methods=["PUT"])
@jwt_required()
def update_me():
    user = db.session.get(User, get_jwt_identity())
    if not user:
        return jsonify({"error": "User not found."}), 404
    data = request.get_json() or {}
    for field in ("username", "email", "institution", "bio"):
        if field in data:
            val = data[field].strip() if isinstance(data[field], str) else data[field]
            if field == "username" and val != user.username and User.query.filter_by(username=val).first():
                return jsonify({"error": "Username taken."}), 409
            if field == "email":
                val = val.lower()
                if User.query.filter_by(email=val).first():
                    return jsonify({"error": "Email taken."}), 409
            setattr(user, field, val)
    db.session.commit()
    return jsonify(user.to_dict())


# ─── Artifact API ─────────────────────────────────────────────────────────────

@app.route("/api/artifacts", methods=["GET"])
@jwt_required()
def list_artifacts():
    query = Artifact.query.filter_by(is_deleted=False)
    if request.args.get("era"):
        query = query.filter(Artifact.era == request.args.get("era"))
    if request.args.get("period"):
        query = query.filter(Artifact.period == request.args.get("period"))
    if request.args.get("material"):
        query = query.filter(Artifact.material == request.args.get("material"))
    q = request.args.get("q", "").strip()
    if q:
        query = query.filter((Artifact.name.ilike(f"%{q}%")) | (Artifact.description.ilike(f"%{q}%")))
    if request.args.get("owner_id"):
        query = query.filter(Artifact.owner_id == request.args.get("owner_id"))
    pg = paginate(query.order_by(Artifact.created_at.desc()))
    return jsonify({"items": [a.to_dict() for a in pg.items], "total": pg.total, "pages": pg.pages, "page": pg.page})

@app.route("/api/artifacts/mine", methods=["GET"])
@jwt_required()
def my_artifacts():
    pg = paginate(Artifact.query.filter_by(owner_id=get_jwt_identity(), is_deleted=False), per_page=50)
    return jsonify({"items": [a.to_dict() for a in pg.items], "total": pg.total, "pages": pg.pages, "page": pg.page})

@app.route("/api/artifacts/<artifact_id>", methods=["GET"])
def get_artifact(artifact_id):
    artifact = Artifact.query.filter_by(id=artifact_id, is_deleted=False).first()
    if not artifact:
        return jsonify({"error": "Artifact not found."}), 404
    include_polygon = request.args.get("include_polygon", "false") == "true"
    return jsonify(artifact.to_dict(include_polygon=include_polygon))

@app.route("/api/artifacts", methods=["POST"])
@jwt_required()
def upload_artifact():
    user_id = get_jwt_identity()
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error": "Image file is required."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed."}), 400

    filename = save_upload(file)
    if not filename:
        return jsonify({"error": "Failed to save file."}), 500

    name = request.form.get("name", "").strip()
    if not name:
        name = f"Artifact {datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    meter_size_cm = float(request.form.get("meter_size_cm", 8.0))
    multi = request.form.get("multi", "false").lower() in ("true", "1", "yes")
    filepath = os.path.join(UPLOAD_DIR, filename)

    common_fields = {
        "description": request.form.get("description", "").strip(),
        "era": request.form.get("era", "").strip() or None,
        "period": request.form.get("period", "").strip() or None,
        "region": request.form.get("region", "").strip() or None,
        "material": request.form.get("material", "").strip() or None,
        "meter_size_cm": meter_size_cm,
        "owner_id": user_id,
    }

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if multi:
        from InstanceSegmentation import get_multi_artifact_data
        try:
            result = get_multi_artifact_data(filepath, meter_size_cm, min_area=500)
            if not result or result.get('error'):
                return jsonify({"error": result.get('error', 'No artifacts detected.')}), 400
            artifacts_data = result.get('artifacts', [])
            if not artifacts_data:
                return jsonify({"error": "No artifacts detected in image."}), 400

            created = []
            for i, art in enumerate(artifacts_data):
                suffix = f" #{i+1}" if len(artifacts_data) > 1 else ""
                art_name = f"{name}{suffix}"
                orig_filename = f"{uuid.uuid4().hex[:12]}_orig_{i}.png"
                crop_filename = f"{uuid.uuid4().hex[:12]}_crop_{i}.png"
                orig_filepath = os.path.join(UPLOAD_DIR, orig_filename)
                crop_filepath = os.path.join(UPLOAD_DIR, crop_filename)
                cv2.imwrite(orig_filepath, cv2.imread(filepath))
                crop = art.get('crop')
                if crop is not None:
                    cv2.imwrite(crop_filepath, crop)

                artifact = Artifact(
                    name=art_name,
                    description=common_fields["description"],
                    era=common_fields["era"],
                    period=common_fields["period"],
                    region=common_fields["region"],
                    material=common_fields["material"],
                    image_filename=orig_filename,
                    thumbnail_filename=crop_filename,
                    meter_size_cm=meter_size_cm,
                    owner_id=user_id,
                    classification_status="analyzed",
                )
                dims = art.get('dimensions', {})
                artifact.polygon_data = json.dumps(art.get('polygon_cm'))
                artifact.polygon_px = json.dumps(art.get('polygon_px'))
                artifact.px_per_cm = result.get('scale', {}).get('pixels_per_cm', 0)
                meter_poly_cm = result.get('meter', {}).get('polygon_cm')
                meter_poly_px = result.get('meter', {}).get('polygon_px')
                if meter_poly_cm:
                    artifact.meter_polygon_cm = json.dumps(meter_poly_cm)
                if meter_poly_px:
                    artifact.meter_polygon_px = json.dumps(meter_poly_px)
                artifact.width_cm = dims.get('width_cm')
                artifact.height_cm = dims.get('height_cm')
                artifact.area_cm2 = dims.get('area_cm2')
                polygon_px = art.get('polygon_px')
                if polygon_px and crop is not None:
                    artifact.dominant_colors = json.dumps(extract_average_rgb(crop_filepath, polygon_px))
                db.session.add(artifact)
                created.append(artifact)

            db.session.commit()
            log_audit(user_id, "upload_multi", "artifacts", ",".join(a.id for a in created))
            return jsonify({"ids": [a.id for a in created], "count": len(created)}), 201
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"Multi-upload failed: {e}"}), 500

    artifact = Artifact(
        name=name,
        description=common_fields["description"],
        era=common_fields["era"],
        period=common_fields["period"],
        region=common_fields["region"],
        material=common_fields["material"],
        image_filename=filename,
        thumbnail_filename=create_thumbnail(filename),
        meter_size_cm=meter_size_cm,
        owner_id=user_id,
        classification_status="pending",
    )
    db.session.add(artifact)
    db.session.commit()
    log_audit(user_id, "upload_artifact", "artifact", artifact.id)

    try:
        from InstanceSegmentation import get_all_artifact_polygons
        result = get_all_artifact_polygons(filepath, meter_size_cm, min_area=500)
        if result and isinstance(result, dict) and result.get('artifact'):
            art_data = result['artifact']
            dims = art_data.get('dimensions', {})
            scale = result.get('scale', {})
            artifact.polygon_data = json.dumps(art_data.get('polygon'))
            artifact.polygon_px = json.dumps(art_data.get('polygon_px'))
            artifact.px_per_cm = scale.get('pixels_per_cm')
            artifact.width_cm = dims.get('width_cm')
            artifact.height_cm = dims.get('height_cm')
            artifact.area_cm2 = dims.get('area_cm2')
            artifact.classification_status = "analyzed"
            polygon_px = art_data.get('polygon_px')
            if polygon_px:
                artifact.dominant_colors = json.dumps(extract_average_rgb(filepath, polygon_px))
            db.session.commit()
    except Exception:
        pass

    return jsonify(artifact.to_dict(include_polygon=True)), 201

@app.route("/api/artifacts/<artifact_id>", methods=["PUT"])
@jwt_required()
def update_artifact(artifact_id):
    user_id = get_jwt_identity()
    artifact = Artifact.query.filter_by(id=artifact_id, is_deleted=False).first()
    if not artifact:
        return jsonify({"error": "Artifact not found."}), 404
    if artifact.owner_id != user_id:
        return jsonify({"error": "Permission denied."}), 403
    data = request.get_json() or {}
    for field in ("name", "description", "era", "period", "region", "material", "classification_status"):
        if field in data:
            val = data[field]
            if isinstance(val, str):
                val = val.strip() or None
            setattr(artifact, field, val)
    if "meter_size_cm" in data:
        artifact.meter_size_cm = float(data["meter_size_cm"])
    db.session.commit()
    return jsonify(artifact.to_dict(include_polygon=True))

@app.route("/api/artifacts/<artifact_id>", methods=["DELETE"])
@jwt_required()
def delete_artifact(artifact_id):
    user_id = get_jwt_identity()
    artifact = Artifact.query.filter_by(id=artifact_id, is_deleted=False).first()
    if not artifact:
        return jsonify({"error": "Artifact not found."}), 404
    if artifact.owner_id != user_id:
        return jsonify({"error": "Permission denied."}), 403
    artifact.is_deleted = True
    db.session.commit()
    return jsonify({"message": "Artifact deleted."})


# ─── Matching API ─────────────────────────────────────────────────────────────

@app.route("/api/artifacts/<artifact_id>/match", methods=["GET"])
@jwt_required()
def match_artifact(artifact_id):
    source = Artifact.query.filter_by(id=artifact_id, is_deleted=False).first()
    if not source:
        return jsonify({"error": "Artifact not found."}), 404
    limit = min(int(request.args.get("limit", 10)), 50)
    candidates = Artifact.query.filter(
        Artifact.is_deleted == False, Artifact.id != artifact_id,
        Artifact.polygon_data.isnot(None)
    ).all()
    if not source.polygon_data:
        return jsonify({"error": "Source artifact has no polygon data."}), 400

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from InstanceSegmentation import match_fragments

    results = []
    try:
        polyA = source._parse_json(source.polygon_data)
        for cand in candidates:
            if not cand.polygon_data:
                continue
            polyB = cand._parse_json(cand.polygon_data)
            mr = match_fragments(polyA, polyB)
            raw_score = mr.get("score", 999) or 999
            norm_score = min(raw_score / 2.0, 1.0)
            match_type = ("excellent" if norm_score < 0.15
                          else "good" if norm_score < 0.35
                          else "moderate" if norm_score < 0.5
                          else "poor")
            results.append({
                "artifact": cand.to_dict(),
                "score": norm_score,
                "match_type": match_type,
            })
    except Exception as e:
        return jsonify({"error": f"Matching failed: {e}"}), 500

    results.sort(key=lambda x: x["score"])
    return jsonify({"matches": results[:limit]})

@app.route("/api/artifacts/compare-batch", methods=["POST"])
@jwt_required()
def compare_batch():
    data = request.get_json() or {}
    artifact_ids = data.get("artifact_ids", [])
    if len(artifact_ids) < 2 or len(artifact_ids) > 10:
        return jsonify({"error": "Need 2-10 artifacts."}), 400
    artifacts = Artifact.query.filter(
        Artifact.id.in_(artifact_ids), Artifact.is_deleted == False,
        Artifact.polygon_data.isnot(None)
    ).all()
    if len(artifacts) < 2:
        return jsonify({"error": "Need at least 2 artifacts with polygon data."}), 400

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from InstanceSegmentation import match_fragments

    results = []
    for i, a in enumerate(artifacts):
        for j, b in enumerate(artifacts):
            if i >= j:
                continue
            mr = match_fragments(a._parse_json(a.polygon_data), b._parse_json(b.polygon_data))
            raw_score = mr.get("score", 999) or 999
            norm_score = min(raw_score / 2.0, 1.0)
            match_type = ("excellent" if norm_score < 0.15
                          else "good" if norm_score < 0.35
                          else "moderate" if norm_score < 0.5
                          else "poor")
            results.append({
                "artifact_a": a.to_dict(), "artifact_b": b.to_dict(),
                "score": norm_score,
                "match_type": match_type,
            })
    results.sort(key=lambda x: x["score"])
    return jsonify({"comparisons": results})


# ─── Reconstruct API ─────────────────────────────────────────────────────────

@app.route("/api/reconstruct", methods=["POST"])
@jwt_required()
def reconstruct():
    user_id = get_jwt_identity()
    data = request.get_json() or {}
    artifact_ids = data.get("artifact_ids", [])
    if len(artifact_ids) < 2:
        return jsonify({"error": "At least 2 artifacts required."}), 400

    artifacts = Artifact.query.filter(
        Artifact.id.in_(artifact_ids), Artifact.is_deleted == False
    ).order_by(Artifact.area_cm2.desc()).all()
    if len(artifacts) < 2:
        return jsonify({"error": "Not enough valid artifacts found."}), 400

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    static_dir = os.path.join(BASE_DIR, "static")
    os.makedirs(static_dir, exist_ok=True)
    output_name = f"reconstruction_{uuid.uuid4().hex[:8]}.png"
    output_path = os.path.join(static_dir, output_name)

    try:
        from InstanceSegmentation import reconstruct_multi_separated

        image_meters = []
        artifact_names = []
        for i, a in enumerate(artifacts):
            fp = os.path.join(UPLOAD_DIR, a.image_filename)
            if os.path.exists(fp):
                meter_poly_px = Artifact._parse_json(a.meter_polygon_px)
                meter_poly_cm = Artifact._parse_json(a.meter_polygon_cm)
                art_poly_px = Artifact._parse_json(a.polygon_px)
                image_meters.append((fp, artifacts[i].meter_size_cm or 8.0, meter_poly_px, meter_poly_cm, art_poly_px))
                artifact_names.append(a.name)

        if len(image_meters) < 2:
            return jsonify({"error": "Not enough fragments detected."}), 500

        result = reconstruct_multi_separated(image_meters, output_path, artifact_names=artifact_names)
        if result is None:
            return jsonify({"error": "Fragments do not connect."}), 400

        log_audit(user_id, "reconstruct", "artifacts", ",".join(artifact_ids))
        return jsonify({"reconstruction_url": f"/static/{output_name}", "artifact_count": len(artifacts),
                        "artifacts": [a.to_dict() for a in artifacts]})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Reconstruction failed: {e}"}), 500


# ─── Collections API ─────────────────────────────────────────────────────────

@app.route("/api/collections", methods=["GET"])
@jwt_required()
def list_collections():
    query = Collection.query
    if request.args.get("mine") == "true":
        uid = get_jwt_identity()
        query = query.filter_by(owner_id=uid)
    pg = paginate(query)
    return jsonify({"items": [c.to_dict() for c in pg.items], "total": pg.total, "pages": pg.pages, "page": pg.page})

@app.route("/api/collections", methods=["POST"])
@jwt_required()
def create_collection():
    user_id = get_jwt_identity()
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Collection name is required."}), 400
    collection = Collection(name=name, description=data.get("description", "").strip() or None,
                           era=data.get("era", "").strip() or None,
                           period=data.get("period", "").strip() or None,
                           owner_id=user_id)
    db.session.add(collection)
    db.session.commit()
    log_audit(user_id, "create_collection", "collection", collection.id)
    return jsonify(collection.to_dict()), 201

@app.route("/api/collections/<collection_id>", methods=["GET"])
@jwt_required()
def get_collection(collection_id):
    collection = db.session.get(Collection, collection_id)
    if not collection:
        return jsonify({"error": "Collection not found."}), 404
    return jsonify(collection.to_dict(include_artifacts=True))

@app.route("/api/collections/<collection_id>", methods=["PUT"])
@jwt_required()
def update_collection(collection_id):
    user_id = get_jwt_identity()
    collection = db.session.get(Collection, collection_id)
    if not collection or collection.owner_id != user_id:
        return jsonify({"error": "Collection not found or access denied."}), 404
    data = request.get_json() or {}
    for field in ("name", "description", "era", "period"):
        if field in data:
            val = data[field]
            if isinstance(val, str):
                val = val.strip() or None
            setattr(collection, field, val)
    db.session.commit()
    return jsonify(collection.to_dict())

@app.route("/api/collections/<collection_id>", methods=["DELETE"])
@jwt_required()
def delete_collection(collection_id):
    user_id = get_jwt_identity()
    collection = db.session.get(Collection, collection_id)
    if not collection or collection.owner_id != user_id:
        return jsonify({"error": "Collection not found or access denied."}), 404
    db.session.delete(collection)
    db.session.commit()
    return jsonify({"message": "Collection deleted."})

@app.route("/api/collections/<collection_id>/artifacts/<artifact_id>", methods=["POST"])
@jwt_required()
def add_to_collection(collection_id, artifact_id):
    user_id = get_jwt_identity()
    collection = db.session.get(Collection, collection_id)
    if not collection or collection.owner_id != user_id:
        return jsonify({"error": "Collection not found or access denied."}), 404
    artifact = Artifact.query.filter_by(id=artifact_id, is_deleted=False, owner_id=user_id).first()
    if not artifact:
        return jsonify({"error": "Artifact not found or access denied."}), 404
    artifact.collection_id = collection_id
    db.session.commit()
    return jsonify(collection.to_dict(include_artifacts=True))

@app.route("/api/collections/<collection_id>/artifacts/<artifact_id>", methods=["DELETE"])
@jwt_required()
def remove_from_collection(collection_id, artifact_id):
    user_id = get_jwt_identity()
    collection = db.session.get(Collection, collection_id)
    if not collection or collection.owner_id != user_id:
        return jsonify({"error": "Collection not found or access denied."}), 404
    artifact = Artifact.query.filter_by(id=artifact_id, collection_id=collection_id).first()
    if not artifact:
        return jsonify({"error": "Artifact not in this collection."}), 404
    artifact.collection_id = None
    db.session.commit()
    return jsonify({"message": "Removed from collection."})


# ─── Uploads ─────────────────────────────────────────────────────────────────

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
