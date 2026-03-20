let currentUser = null;

function showAlert(message, type = 'info') {
  const container = document.getElementById('alert-container');
  if (!container) return;
  const el = document.createElement('div');
  el.className = `alert alert-${type}`;
  el.textContent = message;
  container.innerHTML = '';
  container.appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

function getScoreClass(score) {
  if (score < 0.15) return 'low';
  if (score < 0.35) return 'medium';
  return 'high';
}

function formatScore(score) {
  return typeof score === 'number' ? score.toFixed(3) : '—';
}

function formatDim(w, h, area) {
  if (w && h) return `${w.toFixed(1)} x ${h.toFixed(1)} cm`;
  if (area) return `${area.toFixed(1)} cm²`;
  return 'No measurement';
}

function artifactCardHTML(a) {
  const raw = a.dominant_colors;
  const colors = (raw && Array.isArray(raw[0])) ? raw : (raw ? [raw] : []);
  const tags = [];
  if (a.era) tags.push(`<span class="card-tag">${a.era}</span>`);
  if (a.material) tags.push(`<span class="card-tag">${a.material}</span>`);
  if (a.classification_status) {
    const bg = a.classification_status === 'analyzed' ? '#e8f5e9' : '#fff3e0';
    tags.push(`<span class="card-tag" style="background:${bg}">${a.classification_status}</span>`);
  }
  const swatches = colors.slice(0, 5).map(c =>
    `<span class="color-swatch" style="background:rgb(${c[0]},${c[1]},${c[2]})" title="RGB(${c[0]},${c[1]},${c[2]})"></span>`
  ).join('');
  const img = a.thumbnail_url || a.image_url;
  return `
    <div class="card">
      <a href="/artifact/${a.id}">
        <img class="card-img" src="${img}" alt="${a.name}" onerror="this.style.display='none'" />
      </a>
      <div class="card-body">
        <div class="card-title">${a.name}</div>
        <div class="card-meta">${formatDim(a.width_cm, a.height_cm, a.area_cm2)}</div>
        ${tags.length ? `<div class="mt-1">${tags.join('')}</div>` : ''}
        ${swatches ? `<div class="color-swatches mt-1">${swatches}</div>` : ''}
      </div>
    </div>`;
}

function initAuth() {
  currentUser = window.SERVER_USER || null;
  if (currentUser) {
    localStorage.setItem('flintai_user', JSON.stringify(currentUser));
  }
}

async function logout() {
  try { await api.auth.logout(); } catch {}
  localStorage.removeItem('flintai_token');
  localStorage.removeItem('flintai_user');
  currentUser = null;
  window.location.href = '/login';
}

window.showAlert = showAlert;
window.getScoreClass = getScoreClass;
window.formatScore = formatScore;
window.formatDim = formatDim;
window.artifactCardHTML = artifactCardHTML;
window.currentUser = currentUser;
window.initAuth = initAuth;
window.logout = logout;

document.addEventListener('DOMContentLoaded', initAuth);
