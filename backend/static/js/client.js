const API_BASE = '/api';

function getToken() {
  return localStorage.getItem('flintai_token');
}

async function apiRequest(path, options = {}) {
  const token = getToken();
  const headers = { 'Content-Type': 'application/json', ...options.headers };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  let data;
  try { data = await res.json(); } catch { data = { error: 'Invalid response' }; }

  if (!res.ok) {
    if (res.status === 401) {
      localStorage.removeItem('flintai_token');
      localStorage.removeItem('flintai_user');
      window.location.href = '/login';
      throw new Error('Unauthorized');
    }
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

const api = {
  auth: {
    register: (body) => apiRequest('/auth/register', { method: 'POST', body: JSON.stringify(body) }),
    login: (body) => apiRequest('/auth/login', { method: 'POST', body: JSON.stringify(body) }),
    logout: () => apiRequest('/auth/logout', { method: 'POST' }),
  },
  users: {
    me: () => apiRequest('/users/me'),
    update: (body) => apiRequest('/users/me', { method: 'PUT', body: JSON.stringify(body) }),
  },
  artifacts: {
    list: (params = {}) => {
      const qs = new URLSearchParams(params).toString();
      return apiRequest(`/artifacts${qs ? '?' + qs : ''}`);
    },
    mine: () => apiRequest('/artifacts/mine'),
    get: (id, params = {}) => {
      const qs = new URLSearchParams(params).toString();
      return apiRequest(`/artifacts/${id}${qs ? '?' + qs : ''}`);
    },
    upload: (formData) => {
      const token = getToken();
      return fetch(`${API_BASE}/artifacts`, {
        method: 'POST',
        headers: token ? { 'Authorization': `Bearer ${token}` } : {},
        body: formData,
      }).then(async r => {
        const text = await r.text();
        try { return JSON.parse(text); } catch { return { error: text }; }
      }).catch(err => ({ error: err.message }));
    },
    update: (id, body) => apiRequest(`/artifacts/${id}`, { method: 'PUT', body: JSON.stringify(body) }),
    delete: (id) => apiRequest(`/artifacts/${id}`, { method: 'DELETE' }),
    match: (id, limit = 10) => apiRequest(`/artifacts/${id}/match?limit=${limit}`),
    compareBatch: (ids) => apiRequest('/artifacts/compare-batch', { method: 'POST', body: JSON.stringify({ artifact_ids: ids }) }),
  },
  collections: {
    list: (params = {}) => {
      const qs = new URLSearchParams(params).toString();
      return apiRequest(`/collections${qs ? '?' + qs : ''}`);
    },
    create: (body) => apiRequest('/collections', { method: 'POST', body: JSON.stringify(body) }),
    get: (id) => apiRequest(`/collections/${id}`),
    update: (id, body) => apiRequest(`/collections/${id}`, { method: 'PUT', body: JSON.stringify(body) }),
    delete: (id) => apiRequest(`/collections/${id}`, { method: 'DELETE' }),
    addArtifact: (cid, aid) => apiRequest(`/collections/${cid}/artifacts/${aid}`, { method: 'POST' }),
    removeArtifact: (cid, aid) => apiRequest(`/collections/${cid}/artifacts/${aid}`, { method: 'DELETE' }),
  },
  reconstruct: (ids) => apiRequest('/reconstruct', { method: 'POST', body: JSON.stringify({ artifact_ids: ids }) }),
};

window.api = api;
