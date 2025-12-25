# Q-Route Alpha: Technology Stack Documentation

## Frontend, Backend & Infrastructure

**Version:** 1.0.0  
**Organization:** Quantum Gandiva AI  
**Purpose:** Detailed documentation of all technologies used and why

---

## 1. Technology Overview

Q-Route Alpha is a full-stack application with three main layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                         PRESENTATION                             │
│            React 19 + Vite + Tailwind CSS + Leaflet             │
├─────────────────────────────────────────────────────────────────┤
│                            API                                   │
│              FastAPI + Pydantic + CORS Middleware               │
├─────────────────────────────────────────────────────────────────┤
│                          CORE                                    │
│          Python + D-Wave Samplers + NumPy + dimod               │
├─────────────────────────────────────────────────────────────────┤
│                      INFRASTRUCTURE                              │
│           Docker + nginx + GitHub Actions + Cloud               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Frontend Technologies

### 2.1 React 19

**What it is:** A JavaScript library for building user interfaces

**Why we use it:**
- **Component-Based**: Reusable UI components (RouteMap, ProblemForm, ResultsPanel)
- **Virtual DOM**: Efficient updates when route solutions change
- **Hooks API**: Clean state management with useState, useEffect
- **Large Ecosystem**: Extensive libraries and community support

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `App.jsx` | Main application, state management |
| `Header.jsx` | Navigation bar, connection status |
| `RouteMap.jsx` | Interactive map visualization |
| `ProblemForm.jsx` | Customer input form |
| `ResultsPanel.jsx` | Solution display |

**Example:**
```jsx
// State management with React hooks
const [solution, setSolution] = useState(null);

// API call on form submit
const handleSubmit = async (problemData) => {
  const result = await solveProblem(problemData);
  setSolution(result);
};
```

### 2.2 Vite 6

**What it is:** Next-generation frontend build tool

**Why we use it:**
- **Lightning Fast HMR**: Changes reflect instantly during development
- **Native ES Modules**: No bundling during development
- **Optimized Builds**: Tree-shaking and code-splitting for production
- **Simple Configuration**: Minimal setup required

**Why NOT Webpack/Create React App:**
- CRA is deprecated
- Webpack is slower and more complex
- Vite is the modern standard

**Configuration (`vite.config.js`):**
```javascript
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
})
```

### 2.3 Tailwind CSS 4

**What it is:** Utility-first CSS framework

**Why we use it:**
- **Rapid Prototyping**: Style directly in JSX, no CSS file switching
- **Consistent Design**: Predefined spacing, colors, typography scales
- **No Dead CSS**: Only includes classes you actually use
- **Responsive by Default**: Mobile-first with easy breakpoints

**Why NOT Bootstrap/Material UI:**
- Bootstrap: Too opinionated, harder to customize
- Material UI: Large bundle size, Google-specific aesthetic
- Tailwind: Maximum flexibility with minimal overhead

**Example:**
```jsx
<div className="bg-white rounded-lg border border-[#D1D1E0] p-4">
  <h2 className="text-lg font-semibold text-[#1A1A2E] mb-4">
    Route Visualization
  </h2>
</div>
```

### 2.4 Leaflet

**What it is:** Open-source JavaScript library for interactive maps

**Why we use it:**
- **Lightweight**: ~42KB gzipped (vs Google Maps SDK ~100KB+)
- **Free**: No API key required for basic usage
- **Mobile-Friendly**: Touch and gesture support built-in
- **Extensible**: Plugin ecosystem for advanced features

**Why NOT Google Maps/Mapbox:**
- Google Maps: Requires API key, usage-based pricing
- Mapbox: More complex setup, overkill for our needs
- Leaflet: Free, simple, sufficient for route visualization

**Features Used:**
- Marker placement for depot/customers
- Polyline for route path
- Popup for node details
- Custom icons for depot vs customers

---

## 3. Backend Technologies

### 3.1 Python 3.11+

**What it is:** High-level programming language

**Why we use it:**
- **D-Wave SDK**: Native Python support for quantum/classical optimization
- **NumPy Ecosystem**: Efficient numerical operations
- **Type Hints**: Better code quality with static analysis
- **Rapid Development**: Clean syntax, extensive libraries

**Key Features Used:**
- Type annotations for function signatures
- Dataclasses for problem/solution models
- f-strings for formatted output
- Walrus operator for concise expressions

### 3.2 FastAPI

**What it is:** Modern, fast web framework for building APIs

**Why we use it:**
- **Automatic OpenAPI Docs**: `/docs` endpoint with Swagger UI
- **Type Validation**: Pydantic models for request/response validation
- **Async Support**: Non-blocking I/O for concurrent requests
- **High Performance**: One of the fastest Python frameworks

**Why NOT Flask/Django:**
- Flask: No built-in validation, manual schema definitions
- Django: Too heavyweight, ORM overkill for API-only service
- FastAPI: Perfect balance of features and simplicity

**Endpoint Example:**
```python
@app.post("/solve", response_model=SolveResponse)
async def solve_cvrp(request: SolveRequest):
    """Solve CVRP problem with simulated annealing."""
    problem = CVRPProblem(
        depot=(request.depot.x, request.depot.y),
        customers=request.customers,
        vehicle_capacity=request.vehicle_capacity
    )
    solution = solver.solve(problem)
    return SolveResponse(...)
```

### 3.3 Pydantic

**What it is:** Data validation library using Python type annotations

**Why we use it:**
- **Automatic Validation**: Type checking at runtime
- **Clear Error Messages**: Helpful validation errors
- **JSON Schema**: Automatic API documentation
- **FastAPI Integration**: Native support

**Models Example:**
```python
class SolveRequest(BaseModel):
    depot: DepotInput
    customers: List[CustomerInput] = Field(..., min_length=1)
    vehicle_capacity: int = Field(..., gt=0)
    num_reads: int = Field(default=1000, ge=100, le=10000)
```

### 3.4 D-Wave Ocean SDK

**What it is:** Toolkit for quantum and quantum-inspired optimization

**Components Used:**

| Package | Purpose | Why |
|---------|---------|-----|
| `dwave-samplers` | Simulated Annealing | Local solver, no API key |
| `dimod` | BQM/QUBO construction | Standard representation |
| `neal` | Legacy SA (wrapper) | Backward compatibility |

**Why D-Wave (not IBM Qiskit):**
- D-Wave: Native optimization focus, QUBO/Ising native
- IBM Qiskit: Gate-based quantum, requires circuit compilation
- For optimization problems, annealing is more suitable

**Quantum Migration Path:**
```python
# Current: Local ($0)
from dwave.samplers import SimulatedAnnealingSampler

# Future: Cloud Hybrid
from dwave.system import LeapHybridSampler

# Future: Direct QPU
from dwave.system import DWaveSampler, EmbeddingComposite
```

### 3.5 NumPy

**What it is:** Fundamental package for numerical computing

**Why we use it:**
- **Distance Matrix**: Efficient pairwise distance calculation
- **QUBO Matrix**: Sparse matrix operations
- **Vectorization**: Much faster than Python loops

**Example:**
```python
def compute_distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i,j] = np.sqrt((coords[i][0]-coords[j][0])**2 + 
                            (coords[i][1]-coords[j][1])**2)
    return D
```

---

## 4. Infrastructure Technologies

### 4.1 Docker

**What it is:** Container platform for packaging applications

**Why we use it:**
- **Environment Consistency**: Same behavior dev → prod
- **Easy Deployment**: Single command deployment
- **Isolation**: No dependency conflicts
- **Scalability**: Easy horizontal scaling

**Containers:**

| Container | Base Image | Purpose |
|-----------|------------|---------|
| `backend` | python:3.11-slim | FastAPI + solver |
| `frontend` | node:20 → nginx | React build + static serving |

**Backend Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4.2 Docker Compose

**What it is:** Multi-container orchestration tool

**Why we use it:**
- **Single Command**: `docker-compose up` starts everything
- **Service Networking**: Containers communicate by name
- **Environment Variables**: Centralized configuration
- **Volume Mounting**: Development hot-reload

**Configuration:**
```yaml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    environment:
      - CORS_ORIGINS=http://localhost:3000
  
  frontend:
    build: ./frontend
    ports: ["3000:80"]
    depends_on: [backend]
```

### 4.3 nginx

**What it is:** High-performance web server and reverse proxy

**Why we use it:**
- **Static File Serving**: Optimized for React build files
- **Gzip Compression**: Smaller file transfers
- **API Proxying**: Route `/api` to backend
- **Production Ready**: Battle-tested at scale

**Configuration (`nginx.conf`):**
```nginx
server {
    listen 80;
    
    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }
    
    location /api {
        proxy_pass http://backend:8000;
    }
}
```

### 4.4 GitHub Actions

**What it is:** CI/CD automation platform

**Why we use it:**
- **Free for Public Repos**: No additional cost
- **GitHub Integration**: Native to our repo host
- **Extensive Marketplace**: Pre-built actions
- **Matrix Testing**: Test across multiple Python/Node versions

**CI Pipeline (`.github/workflows/ci.yml`):**

```yaml
jobs:
  test-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -r backend/requirements.txt
      - run: pytest backend/test_main.py

  test-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: npm ci
      - run: npm test
```

---

## 5. Development Tools

### 5.1 Testing

| Tool | Domain | Tests |
|------|--------|-------|
| pytest | Backend/Core | 27 tests |
| Vitest | Frontend | 51 tests |

### 5.2 Code Quality

| Tool | Purpose |
|------|---------|
| Black | Python code formatting |
| Ruff | Python linting |
| ESLint | JavaScript linting |
| mypy | Python type checking |

### 5.3 IDE Support

- **VS Code**: Primary development environment
- **Extensions**: Python, ESLint, Tailwind IntelliSense

---

## 6. Cloud Deployment Options

### 6.1 Supported Platforms

| Platform | Configuration | Pros |
|----------|---------------|------|
| Railway | `railway.json` | Easy, good free tier |
| Render | `render.yaml` | Auto-deploy from GitHub |
| Fly.io | `fly.toml` | Edge deployment |
| AWS/GCP | Docker | Full control |

### 6.2 Environment Variables

```bash
# Backend
CORS_ORIGINS=https://your-frontend.com
DEFAULT_NUM_READS=1000
MAX_CUSTOMERS=50

# Frontend
VITE_API_URL=https://your-backend.com
```

---

## 7. Technology Decision Matrix

| Requirement | Chosen Technology | Alternatives Considered | Decision Rationale |
|-------------|-------------------|------------------------|-------------------|
| UI Framework | React | Vue, Svelte, Angular | Largest ecosystem, team familiarity |
| Build Tool | Vite | Webpack, Parcel | Speed, modern defaults |
| CSS | Tailwind | CSS Modules, Styled-components | Rapid development, consistency |
| Maps | Leaflet | Google Maps, Mapbox | Free, lightweight, sufficient |
| API Framework | FastAPI | Flask, Django | Performance, validation, docs |
| Optimizer | D-Wave Samplers | OR-Tools, Gurobi | Quantum-ready, free |
| Containers | Docker | Podman, LXD | Industry standard |
| CI/CD | GitHub Actions | Jenkins, CircleCI | Native GitHub integration |

---

## 8. Performance Characteristics

### 8.1 Frontend

| Metric | Value |
|--------|-------|
| Initial Load | < 500ms |
| Bundle Size | ~180KB gzipped |
| Lighthouse Score | 95+ |

### 8.2 Backend

| Metric | Value |
|--------|-------|
| Cold Start | < 1s |
| 5-node Solve | < 1s |
| 10-node Solve | 2-5s |
| Max Concurrent | 100+ requests |

### 8.3 Infrastructure

| Metric | Value |
|--------|-------|
| Container Build | ~2 min |
| Memory (Backend) | ~256MB |
| Memory (Frontend) | ~64MB |
| CI Pipeline | ~3 min |

---

*Technology Stack Documentation - Quantum Gandiva AI - December 2025*
