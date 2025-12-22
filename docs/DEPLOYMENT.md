# Q-Route Alpha Deployment Guide

This guide covers deploying Q-Route Alpha to various cloud platforms.

## Prerequisites

- Docker and Docker Compose installed locally
- Git repository pushed to GitHub
- Account on your chosen cloud platform

## Local Docker Deployment

```bash
# Build and run both services
docker-compose up --build

# Access:
# - Frontend: http://localhost:3000
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

## Cloud Deployment Options

### Option 1: Railway (Recommended)

Railway offers the simplest deployment with automatic Docker detection.

#### Steps:

1. **Create Railway Account**: https://railway.app

2. **Install Railway CLI**:
   ```bash
   npm install -g @railway/cli
   railway login
   ```

3. **Create New Project**:
   ```bash
   cd QOptimiser
   railway init
   ```

4. **Deploy Backend**:
   ```bash
   # Create backend service
   railway add
   # Select "Empty Service"
   # Name it "backend"

   # Configure and deploy
   railway link
   railway up
   ```

5. **Deploy Frontend**:
   ```bash
   cd frontend
   railway add
   # Select "Empty Service"
   # Name it "frontend"
   railway up
   ```

6. **Set Environment Variables**:
   - In Railway dashboard, set `CORS_ORIGINS` to your frontend URL
   - Set `VITE_API_URL` in frontend to `/api` or backend URL

#### Railway with Docker Compose:
```bash
railway up --service backend
railway up --service frontend
```

---

### Option 2: Render

Render provides free tier hosting with automatic deploys from GitHub.

#### Steps:

1. **Create Render Account**: https://render.com

2. **Deploy from Blueprint**:
   - Go to Dashboard → New → Blueprint
   - Connect your GitHub repository
   - Render will detect `render.yaml` and create services

3. **Manual Deployment**:

   **Backend Service**:
   - New → Web Service
   - Connect GitHub repo
   - Environment: Docker
   - Dockerfile Path: `./backend/Dockerfile`
   - Docker Context: `.`

   **Frontend Service**:
   - New → Web Service
   - Connect GitHub repo
   - Environment: Docker
   - Dockerfile Path: `./frontend/Dockerfile`
   - Docker Context: `./frontend`

4. **Environment Variables**:
   ```
   # Backend
   APP_VERSION=0.1.0
   CORS_ORIGINS=https://your-frontend.onrender.com

   # Frontend
   VITE_API_URL=/api
   ```

---

### Option 3: Fly.io

Fly.io offers edge deployment with generous free tier.

#### Steps:

1. **Install Fly CLI**:
   ```bash
   # macOS
   brew install flyctl

   # Windows
   powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"

   # Linux
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login and Launch**:
   ```bash
   fly auth login
   fly launch
   ```

3. **Deploy**:
   ```bash
   fly deploy
   ```

4. **Scale** (optional):
   ```bash
   fly scale count 2  # Run 2 instances
   fly scale memory 1024  # Increase memory
   ```

---

### Option 4: DigitalOcean App Platform

#### Steps:

1. **Create DigitalOcean Account**: https://digitalocean.com

2. **Create App**:
   - Go to Apps → Create App
   - Select GitHub repository
   - Configure as Docker app

3. **Add Services**:
   - Backend: Dockerfile path `backend/Dockerfile`
   - Frontend: Dockerfile path `frontend/Dockerfile`

4. **Environment Variables**:
   - Configure in App Settings

---

### Option 5: AWS (ECS/Fargate)

For production-scale deployment:

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

docker build -t q-route-backend -f backend/Dockerfile .
docker tag q-route-backend:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/q-route-backend:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/q-route-backend:latest

# Deploy with ECS (use AWS Console or CDK)
```

---

## Environment Variables Reference

### Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_VERSION` | `0.1.0` | Application version |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Allowed CORS origins (comma-separated) |
| `DEFAULT_NUM_READS` | `1000` | Default simulated annealing reads |
| `DEFAULT_NUM_SWEEPS` | `1000` | Default simulated annealing sweeps |
| `MAX_CUSTOMERS` | `50` | Maximum customers per problem |

### Frontend

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_URL` | `http://localhost:8000` | Backend API URL |
| `VITE_APP_VERSION` | `0.1.0` | App version for display |

---

## Health Checks

Both services expose health check endpoints:

- **Backend**: `GET /health` → `{"status": "healthy", "version": "0.1.0"}`
- **Frontend**: `GET /health` → `OK`

---

## Troubleshooting

### CORS Errors
Ensure `CORS_ORIGINS` includes your frontend URL.

### 502 Bad Gateway
- Check backend health: `curl https://your-backend/health`
- Verify nginx proxy configuration

### Slow Optimization
- Increase `num_reads` for better results (slower)
- Check available memory (minimum 512MB recommended)

### Docker Build Fails
- Ensure all files are committed to git
- Check `.dockerignore` isn't excluding required files

---

## Monitoring

### Logs

```bash
# Railway
railway logs

# Fly.io
fly logs

# Docker Compose
docker-compose logs -f
```

### Metrics

The backend exposes OpenAPI docs at `/docs` for API exploration.

---

## Scaling Recommendations

| Users | Backend Instances | Memory |
|-------|-------------------|--------|
| < 100 | 1 | 512MB |
| 100-500 | 2 | 1GB |
| 500+ | 3+ | 2GB+ |

For high-traffic production deployments, consider:
- Redis for caching
- PostgreSQL for problem/solution storage
- Load balancer with session affinity
