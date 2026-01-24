# Deployment Guide - Futures Price Prediction System

## Table of Contents
1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Monitoring & Logging](#monitoring--logging)
5. [Production Checklist](#production-checklist)

---

## Local Development

### Prerequisites
- Python 3.10+
- 8GB RAM minimum (16GB recommended)
- GPU optional but recommended for LSTM/CNN training

### Setup
```bash
# Clone and setup
cd Futures_Prediction_Arbitrage_ML
bash setup.sh

# Activate environment
source venv/bin/activate

# Run training
python train.py

# Start API
python api.py
```

### Testing
```bash
# Unit tests
pytest tests/ -v

# Load testing
pip install locust
locust -f tests/load_test.py --host=http://localhost:8000
```

---

## Docker Deployment

### Single Container

Build and run:
```bash
docker build -t futures-prediction:v2.0 .
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --name futures-api \
  futures-prediction:v2.0
```

Check logs:
```bash
docker logs -f futures-api
```

### Docker Compose (Recommended)

Full stack with MLflow:
```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Docker Compose

For production, create `docker-compose.prod.yml`:
```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "80:8000"
    volumes:
      - ./models:/app/models:ro  # Read-only
      - ./logs:/app/logs
      - ./config.yaml:/app/config.yaml:ro
    environment:
      - WORKERS=4
      - LOG_LEVEL=WARNING
    restart: always
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
      replicas: 2

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: always
```

Run:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

---

## Cloud Deployment

### AWS EC2

1. **Launch Instance**
```bash
# EC2 instance: t3.xlarge (4 vCPU, 16GB RAM)
# AMI: Ubuntu 22.04 LTS
# Security Group: Allow ports 22, 80, 443, 8000
```

2. **Setup**
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Clone repo
git clone https://github.com/yourusername/futures-prediction.git
cd futures-prediction

# Deploy
docker-compose -f docker-compose.prod.yml up -d
```

3. **SSL with Let's Encrypt**
```bash
sudo apt install certbot
sudo certbot --nginx -d yourdomain.com
```

### AWS ECS (Elastic Container Service)

1. **Build and push to ECR**
```bash
# Authenticate
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build
docker build -t futures-prediction:v2.0 .

# Tag and push
docker tag futures-prediction:v2.0 <account-id>.dkr.ecr.us-east-1.amazonaws.com/futures-prediction:v2.0
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/futures-prediction:v2.0
```

2. **Create ECS Task Definition**
```json
{
  "family": "futures-prediction",
  "networkMode": "awsvpc",
  "containerDefinitions": [{
    "name": "api",
    "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/futures-prediction:v2.0",
    "cpu": 2048,
    "memory": 4096,
    "portMappings": [{
      "containerPort": 8000,
      "protocol": "tcp"
    }],
    "environment": [
      {"name": "WORKERS", "value": "4"}
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/futures-prediction",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "api"
      }
    }
  }]
}
```

3. **Create Service**
```bash
aws ecs create-service \
  --cluster your-cluster \
  --service-name futures-prediction-service \
  --task-definition futures-prediction \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx]}"
```

### Google Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/your-project/futures-prediction

# Deploy
gcloud run deploy futures-prediction \
  --image gcr.io/your-project/futures-prediction \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10 \
  --allow-unauthenticated
```

### Azure Container Instances

```bash
# Create resource group
az group create --name futures-prediction-rg --location eastus

# Deploy container
az container create \
  --resource-group futures-prediction-rg \
  --name futures-prediction \
  --image yourdockerhub/futures-prediction:v2.0 \
  --cpu 4 \
  --memory 8 \
  --dns-name-label futures-prediction \
  --ports 8000
```

---

## Monitoring & Logging

### Prometheus + Grafana

Add to `docker-compose.yml`:
```yaml
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
    depends_on:
      - prometheus
```

### Application Monitoring

Add to `api.py`:
```python
from prometheus_client import Counter, Histogram, make_asgi_app
from starlette.middleware.wsgi import WSGIMiddleware

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

# Mount metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", WSGIMiddleware(metrics_app))
```

### Centralized Logging (ELK Stack)

```yaml
  elasticsearch:
    image: elasticsearch:8.10.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"

  logstash:
    image: logstash:8.10.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: kibana:8.10.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

---

## Production Checklist

### Security
- [ ] Use HTTPS (TLS/SSL certificates)
- [ ] Implement API authentication (JWT, OAuth2)
- [ ] Rate limiting (e.g., using Redis)
- [ ] Input validation and sanitization
- [ ] Secure model storage (encrypted volumes)
- [ ] Regular security audits

### Performance
- [ ] Load balancing (Nginx, AWS ALB)
- [ ] Caching (Redis for predictions)
- [ ] Database for persistence (PostgreSQL)
- [ ] CDN for static assets
- [ ] Async prediction queue (Celery + RabbitMQ)

### Reliability
- [ ] Health checks configured
- [ ] Auto-scaling policies
- [ ] Circuit breakers
- [ ] Graceful shutdown
- [ ] Backup and disaster recovery

### Monitoring
- [ ] Application metrics (Prometheus)
- [ ] Logging aggregation (ELK)
- [ ] Alerting (PagerDuty, Opsgenie)
- [ ] APM (Datadog, New Relic)
- [ ] Model performance monitoring

### MLOps
- [ ] CI/CD pipeline (GitHub Actions, GitLab CI)
- [ ] Automated testing
- [ ] Model versioning (MLflow, DVC)
- [ ] A/B testing framework
- [ ] Model retraining pipeline
- [ ] Data drift detection

### Compliance
- [ ] Data privacy (GDPR, CCPA)
- [ ] Audit logs
- [ ] Model explainability
- [ ] Financial regulations (if applicable)
- [ ] Documentation

---

## Environment Variables

Create `.env` file:
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model Configuration
MODEL_DIR=./models
DEFAULT_MODEL=xgb_regressor

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/api.log

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# Security (for production)
API_KEY=your-secret-api-key
JWT_SECRET=your-jwt-secret
ALLOWED_ORIGINS=https://yourdomain.com

# Database (optional)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=futures_prediction
POSTGRES_USER=user
POSTGRES_PASSWORD=password

# Redis (optional)
REDIS_HOST=redis
REDIS_PORT=6379
```

---

## Troubleshooting

### Common Issues

1. **Out of Memory**
```bash
# Increase Docker memory limit
docker run --memory="8g" ...

# Or use memory-efficient models
# Set in config.yaml: use_smote: false
```

2. **Slow Predictions**
```bash
# Use ONNX for faster inference
pip install onnxruntime
# Convert model: skl2onnx, tf2onnx
```

3. **Port Already in Use**
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9

# Or use different port
python api.py --port 8001
```

4. **GPU Not Detected**
```bash
# Check CUDA installation
nvidia-smi

# Use CPU-only TensorFlow
pip uninstall tensorflow
pip install tensorflow-cpu
```

---

## Performance Tuning

### Model Optimization
- Use ONNX Runtime for 2-5x faster inference
- Quantize models (INT8 instead of FP32)
- Prune unnecessary features
- Batch predictions when possible

### API Optimization
- Use Gunicorn workers: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app`
- Enable response caching
- Implement request batching
- Use async endpoints

### Database Optimization
- Index frequently queried fields
- Use connection pooling
- Implement read replicas
- Cache frequent queries (Redis)

---

**For questions or issues, please refer to README_v2.md or open an issue.**
