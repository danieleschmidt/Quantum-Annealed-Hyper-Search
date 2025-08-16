# Quantum Hyper Search API Documentation

## Base URL
```
https://api.quantum-optimizer.com
```

## Authentication
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     https://api.quantum-optimizer.com/v1/optimize
```

## Endpoints

### Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-08-14T20:30:00Z"
}
```

### Optimize Parameters
```
POST /v1/optimize
```

**Request:**
```json
{
  "objective_function": "maximize_accuracy",
  "parameter_space": {
    "learning_rate": [0.001, 0.01, 0.1],
    "batch_size": [16, 32, 64, 128]
  },
  "constraints": {
    "max_iterations": 100,
    "timeout": 300
  },
  "quantum_backend": "dwave"
}
```

**Response:**
```json
{
  "optimization_id": "opt_123456",
  "status": "completed",
  "best_parameters": {
    "learning_rate": 0.01,
    "batch_size": 64
  },
  "best_score": 0.95,
  "quantum_advantage": 1.3,
  "execution_time": 45.2
}
```

### Get Optimization Status
```
GET /v1/optimize/{optimization_id}
```

**Response:**
```json
{
  "optimization_id": "opt_123456",
  "status": "running",
  "progress": 0.65,
  "estimated_completion": "2025-08-14T20:35:00Z"
}
```

## Rate Limits
- **Standard**: 100 requests/hour
- **Premium**: 1000 requests/hour
- **Enterprise**: Unlimited

## Error Codes
- **400**: Bad Request - Invalid parameters
- **401**: Unauthorized - Invalid or missing token
- **429**: Too Many Requests - Rate limit exceeded
- **500**: Internal Server Error - System error

## SDKs
- Python: `pip install quantum-hyper-search-client`
- JavaScript: `npm install quantum-hyper-search-client`
- R: `install.packages("quantumHyperSearch")`
