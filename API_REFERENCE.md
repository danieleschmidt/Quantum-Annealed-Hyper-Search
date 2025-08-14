# Quantum Hyperparameter Search API Reference

## Overview

The Quantum Hyperparameter Search System provides a comprehensive RESTful API for quantum-enhanced optimization operations. This API supports both synchronous and asynchronous optimization requests with enterprise-grade security and monitoring.

## Base URL
```
https://api.quantum-optimizer.com/v1
```

## Authentication

All API requests require authentication using JWT Bearer tokens:

```http
Authorization: Bearer <your-jwt-token>
```

### Obtaining a Token
```http
POST /auth/login
Content-Type: application/json

{
  "user_id": "your_username",
  "password": "your_secure_password"
}
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_at": "2025-08-15T20:30:00Z",
  "permissions": ["quantum_operations", "read_data", "write_data"]
}
```

## Core Endpoints

### 1. Health Check

**GET /health**

Check system health and status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-08-14T20:30:00Z",
  "components": {
    "quantum_backend": "operational",
    "cache_system": "operational",
    "security_framework": "operational",
    "monitoring": "operational"
  },
  "metrics": {
    "uptime_seconds": 86400,
    "active_optimizations": 5,
    "cache_hit_ratio": 0.87,
    "quantum_advantage_score": 1.35
  }
}
```

### 2. Quantum Optimization

**POST /optimize**

Submit a quantum optimization request.

**Request:**
```json
{
  "objective_function": "maximize_accuracy",
  "parameter_space": {
    "learning_rate": [0.001, 0.01, 0.1],
    "batch_size": [16, 32, 64, 128],
    "hidden_layers": [1, 2, 3, 4],
    "dropout_rate": [0.0, 0.1, 0.2, 0.3]
  },
  "constraints": {
    "max_iterations": 100,
    "timeout_seconds": 300,
    "min_improvement": 0.01
  },
  "quantum_config": {
    "backend": "dwave",
    "num_reads": 1000,
    "annealing_time": 20.0,
    "chain_strength": 1.0
  },
  "optimization_options": {
    "enable_caching": true,
    "enable_quantum_advantage": true,
    "coherence_preservation": true,
    "adaptive_weighting": true
  }
}
```

**Response:**
```json
{
  "optimization_id": "opt_1692043800_abc123",
  "status": "completed",
  "execution_time": 45.2,
  "best_parameters": {
    "learning_rate": 0.01,
    "batch_size": 64,
    "hidden_layers": 2,
    "dropout_rate": 0.1
  },
  "best_score": 0.9548,
  "quantum_metrics": {
    "quantum_advantage_score": 1.42,
    "coherence_preserved": 0.89,
    "speedup_ratio": 2.1,
    "solution_quality_improvement": 0.12
  },
  "optimization_details": {
    "iterations_completed": 87,
    "convergence_achieved": true,
    "techniques_used": ["parallel_tempering", "quantum_walk"],
    "cache_hits": 23,
    "cache_misses": 64
  }
}
```

### 3. Optimization Status

**GET /optimize/{optimization_id}**

Get the status of a running or completed optimization.

**Response:**
```json
{
  "optimization_id": "opt_1692043800_abc123",
  "status": "running",
  "progress": 0.65,
  "estimated_completion": "2025-08-14T20:35:00Z",
  "current_best": {
    "parameters": {
      "learning_rate": 0.01,
      "batch_size": 32
    },
    "score": 0.9234
  },
  "metrics": {
    "iterations_completed": 65,
    "quantum_operations": 1250,
    "cache_utilization": 0.78
  }
}
```

### 4. Optimization History

**GET /optimize/history**

Retrieve optimization history for the authenticated user.

**Query Parameters:**
- `limit` (integer): Maximum number of results (default: 50, max: 500)
- `offset` (integer): Number of results to skip (default: 0)
- `status` (string): Filter by status (running, completed, failed)
- `since` (ISO timestamp): Only include optimizations since this time

**Response:**
```json
{
  "total_count": 142,
  "optimizations": [
    {
      "optimization_id": "opt_1692043800_abc123",
      "created_at": "2025-08-14T20:00:00Z",
      "status": "completed",
      "execution_time": 45.2,
      "best_score": 0.9548,
      "quantum_advantage_score": 1.42
    }
  ],
  "pagination": {
    "limit": 50,
    "offset": 0,
    "has_more": true
  }
}
```

## Advanced Features

### 5. Batch Optimization

**POST /optimize/batch**

Submit multiple optimization requests simultaneously.

**Request:**
```json
{
  "optimizations": [
    {
      "id": "batch_1",
      "objective_function": "maximize_accuracy",
      "parameter_space": {...}
    },
    {
      "id": "batch_2", 
      "objective_function": "minimize_loss",
      "parameter_space": {...}
    }
  ],
  "batch_options": {
    "parallel_execution": true,
    "priority": "high",
    "max_concurrent": 5
  }
}
```

### 6. Quantum Algorithm Selection

**POST /algorithms/recommend**

Get recommendations for optimal quantum algorithms based on problem characteristics.

**Request:**
```json
{
  "problem_characteristics": {
    "parameter_count": 50,
    "search_space_size": 10000,
    "objective_complexity": "non_convex",
    "constraints_present": true,
    "noise_tolerance": "medium"
  },
  "performance_requirements": {
    "max_execution_time": 300,
    "min_solution_quality": 0.9,
    "prefer_quantum_advantage": true
  }
}
```

**Response:**
```json
{
  "recommended_algorithms": [
    {
      "algorithm": "adaptive_parallel_tempering",
      "confidence": 0.92,
      "expected_quantum_advantage": 1.8,
      "estimated_execution_time": 180,
      "reasoning": "Optimal for non-convex optimization with medium noise tolerance"
    },
    {
      "algorithm": "quantum_walk_search",
      "confidence": 0.78,
      "expected_quantum_advantage": 1.4,
      "estimated_execution_time": 240,
      "reasoning": "Good exploration capabilities for large search spaces"
    }
  ]
}
```

### 7. Performance Analytics

**GET /analytics/performance**

Get detailed performance analytics and system metrics.

**Query Parameters:**
- `time_range` (string): "1h", "24h", "7d", "30d"
- `metric_type` (string): "quantum_advantage", "execution_time", "cache_performance"
- `aggregation` (string): "avg", "min", "max", "p95", "p99"

**Response:**
```json
{
  "time_range": "24h",
  "metrics": {
    "quantum_advantage": {
      "average": 1.35,
      "min": 0.98,
      "max": 2.41,
      "p95": 2.15,
      "trend": "increasing"
    },
    "execution_time": {
      "average": 42.5,
      "min": 5.2,
      "max": 180.3,
      "p95": 125.7,
      "trend": "stable"
    },
    "cache_performance": {
      "hit_ratio": 0.87,
      "avg_lookup_time": 0.0008,
      "memory_usage_mb": 1024.5,
      "evictions": 23
    }
  },
  "system_health": {
    "cpu_utilization": 0.34,
    "memory_utilization": 0.68,
    "active_optimizations": 12,
    "queue_length": 3
  }
}
```

## Security Endpoints

### 8. Token Management

**POST /auth/refresh**

Refresh an expiring token.

**Request:**
```json
{
  "refresh_token": "your_refresh_token"
}
```

**DELETE /auth/logout**

Revoke the current token.

**GET /auth/permissions**

Get current user permissions and security level.

**Response:**
```json
{
  "user_id": "quantum_user_123",
  "permissions": ["quantum_operations", "read_data", "write_data"],
  "security_level": "confidential",
  "token_expires_at": "2025-08-15T20:30:00Z",
  "session_id": "sess_abc123def456"
}
```

## Error Handling

All API endpoints return appropriate HTTP status codes and detailed error information:

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_PARAMETERS",
    "message": "Parameter 'learning_rate' values must be positive numbers",
    "details": {
      "field": "parameter_space.learning_rate",
      "provided_value": [-0.1, 0.01, 0.1],
      "expected_format": "Array of positive numbers"
    },
    "request_id": "req_1692043800_xyz789",
    "timestamp": "2025-08-14T20:30:00Z"
  }
}
```

### Common Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | `INVALID_PARAMETERS` | Request parameters are invalid or missing |
| 401 | `UNAUTHORIZED` | Authentication token is missing or invalid |
| 403 | `FORBIDDEN` | Insufficient permissions for the requested operation |
| 404 | `NOT_FOUND` | Requested resource (e.g., optimization_id) not found |
| 409 | `CONFLICT` | Resource conflict (e.g., duplicate optimization ID) |
| 429 | `RATE_LIMITED` | Too many requests, rate limit exceeded |
| 500 | `INTERNAL_ERROR` | Unexpected server error |
| 503 | `SERVICE_UNAVAILABLE` | System temporarily unavailable |

## Rate Limiting

API requests are subject to rate limiting based on user authentication level:

- **Standard Users**: 100 requests/hour
- **Premium Users**: 1,000 requests/hour  
- **Enterprise Users**: 10,000 requests/hour

Rate limit headers are included in all responses:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1692047400
```

## Webhooks

Configure webhooks to receive notifications about optimization completion:

**POST /webhooks**

Create a webhook endpoint.

**Request:**
```json
{
  "url": "https://your-app.com/webhook/quantum-optimization",
  "events": ["optimization.completed", "optimization.failed"],
  "secret": "your_webhook_secret"
}
```

**Webhook Payload Example:**
```json
{
  "event": "optimization.completed",
  "optimization_id": "opt_1692043800_abc123",
  "timestamp": "2025-08-14T20:30:00Z",
  "data": {
    "best_parameters": {...},
    "best_score": 0.9548,
    "quantum_advantage_score": 1.42
  }
}
```

## SDK Support

Official SDKs are available for multiple programming languages:

### Python SDK
```python
from quantum_hyper_search import QuantumOptimizer

client = QuantumOptimizer(api_key="your_api_key")
result = client.optimize(
    objective_function="maximize_accuracy",
    parameter_space={
        "learning_rate": [0.001, 0.01, 0.1],
        "batch_size": [16, 32, 64]
    }
)
print(f"Best score: {result.best_score}")
```

### JavaScript SDK
```javascript
import { QuantumOptimizer } from 'quantum-hyper-search';

const client = new QuantumOptimizer({ apiKey: 'your_api_key' });
const result = await client.optimize({
  objectiveFunction: 'maximize_accuracy',
  parameterSpace: {
    learningRate: [0.001, 0.01, 0.1],
    batchSize: [16, 32, 64]
  }
});
console.log(`Best score: ${result.bestScore}`);
```

## Support

For API support and questions:
- **Documentation**: [https://docs.quantum-optimizer.com](https://docs.quantum-optimizer.com)
- **Support Email**: api-support@quantum-optimizer.com
- **Status Page**: [https://status.quantum-optimizer.com](https://status.quantum-optimizer.com)
- **Community Forum**: [https://community.quantum-optimizer.com](https://community.quantum-optimizer.com)

---

**API Version**: v1.0  
**Last Updated**: 2025-08-14  
**Status**: Production Ready ✅