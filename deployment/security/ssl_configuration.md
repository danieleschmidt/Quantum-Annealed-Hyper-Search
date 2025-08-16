# SSL/TLS Configuration

## Generate SSL Certificate
```bash
# Self-signed certificate for development
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout quantum.key \
  -out quantum.crt \
  -subj "/C=US/ST=CA/L=SF/O=QuantumLabs/CN=quantum-optimizer.local"
```

## Nginx SSL Configuration
```nginx
server {
    listen 443 ssl http2;
    server_name quantum-optimizer.com;
    
    ssl_certificate /etc/nginx/ssl/quantum.crt;
    ssl_certificate_key /etc/nginx/ssl/quantum.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://quantum-optimizer:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
