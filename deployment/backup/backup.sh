#!/bin/bash
# Backup Script for Quantum Hyper Search

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/quantum-hyper-search"
RETENTION_DAYS=30

echo "🗄️ Starting backup process..."

# Create backup directory
mkdir -p ${BACKUP_DIR}/${TIMESTAMP}

# Backup application data
echo "📁 Backing up application data..."
if [ -d "/app/data" ]; then
    cp -r /app/data ${BACKUP_DIR}/${TIMESTAMP}/data
fi

# Backup configuration
echo "⚙️ Backing up configuration..."
cp -r deployment/production ${BACKUP_DIR}/${TIMESTAMP}/config

# Backup logs (last 7 days)
echo "📝 Backing up logs..."
if [ -d "/app/logs" ]; then
    find /app/logs -name "*.log" -mtime -7 -exec cp {} ${BACKUP_DIR}/${TIMESTAMP}/logs/ \;
fi

# Create manifest
echo "📋 Creating backup manifest..."
cat > ${BACKUP_DIR}/${TIMESTAMP}/manifest.json << EOF
{
  "backup_id": "${TIMESTAMP}",
  "created_at": "$(date -Iseconds)",
  "components": ["data", "config", "logs"],
  "version": "1.0.0",
  "retention_until": "$(date -d '+${RETENTION_DAYS} days' -Iseconds)"
}
EOF

# Compress backup
echo "🗜️ Compressing backup..."
cd ${BACKUP_DIR}
tar -czf quantum_backup_${TIMESTAMP}.tar.gz ${TIMESTAMP}/
rm -rf ${TIMESTAMP}/

# Clean old backups
echo "🧹 Cleaning old backups..."
find ${BACKUP_DIR} -name "quantum_backup_*.tar.gz" -mtime +${RETENTION_DAYS} -delete

echo "✅ Backup completed: quantum_backup_${TIMESTAMP}.tar.gz"
