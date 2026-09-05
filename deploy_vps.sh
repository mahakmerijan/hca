#!/bin/bash
# Deployment script for hca_v2 on Contabo VPS (169.58.250.61)
set -e

echo "=== HCA v2 Deployment ==="

# 1. System update & deps
apt-get update -y
apt-get install -y python3.11 python3.11-venv python3-pip git curl nginx supervisor libsndfile1 ffmpeg libgl1-mesa-glx libglib2.0-0

# 2. Clone or pull repo
if [ -d /opt/hca ]; then
  echo "Pulling latest..."
  cd /opt/hca && git pull origin main
else
  echo "Cloning repo..."
  git clone https://github.com/mahakmerijan/hca.git /opt/hca
  cd /opt/hca
fi

cd /opt/hca

# 3. Python virtual environment
if [ ! -d venv ]; then
  python3.11 -m venv venv
fi
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

# 4. Create .env if not present
if [ ! -f .env ]; then
cat > .env << 'EOF'
VERTEX_PROJECT=ai-ml-integrations
VERTEX_LOCATION=us-central1
LLM_MODEL=gemini-3.5-flash
JWT_SECRET=change-me-in-production-$(openssl rand -hex 16)
PORT=5004
EOF
  echo "Created .env — edit /opt/hca/.env to add your GOOGLE_API_KEY or GCP credentials"
fi

# 5. Create output dirs
install -d -m 0750 output uploads logs /var/log/hca
chown -R root:root logs /var/log/hca

# Keep detailed per-run telemetry separate from Supervisor output and bounded on disk.
cat > /etc/logrotate.d/hca-telemetry << 'EOF'
/opt/hca/logs/token_usage.jsonl {
  daily
  rotate 30
  maxsize 100M
  compress
  missingok
  notifempty
  copytruncate
  create 0640 root root
}
EOF

# 6. Supervisor config to keep Flask running
cat > /etc/supervisor/conf.d/hca.conf << 'EOF'
[program:hca]
command=/opt/hca/venv/bin/python /opt/hca/app.py
directory=/opt/hca
user=root
autostart=true
autorestart=true
stderr_logfile=/var/log/hca.err.log
stdout_logfile=/var/log/hca.out.log
environment=PYTHONUNBUFFERED="1",HCA_TELEMETRY_LOG_DIR="/opt/hca/logs"
EOF

# 7. Nginx reverse proxy (port 80 → Flask 5004)
cat > /etc/nginx/sites-available/hca << 'EOF'
server {
    listen 80;
    server_name _;
    client_max_body_size 600M;

    location / {
        proxy_pass http://127.0.0.1:5004;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300;
        proxy_send_timeout 300;
    }
}
EOF

ln -sf /etc/nginx/sites-available/hca /etc/nginx/sites-enabled/hca
rm -f /etc/nginx/sites-enabled/default

# 8. Start services
supervisorctl reread && supervisorctl update && supervisorctl restart hca || supervisorctl start hca
nginx -t && systemctl restart nginx

echo ""
echo "✅ Deployment complete!"
echo "   App running at: http://169.58.250.61"
echo "   Logs: tail -f /var/log/hca.out.log"
echo "   Run telemetry: tail -f /opt/hca/logs/token_usage.jsonl"
echo "   Edit credentials: nano /opt/hca/.env"
