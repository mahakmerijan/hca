#!/bin/bash
# Deployment script for hca_v2 on Contabo VPS (169.58.250.61)
set -e

PUBLIC_IP="${PUBLIC_IP:-169.58.250.61}"
# Browsers expose camera/mic (getUserMedia) only on secure origins, and Let's Encrypt
# will not issue for a bare IP — sslip.io resolves <dashed-ip>.sslip.io to that IP, so it
# gives us a certifiable hostname with no DNS registrar. Override DOMAIN for a real domain.
DOMAIN="${DOMAIN:-${PUBLIC_IP//./-}.sslip.io}"
CERTBOT_EMAIL="${CERTBOT_EMAIL:-}"

echo "=== HCA v2 Deployment ==="
echo "    Domain: $DOMAIN"

# 1. System update & deps
apt-get update -y
apt-get install -y python3.11 python3.11-venv python3-pip git curl nginx supervisor libsndfile1 ffmpeg libgl1-mesa-glx libglib2.0-0 certbot python3-certbot-nginx

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

# 7. Nginx reverse proxy (port 80 → Flask 5004). Certbot rewrites this file in step 8
#    to add the TLS listener and the 80 → 443 redirect.
cat > /etc/nginx/sites-available/hca << EOF
server {
    listen 80;
    server_name $DOMAIN;
    client_max_body_size 600M;

    location / {
        proxy_pass http://127.0.0.1:5004;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300;
        proxy_send_timeout 300;
    }
}
EOF

ln -sf /etc/nginx/sites-available/hca /etc/nginx/sites-enabled/hca
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

# 8. TLS certificate (HTTP-01 challenge needs port 80 reachable from the internet)
CERTBOT_ARGS=(--nginx -d "$DOMAIN" --redirect --non-interactive --agree-tos)
if [ -n "$CERTBOT_EMAIL" ]; then
  CERTBOT_ARGS+=(-m "$CERTBOT_EMAIL")
else
  CERTBOT_ARGS+=(--register-unsafely-without-email)
fi

if certbot "${CERTBOT_ARGS[@]}"; then
  systemctl enable --now certbot.timer || true
  APP_URL="https://$DOMAIN"
else
  echo "⚠️  certbot failed — site stays on plain HTTP and camera/mic will NOT work."
  echo "    Check that port 80 is open and that $DOMAIN resolves to $PUBLIC_IP, then rerun:"
  echo "    certbot --nginx -d $DOMAIN --redirect"
  APP_URL="http://$DOMAIN"
fi

# 9. Start services
supervisorctl reread && supervisorctl update && supervisorctl restart hca || supervisorctl start hca
nginx -t && systemctl restart nginx

echo ""
echo "✅ Deployment complete!"
echo "   App running at: $APP_URL"
echo "   Use this URL (not the bare IP) — camera/mic require HTTPS."
echo "   Logs: tail -f /var/log/hca.out.log"
echo "   Run telemetry: tail -f /opt/hca/logs/token_usage.jsonl"
echo "   Edit credentials: nano /opt/hca/.env"
