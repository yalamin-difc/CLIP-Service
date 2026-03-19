## Deploy CLIP-Service to a Google GPU VM (safe, repeatable)
 
### Endpoints used by backend
- `GET /health`
- `POST /encode-image` (multipart: `file` or `image`)
- `POST /encode-text` (multipart: `text` or `queryText`)
- `POST /analyze-image` (multipart: `file` or `image`)
- `GET /metrics` (protected in production)
 
### Pull updates on the GPU VM
```bash
ssh <user>@<vm-ip>
cd ~/CLIP-Service
git fetch origin
git checkout cursor/items-endpoints-review-f8df
git pull origin cursor/items-endpoints-review-f8df
```

Install dependencies (venv recommended)

```bash
cd ~/CLIP-Service
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

Run (defaults to 8000)

```bash
export PORT=8000
bash scripts/start.sh
```

Health check:

```bash
curl -i http://127.0.0.1:8000/health
```

Run as a service (systemd)
Create /etc/systemd/system/clip.service:

```ini
[Unit]
Description=CLIP Service
After=network.target
 
[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/CLIP-Service
Environment="PORT=8000"
ExecStart=/home/ubuntu/CLIP-Service/.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=2
 
[Install]
WantedBy=multi-user.target
```

Enable/restart:

```bash
sudo systemctl daemon-reload
sudo systemctl enable clip
sudo systemctl restart clip
sudo systemctl status clip --no-pager
```
