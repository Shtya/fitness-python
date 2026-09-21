# So7baFit AI Body Scan

Nest calls this on `http://127.0.0.1:8010`. Keep that port local (do not open it on the public IP).

Needs **Python 3.10 or 3.11** (3.13 will fail with MediaPipe).

## Local (Windows)

```powershell
cd E:\.env\Me\So7baFit\ai-body-scan
.\.venv\Scripts\Activate.ps1
python -m uvicorn main:app --host 127.0.0.1 --port 8010
```

Health: http://127.0.0.1:8010/health

---

## Server (Linux + PM2)

Same pattern as the Nest backend: clone/pull, install, then PM2.

### 1. Python 3.11 (once)

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
python3.11 --version
```

### 2. App + venv

On this server the folder is `/root/python` (next to `backend/` and `frontend/`).

```bash
cd /root
git clone https://github.com/Shtya/fitness-python.git python
cd python
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Later updates (`/root/Makefile`):

```bash
make python
```

Or all three:

```bash
make both
```

### 3. Start with PM2

GitHub’s first commit does **not** include `ecosystem.config.cjs`. Start without it:

```bash
cd /root/python
pm2 start ./.venv/bin/python --name ai-body-scan --interpreter none -- -m uvicorn main:app --host 127.0.0.1 --port 8010
pm2 save
```

If `ecosystem.config.cjs` exists after a later pull:

```bash
cd /root/python
pm2 start ecosystem.config.cjs
pm2 save
```

Useful commands:

```bash
pm2 status
pm2 logs ai-body-scan
pm2 restart ai-body-scan
```

Check it is up:

```bash
curl http://127.0.0.1:8010/health
```

Expect: `{"ok":true,"service":"ai-body-scan"}`

### 4. Point Nest at it

In the **backend** `.env` on the server:

```
BODY_MEASUREMENT_SERVICE_URL=http://127.0.0.1:8010
BODY_MEASUREMENT_TIMEOUT_MS=90000
```

Then restart Nest as you already do (`pm2 restart 0` or the Nest app name).
