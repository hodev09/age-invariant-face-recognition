# Age-Invariant Face Recognition System

A web app that compares two face photos and tells you if they're the same person — even if the photos were taken years apart (like a childhood photo vs a recent one).

Upload two face images → the system detects faces, estimates ages, and uses AI to determine if it's the same person.

---

## What You Need Before Starting

You need to install two things on your Windows laptop. If you already have them, skip to [Running the Project](#running-the-project).

### 1. Install Python (version 3.11 or newer)

1. Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Click the big yellow "Download Python" button
3. Run the installer
4. **IMPORTANT: Check the box that says "Add Python to PATH"** at the bottom of the installer. This is the most common mistake — don't skip it.
5. Click "Install Now"
6. To verify, open **Command Prompt** (search "cmd" in Start menu) and type:
   ```
   python --version
   ```
   You should see something like `Python 3.12.x`

### 2. Install Node.js (version 20 or newer)

1. Go to [https://nodejs.org/](https://nodejs.org/)
2. Download the **LTS** version (the one on the left)
3. Run the installer — just click Next through everything
4. To verify, open a **new** Command Prompt and type:
   ```
   node --version
   npm --version
   ```
   You should see version numbers for both

---

## Running the Project

You need two terminals open — one for the backend (Python server) and one for the frontend (React app).

### Step 1: Start the Backend

Open **Command Prompt** (search "cmd" in Start menu) and run these commands one by one:

```bash
cd backend
```

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

After this, you should see `(venv)` at the beginning of your command line. That means the virtual environment is active.

Now install the dependencies:

```bash
pip install -r requirements.txt
```

This will take a few minutes — it's downloading AI models and libraries (~500MB). Be patient.

Now start the server:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

You should see something like:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Leave this terminal open. Don't close it.**

> **Note:** The first time you compare faces, the AI models (~300MB) will download automatically. This is a one-time thing.

### Step 2: Start the Frontend

Open a **second** Command Prompt window and run:

```bash
cd frontend
```

```bash
npm install
```

This installs the frontend dependencies. Takes a minute or two.

Then start the frontend:

```bash
npm run dev
```

You should see something like:
```
  VITE v7.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
```

### Step 3: Open the App

Open your browser and go to:

```
http://localhost:5173
```

You should see the app with two image upload areas and a Compare button.

### Step 4: Try It Out

1. Upload a face photo in the first slot (drag and drop, or click to browse)
2. Upload another face photo in the second slot
3. Click "Compare"
4. Wait a few seconds — the result will show whether it's the same person, along with estimated ages and a similarity score

---

## Troubleshooting

### "python is not recognized as an internal or external command"
You didn't check "Add Python to PATH" during installation. Uninstall Python, reinstall it, and make sure to check that box.

### "pip is not recognized"
Same issue as above — Python wasn't added to PATH. Or try using `python -m pip` instead of `pip`.

### "No face detected in the image"
The photo doesn't have a clear, visible face. Try a different photo where the face is clearly visible and facing the camera.

### "Multiple faces detected"
The photo has more than one person. Use a photo with only one face.

### The backend is slow on first request
That's normal. The AI models are being downloaded and loaded into memory for the first time. Subsequent requests will be much faster.

### "AI models are not loaded"
The backend is still starting up. Wait a few seconds and try again.

### Port already in use
If you see an error about port 8000 or 5173 being in use, another program is using that port. Close it, or use a different port:
```bash
# Backend on a different port
uvicorn app:app --reload --port 8001

# Frontend on a different port
npm run dev -- --port 3000
```

---

## Running Tests

### Backend tests
```bash
cd backend
venv\Scripts\activate
python -m pytest -v
```

### Frontend tests
```bash
cd frontend
npm test
```

---

## Project Structure (Quick Overview)

```
├── backend/                # Python server (FastAPI)
│   ├── ai_providers/       # AI models (InsightFace + OpenAI fallback)
│   ├── services/           # Core logic (pipeline, age rules, similarity)
│   ├── routes/             # API endpoints
│   ├── tests/              # Backend tests
│   └── app.py              # Server entry point
│
├── frontend/               # React web interface
│   ├── src/components/     # UI components (upload, button, results)
│   └── src/__tests__/      # Frontend tests
│
├── docker/                 # Docker deployment config
├── DOCUMENTATION.md        # Full technical documentation
└── README.md               # This file
```

---

## How It Works (Short Version)

1. You upload two face photos
2. The AI detects and crops the face from each photo
3. It estimates the age of each person
4. If the ages are too far apart (infant vs adult), it rejects the comparison as unreliable
5. Otherwise, it generates a 512-number "fingerprint" (embedding) for each face using ArcFace
6. It compares the two fingerprints using cosine similarity
7. If similarity ≥ 35% → same person. If < 35% → different person.

For the full technical deep-dive, see [DOCUMENTATION.md](DOCUMENTATION.md).

---

## Docker (Optional — For Advanced Users)

If you have Docker installed, you can run everything with one command:

```bash
cd docker
docker compose up --build
```

This starts the backend on `http://localhost:8000` and the frontend on `http://localhost:3000`.

---

## Tech Stack

- **Backend:** Python, FastAPI, InsightFace (RetinaFace + ArcFace), OpenCV, NumPy
- **Frontend:** React, Vite, Axios
- **Testing:** pytest + Hypothesis (backend), Vitest + fast-check (frontend)
- **Deployment:** Docker, Nginx
