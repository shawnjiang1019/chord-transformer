# Frontend Interface Plan: Chord Editor with Model Suggestions

## Context
The chord transformer backend (tokenizer, dataset, model, graph) is in progress. The user wants a React + FastAPI frontend for interactive chord composition — user adds chords and the model suggests next chords. Initially, the `ChordGraph` provides recommendations (the LM-based recommender is not yet implemented). The `SongwriterSession` class in `src/tools/songwriter.py` already has the session logic (add, suggest, undo, reset).

---

## Architecture

```
React (Vite + TS)  ←→  FastAPI  ←→  SongwriterSession / ChordGraph
   :5173                :8000          (Python backend)
```

Server-side sessions: `SongwriterSession` is stateful (holds history + undo snapshots), so the API manages sessions keyed by UUID. Frontend gets a session ID on creation.

---

## Directory Structure

```
chord_transformer/
├── api/                              # NEW
│   ├── main.py                       # FastAPI app, CORS, lifespan
│   ├── routes/
│   │   ├── session.py                # Session CRUD + chord ops
│   │   └── vocab.py                  # Static vocab endpoint
│   ├── schemas.py                    # Pydantic models
│   └── dependencies.py              # Shared state (sessions, graph, tokenizer)
│
├── frontend/                         # NEW
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       ├── api/client.ts             # Typed fetch wrapper
│       ├── types/index.ts
│       ├── components/
│       │   ├── ChordEditor.tsx       # Main layout
│       │   ├── ChordTimeline.tsx     # Horizontal chord sequence
│       │   ├── ChordChip.tsx         # Single chord badge
│       │   ├── SuggestionPanel.tsx   # Top-K suggestion buttons
│       │   ├── ConditioningControls.tsx  # Genre/decade/structure dropdowns
│       │   ├── ChordInput.tsx        # Text input with autocomplete
│       │   └── Toolbar.tsx           # Undo/reset/surprise/export
│       └── hooks/
│           ├── useSession.ts         # Session lifecycle
│           └── useSuggestions.ts     # Auto-fetch suggestions
├── src/                              # EXISTING - unchanged
└── configs/                          # EXISTING - unchanged
```

---

## Phase 1: FastAPI Backend

### Endpoints

| Method | Path | Description | Wraps |
|--------|------|-------------|-------|
| POST | `/api/sessions` | Create session → returns UUID | `SongwriterSession(recommender)` |
| GET | `/api/sessions/{id}` | Get history | `session.history` |
| POST | `/api/sessions/{id}/chords` | Add chord | `session.add(chord)` |
| POST | `/api/sessions/{id}/suggest` | Get suggestions | `session.suggest(top_k, surprise)` |
| POST | `/api/sessions/{id}/undo` | Undo last | `session.undo()` |
| POST | `/api/sessions/{id}/reset` | Clear all | `session.reset()` |
| GET | `/api/vocab` | Genres, decades, structures, roots | Read from vocab constants |

### Key files to create (in order)
1. `api/schemas.py` — Pydantic request/response models
2. `api/dependencies.py` — Init tokenizer, graph, recommender, session store
3. `api/routes/vocab.py` — Returns vocab constants (strip `<>` brackets)
4. `api/routes/session.py` — All 6 session endpoints; catch `NotImplementedError` on suggest, fall back to graph
5. `api/main.py` — CORS (allow `:5173`), include routers, lifespan init

### Critical backend files used
- `src/tools/songwriter.py` — `SongwriterSession` (add, suggest, undo, reset)
- `src/tools/recommend.py` — `ChordRecommender` (stub, falls back to graph)
- `src/graph/chord_graph.py` — `ChordGraph.recommend()` (working)
- `src/data/tokenizer.py` — `parse_chord()` for validation
- `src/data/vocab/special.py` — `GENRE_TOKENS`, `STRUCTURE_TOKENS`, `DECADE_TOKENS`

---

## Phase 2: React Frontend Scaffold

1. Scaffold: `npm create vite@latest frontend -- --template react-ts`
2. Configure Vite proxy → `/api` forwards to `localhost:8000`
3. Create `types/index.ts` — mirrors Pydantic schemas
4. Create `api/client.ts` — typed fetch wrapper, base URL from `VITE_API_URL`

---

## Phase 3: Core Components

### Component hierarchy
```
App
└── ChordEditor
    ├── Toolbar              (undo, reset, surprise toggle, export)
    ├── ConditioningControls  (genre, decade, structure dropdowns)
    ├── ChordTimeline         (horizontal scrollable sequence)
    │   └── ChordChip[]       (chord badges)
    ├── SuggestionPanel       (top-K buttons with probability bars)
    └── ChordInput            (text input with autocomplete)
```

### Data flow
```
User types/clicks chord → useSession.addChord()
  → POST /chords (updates history)
  → POST /suggest (auto-fetch suggestions)
  → SuggestionPanel re-renders
```

---

## Phase 4: Hooks & Integration

- `useSession` — manages sessionId, history, addChord/undo/reset
- `useSuggestions` — auto-fetches after history changes, 200ms debounce

---

## Phase 5: Polish

- Chord display formatting: `Fs` → `F#`, `As` → `Bb` (user-facing)
- Error handling for API failures
- "Graph mode" / "Model mode" indicator
- Update `.gitignore` with `frontend/node_modules/`, `frontend/dist/`

---

## Verification
1. Start API: `uvicorn api.main:app --reload` — test all endpoints via `/docs`
2. Start frontend: `cd frontend && npm run dev`
3. Create session → add chords → verify suggestions appear
4. Test undo/reset → verify history updates
5. Test conditioning dropdowns (will have no effect in graph mode — expected)
