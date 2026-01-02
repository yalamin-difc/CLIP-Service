## Zips containing the latest integration changes

This folder contains two zip packages you can forward/share.

### Frontend package

- `frontend-integration-changes.zip`
  - `0001-frontend.patch` (apply via git)
  - `files/` (the exact changed files)

Apply (from your frontend repo root):

```bash
git am < 0001-frontend.patch
```

If you prefer a non-commit apply:

```bash
git apply 0001-frontend.patch
```

### Backend package

- `backend-integration-changes.zip`
  - `0001-backend.patch` (apply via git)
  - `files/` (the exact changed files)

Apply (from your backend repo root):

```bash
git am < 0001-backend.patch
```

If you prefer a non-commit apply:

```bash
git apply 0001-backend.patch
```

### CLIP-Service

No CLIP-Service changes are included in these zips.

