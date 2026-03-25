# Cloudflare Python Worker starter

This is a minimal repo scaffold for deploying Python code to Cloudflare Workers.

## Files

- `src/index.py` — Worker entrypoint
- `wrangler.jsonc` — Cloudflare Wrangler config
- `pyproject.toml` — Python project metadata
- `package.json` — Wrangler scripts

## Local dev

1. Install dependencies:
   ```bash
   npm install
   ```
2. Log in to Cloudflare:
   ```bash
   npx wrangler login
   ```
3. Run locally:
   ```bash
   npm run dev
   ```

## Deploy

```bash
npm run deploy
```

## Important

The current `src/index.py` is only a starter.
Replace the `run_simulation()` function with the actual logic from your `sim.py`.

If your original `sim.py` depends on:
- Streamlit
- Matplotlib GUI windows
- long-running local state
- large scientific Python packages not supported in Workers

then it will likely need refactoring before Cloudflare Workers can run it cleanly.
