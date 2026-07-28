# Deploying to Vercel

## 1. Import the repository

1. Push this project to GitHub.
2. In Vercel, select **Add New** then **Project** and import the repository.
3. Keep the project root as the repository root. Vercel detects `api/index.py` as the Python entry point.
4. Deploy.

`vercel.json` rewrites all API paths to the FastAPI app, so these URLs remain unchanged:

- `/docs`
- `/auth/linkedin/login`
- `/auth/linkedin/callback`
- `/linkedin/post`
- `/instagram/login`
- `/instagram/post`

## 2. Configure environment variables

In **Project Settings > Environment Variables**, add the values from the local `.env` file:

```text
CLIENT_ID
CLIENT_SECRET
DATABASE_URL
SUPABASE_URL
SUPABASE_KEY
INSTAGRAM_USERNAME
INSTAGRAM_PASSWORD
INSTAGRAM_SESSION_ID
```

Set `REDIRECT_URI` to the deployed callback address:

```text
https://YOUR-PROJECT.vercel.app/auth/linkedin/callback
```

Add the same exact production callback URL in the LinkedIn Developer Portal.

## 3. Important Instagram limitation

Vercel functions have ephemeral local storage. The local `instagram_session.json` file is not reliable between cold starts, so Instagram uploads may require signing in again after a deployment or cold start.

For reliable Instagram posting, run this FastAPI backend on a persistent service such as Railway, Render, Fly.io, or Cloud Run, and store encrypted session data in a persistent database or storage service. Vercel is suitable for the LinkedIn endpoints and API documentation.

## 4. Deploy updates

Each Git push to the connected branch triggers a deployment. For a manual deployment with the Vercel CLI:

```powershell
npm install --global vercel
vercel --prod
```