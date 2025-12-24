# PyPI Deployment Guide

## What are Artifacts?

**Artifacts** are files produced during CI/CD runs:
- **Distribution files** (`.whl` and `.tar.gz`) 
- Created by `python -m build`
- Can be downloaded from GitHub Actions > Artifacts tab
- Ready to install with `pip install mlvern-0.1.0-py3-none-any.whl`

## Auto-Deploy to PyPI

The workflow automatically publishes to PyPI when you create a **git tag** starting with `v`:

```bash
git tag v0.1.0
git push origin v0.1.0
```

This triggers the `publish` job which:
1. ✅ Runs all tests and linting checks
2. ✅ Builds distribution files
3. ✅ Publishes to PyPI

## Setup (One-time)

### Step 1: Create PyPI Account
- Go to [pypi.org](https://pypi.org)
- Sign up (or login if you have one)
- Go to **Account Settings → API tokens**
- Click "Add API token"
- Name it `github-actions` (or similar)
- Copy the token (starts with `pypi-`)

### Step 2: Add GitHub Secret
In your GitHub repo:
1. Go to **Settings → Secrets and variables → Actions**
2. Click "New repository secret"
3. Name: `PYPI_API_TOKEN`
4. Paste your API token
5. Click "Add secret"

### Step 3: Verify Workflow
The workflow now has a `publish` job that:
- Only runs on version tags (`v0.1.0`, `v1.0.0`, etc.)
- Automatically publishes to PyPI after successful tests

## Deploying a New Version

```bash
# Update version in pyproject.toml
# e.g., version = "0.2.0"

# Commit changes
git add pyproject.toml
git commit -m "Bump version to 0.2.0"

# Create and push tag
git tag v0.2.0
git push origin main
git push origin v0.2.0
```

Check the **Actions** tab in GitHub to watch the deployment!

## Install from PyPI

Once published:
```bash
pip install mlvern
```

## Troubleshooting

- **"No PYPI_API_TOKEN secret"** → Add the secret in repo settings
- **"Unauthorized"** → Token is invalid or expired, create a new one
- **"Invalid version format"** → Version in `pyproject.toml` must match git tag

## Alternative: Test PyPI (Optional)

To test before publishing to production:

```bash
# Create token at https://test.pypi.org (different from pypi.org)
# Add secret: TEST_PYPI_API_TOKEN
```

Then create a tag like `test-v0.1.0` to publish to test.pypi.org.
