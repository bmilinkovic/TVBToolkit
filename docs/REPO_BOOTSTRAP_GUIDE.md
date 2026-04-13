# Repo Bootstrap Guide

## Local setup

```bash
cd /Users/borjan/CNRS/projects/TVBToolkit
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

## Initialize main branch

```bash
git checkout -b main
git add .
git commit -m "Initial TVBToolkit scaffold with AdEx-first design"
```

## Connect to new GitHub repository

```bash
git remote add origin <NEW_GITHUB_REPO_URL>
git push -u origin main
```

## Suggested release flow

```bash
git checkout -b release/v0.1.0
git push -u origin release/v0.1.0
git tag v0.1.0
git push origin v0.1.0
```

