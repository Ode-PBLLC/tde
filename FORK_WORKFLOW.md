# Fork-Based Development Workflow

This guide explains how to use the fork-based development model for creating domain-specific MCP APIs while staying synchronized with framework improvements.

## 🎯 Overview

**The Big Picture**: One generic framework → Multiple domain implementations → Shared improvements

```
ode-mcp-generic (upstream)
├── tde-climate-api (your fork)
├── finance-api (teammate's fork)
├── healthcare-api (teammate's fork)
└── environmental-monitoring-api (teammate's fork)
```

**Key Benefit**: When anyone improves the core framework, everyone gets those improvements automatically.

## 🚀 Getting Started (First Time)

### Step 1: Fork the Generic Repository

**Option A: GitHub Web Interface**
1. Go to `https://github.com/ode-pbllc/ode-mcp-generic`
2. Click "Fork" button
3. Choose your organization/account
4. Name it something like: `your-domain-api` (e.g., `finance-api`, `healthcare-api`)

**Option B: GitHub CLI**
```bash
gh repo fork ode-pbllc/ode-mcp-generic --clone --fork-name your-domain-api
cd your-domain-api
```

### Step 2: Set Up Your Local Environment

```bash
cd your-domain-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and settings
```

### Step 3: Configure the Upstream Remote

```bash
# Add the generic repo as 'upstream' remote
git remote add upstream https://github.com/ode-pbllc/ode-mcp-generic.git

# Verify your remotes
git remote -v
# Should show:
# origin    https://github.com/your-org/your-domain-api.git (fetch)
# origin    https://github.com/your-org/your-domain-api.git (push)
# upstream  https://github.com/ode-pbllc/ode-mcp-generic.git (fetch)
# upstream  https://github.com/ode-pbllc/ode-mcp-generic.git (push)
```

## 🔧 Setting Up Your Domain

### Step 4: Create Your Domain-Specific Components

```bash
# Create your domain directory structure
mkdir -p data/your_domain
mkdir -p mcp/your_domain

# Copy and customize configuration
cp config/servers.example.json config/servers.json
cp config/citation_sources.template.json config/citation_sources.json
cp config/featured_queries.template.json config/featured_queries.json

# Create your domain server
cp mcp/template_server.py mcp/your_domain_server.py
```

### Step 5: Customize for Your Domain

Edit the following files:

**`config/servers.json`**
```json
{
  "servers": [
    {
      "name": "your_domain_data",
      "path": "mcp/your_domain_server.py",
      "description": "Main domain data server"
    }
  ]
}
```

**`mcp/your_domain_server.py`**
- Replace `TemplateServer` with `YourDomainServer`
- Implement your domain-specific tools
- Update metadata and citation information

**`config/citation_sources.json`**
- Map your tools to proper data source citations

### Step 6: Test Your Implementation

```bash
# Test your server directly
python mcp/your_domain_server.py

# Test the full API
python api_server.py

# Test with curl
curl -X POST http://localhost:8098/query \\
  -H "Content-Type: application/json" \\
  -d '{"query": "test my domain functionality"}'
```

## 🔄 Staying Updated with Framework Improvements

### Daily Workflow: Getting Upstream Changes

```bash
# 1. Fetch latest changes from the generic framework
git fetch upstream

# 2. Check what's changed (optional)
git log HEAD..upstream/main --oneline

# 3. Merge upstream changes into your domain
git merge upstream/main

# 4. If there are conflicts, resolve them
# (Your domain-specific files should rarely conflict)

# 5. Push updates to your fork
git push origin main
```

### Weekly Routine: Check for Updates

```bash
# Set up a weekly reminder to check for improvements
git fetch upstream
if [ $(git rev-list HEAD..upstream/main --count) -gt 0 ]; then
    echo "🔔 Framework updates available! Run 'git merge upstream/main'"
fi
```

### Automated Updates (Advanced)

Add this to `.github/workflows/sync-upstream.yml`:

```yaml
name: Sync with Upstream
on:
  schedule:
    - cron: '0 6 * * 1'  # Every Monday at 6 AM
  workflow_dispatch:

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Sync upstream changes
        run: |
          git remote add upstream https://github.com/ode-pbllc/ode-mcp-generic.git
          git fetch upstream
          git merge upstream/main
          git push origin main
```

## 🤝 Contributing Improvements Back

### When to Contribute Back

**✅ Contribute these improvements:**
- Bug fixes in core infrastructure (`api_server.py`, `mcp_chat.py`)
- New generic chart types in `visualization_server.py`
- Performance optimizations
- Documentation improvements
- New abstract base class features

**❌ Keep these in your fork:**
- Domain-specific servers (`finance_server.py`, `healthcare_server.py`)
- Domain-specific data and configurations
- Business logic specific to your use case

### How to Contribute

```bash
# 1. Create a feature branch for your improvement
git checkout -b improve-citation-system

# 2. Make your changes to generic components only
# Edit files like mcp_chat.py, visualization_server.py, etc.
# Do NOT modify your domain-specific files in this branch

# 3. Test your changes
python -m pytest test_scripts/

# 4. Commit and push to your fork
git commit -m "Improve citation system performance"
git push origin improve-citation-system

# 5. Create a Pull Request to ode-mcp-generic
gh pr create --base main --head improve-citation-system \\
  --title "Improve citation system performance" \\
  --body "Optimizes citation lookup by adding caching layer"
```

## 🚨 Common Scenarios & Solutions

### Scenario 1: Merge Conflicts

```bash
# When you get conflicts during merge
git merge upstream/main
# Auto-merging mcp/mcp_chat.py
# CONFLICT (content): Merge conflict in mcp/mcp_chat.py

# 1. Check what's conflicting
git status

# 2. Edit conflicted files - usually these are safe to resolve:
#    - Keep upstream changes for generic components
#    - Keep your changes for domain-specific components

# 3. Mark as resolved and complete merge
git add .
git commit -m "Merge upstream improvements"
```

### Scenario 2: You Modified a Generic File

If you accidentally modified a generic file instead of extending it:

```bash
# 1. Save your changes
git stash

# 2. Get clean upstream version
git checkout upstream/main -- mcp/visualization_server.py

# 3. Reapply your changes properly (create extension or submit PR)
git stash pop
# Review and move your changes to appropriate domain-specific files
```

### Scenario 3: Major Framework Update

```bash
# For major updates, create a backup branch first
git checkout -b backup-before-major-update
git checkout main

# Then proceed with normal merge
git fetch upstream
git merge upstream/main
```

## 📋 Checklist for New Team Members

- [ ] Fork the `ode-mcp-generic` repository with a descriptive name
- [ ] Set up local development environment
- [ ] Add upstream remote: `git remote add upstream https://github.com/ode-pbllc/ode-mcp-generic.git`
- [ ] Copy and customize configuration files
- [ ] Create domain-specific server extending base classes
- [ ] Test implementation locally
- [ ] Set up weekly upstream sync routine
- [ ] Read the contribution guidelines for improving generic components

## 🤔 FAQ

**Q: How often should I sync with upstream?**
A: At least weekly, or whenever you see notifications about framework improvements.

**Q: What if I break something during a merge?**
A: Every merge creates a commit - you can always `git reset --hard HEAD~1` to undo the merge and try again.

**Q: Can I modify the generic files for my domain?**
A: You can, but it's better to extend them or contribute improvements back. This keeps future merges clean.

**Q: What if my domain needs a new generic feature?**
A: Great! Implement it in the generic components and submit a PR. Everyone benefits.

**Q: How do I handle sensitive data or domain-specific secrets?**
A: Keep these in your fork only. Use environment variables and never commit secrets to any repository.

**Q: What if I want to use a different version of a dependency?**
A: Modify `requirements.txt` in your fork, but be aware this might cause conflicts during upstream merges.

## 🆘 Getting Help

**For Framework Issues:**
- Create an issue in `ode-mcp-generic` repository
- Ask in team chat with `@framework-help` tag

**For Domain-Specific Issues:**
- Check with teammates working on similar domains
- Review the `examples/` directory for patterns
- Consult the abstract base class documentation

**For Workflow Issues:**
- This document covers most scenarios
- Git documentation: https://git-scm.com/docs
- GitHub fork documentation: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks

---

**Remember**: The goal is to maximize code reuse and shared improvements while maintaining independence for domain-specific needs. When in doubt, ask the team! 🚀