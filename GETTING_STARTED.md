# Getting Started with taxembed

Welcome to the restructured taxembed project! This guide will help you get up and running.

## 🚀 Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
make install
# This runs: uv sync
```

### 2. Build C++ Extensions
```bash
make build
# This runs: python setup.py build_ext --inplace
```

### 3. Verify Installation
```bash
make test
# This runs: uv run pytest
```

Done! You're ready to go.

## 📖 What's New?

### Project Structure
```
taxembed/
├── src/taxembed/      ← Main package code
├── scripts/           ← Executable scripts
├── tests/             ← Unit tests
├── pyproject.toml     ← Dependencies (uv)
├── ruff.toml          ← Code quality rules
└── Makefile           ← Convenient commands
```

### Key Tools

| Tool | Purpose | Command |
|------|---------|---------|
| **uv** | Fast package manager | `make install` |
| **ruff** | Linter & formatter | `make lint`, `make format` |
| **pytest** | Testing framework | `make test` |
| **Makefile** | Command shortcuts | `make help` |

## 🎯 Common Tasks

### Training a Model
```bash
uv run python scripts/train.py \
  --dataset data/taxonomy_edges.mapped.edgelist \
  --checkpoint model.pth \
  --epochs 50
```

### Checking Code Quality
```bash
make lint          # Check for issues
make format        # Auto-fix issues
```

### Running Tests
```bash
make test          # Run all tests
make test-cov      # With coverage report
```

### Cleaning Up
```bash
make clean         # Remove build artifacts
```

## 📚 Documentation

- **README.md** - Full project documentation
- **QUICKSTART.md** - Detailed quick start guide
- **STRUCTURE.md** - Project organization
- **CONTRIBUTING.md** - Development guidelines
- **RESTRUCTURING_SUMMARY.md** - What changed and why

## 🔧 Development Workflow

### Step 1: Make Changes
Edit files in `src/taxembed/` or `scripts/`

### Step 2: Check Code Quality
```bash
make lint          # Find issues
make format        # Fix automatically
```

### Step 3: Test Your Changes
```bash
make test          # Run tests
```

### Step 4: Commit
```bash
git add .
git commit -m "Your message"
```

## ⚡ Useful Commands

```bash
# Show all available commands
make help

# Install dependencies
make install

# Build C++ extensions
make build

# Check code style
make lint

# Fix code style
make format

# Run tests with coverage
make test-cov

# Clean build artifacts
make clean

# Run a script
uv run python scripts/train.py --help
```

## 🐛 Troubleshooting

### "Command not found: uv"
Install uv: https://github.com/astral-sh/uv#installation

### "ModuleNotFoundError: No module named 'hype'"
Run: `make install && make build`

### "ruff: command not found"
Use: `uv run ruff check src/` (with `uv run` prefix)

### "Tests fail"
Check that dependencies are installed: `make install`

## 📝 Next Steps

1. Read [QUICKSTART.md](QUICKSTART.md) for detailed instructions
2. Check [STRUCTURE.md](STRUCTURE.md) to understand the layout
3. Review [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
4. Start training: `uv run python scripts/train.py --help`

## 💡 Tips

- Use `make help` to see all available commands
- Use `uv run` to execute Python scripts in the project environment
- Use `make format` before committing to maintain code style
- Use `make test` to verify your changes work

## 🤝 Need Help?

- Check the documentation files (README.md, STRUCTURE.md, etc.)
- Review CONTRIBUTING.md for development guidelines
- Open an issue on GitHub

Happy coding! 🎉
