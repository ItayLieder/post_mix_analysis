# Workspace Configuration

## 📁 Output Directory Configuration

The cleaned up post-mix analysis system now saves all outputs to an **external workspace directory** to keep your Git repository clean and prevent accidentally committing large audio files.

### Default Workspace Location
```
/Users/itay/Documents/post_mix_data/PostMixRuns
```

### Directory Structure
When you run the pipeline, it will create timestamped project directories like:
```
/Users/itay/Documents/post_mix_data/PostMixRuns/
├── postmix_cleaned_v1_20250825-143022/
│   ├── inputs/                  # Imported source files
│   ├── outputs/                 # All processed audio
│   │   ├── premaster/          # Pre-mastered versions
│   │   ├── master/             # Final mastered tracks
│   │   └── stream_previews/    # Platform-normalized versions
│   ├── reports/                # HTML reports and plots
│   │   ├── assets/            # Report files
│   │   └── bundles/           # Reproducibility bundles
│   └── work/                   # Temporary processing files
└── postmix_cleaned_v1_20250825-151534/  # Next run
    └── ...
```

## 🔧 Configuration Options

### Environment Variable Override
You can change the workspace location using an environment variable:
```bash
export POST_MIX_WORKSPACE_ROOT="/path/to/your/preferred/location"
```

### Programmatic Override
Or set it in code:
```python
from config import CONFIG
CONFIG.workspace.workspace_root = "/custom/path/to/workspace"
```

### Temporary Override
For a single run:
```python
workspace_paths = make_workspace(base_dir="/temporary/location", project="test_run")
```

## ✅ Benefits

### Git Repository Stays Clean
- No large audio files committed accidentally
- Repository size stays manageable
- Version control focuses on code, not data

### Organized Output Management
- Timestamped runs for easy comparison
- Complete project isolation
- Easy cleanup of old runs

### Professional Workflow
- Separate code from data
- Easy backup of just the outputs
- Better for collaboration

## 🚀 Getting Started

1. **The workspace directory is already created for you**
2. **Just run the cleaned notebook**: `post_mix_cleaned.ipynb`
3. **All outputs will automatically go to the external directory**
4. **Your Git repo stays clean** ✨

## 📝 Notes

- The old `postmix_runs/` directory (if it exists in your repo) is now ignored by `.gitignore`
- All existing functionality works exactly the same, just with better organization
- You can safely delete any old output directories from your Git repo
- The external workspace directory will be created automatically when you run the pipeline

## 🔍 Verification

You can verify the configuration is working by running:
```python
from config import CONFIG
print(f"Workspace root: {CONFIG.workspace.get_workspace_root()}")
```

This should show: `/Users/itay/Documents/post_mix_data/PostMixRuns`