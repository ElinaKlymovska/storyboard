# Branch Strategy

This project follows a structured branching strategy for organized development:

## Main Branches

### `master`
- **Purpose**: Production-ready code
- **Status**: Stable, tested, and ready for deployment
- **Protection**: Should be protected and require pull requests

### `development`
- **Purpose**: Integration branch for ongoing development
- **Status**: Contains latest features and bug fixes
- **Usage**: Main working branch for development

## Feature Branches

### `feature/enhancements`
- **Purpose**: New features and major enhancements
- **Naming**: `feature/feature-name`
- **Examples**: 
  - `feature/ai-analysis-improvements`
  - `feature/new-export-formats`
  - `feature/batch-processing`

### `bugfix/improvements`
- **Purpose**: Bug fixes and minor improvements
- **Naming**: `bugfix/issue-description`
- **Examples**:
  - `bugfix/memory-optimization`
  - `bugfix/time-parsing-edge-cases`
  - `bugfix/ui-improvements`

## Workflow

1. **Start new work**: Create feature branch from `development`
   ```bash
   git checkout development
   git pull origin development
   git checkout -b feature/your-feature-name
   ```

2. **Development**: Work on your feature/bugfix
   ```bash
   git add .
   git commit -m "feat: add new functionality"
   ```

3. **Integration**: Merge back to `development`
   ```bash
   git checkout development
   git merge feature/your-feature-name
   git push origin development
   ```

4. **Release**: When ready, merge `development` to `master`
   ```bash
   git checkout master
   git merge development
   git tag v1.0.0
   git push origin master --tags
   ```

## Branch Protection Rules

- `master`: Requires pull request review
- `development`: Requires pull request review for major changes
- Feature branches: Can be pushed directly for development

## Current Branches

- `master` - Production branch
- `development` - Main development branch
- `feature/enhancements` - Feature development
- `bugfix/improvements` - Bug fixes and improvements
