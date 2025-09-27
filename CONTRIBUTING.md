# Contributing Guidelines

Thank you for your interest in contributing to the Video Analysis Pipeline! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic understanding of video processing concepts

### Development Setup
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/video-analysis-pipeline.git
   cd video-analysis-pipeline
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Branch Strategy

We follow a structured branching strategy:

- `master`: Production-ready code
- `development`: Main development branch
- `feature/feature-name`: New features
- `bugfix/issue-description`: Bug fixes

### Creating a Branch
```bash
git checkout development
git pull origin development
git checkout -b feature/your-feature-name
```

## Code Standards

### Python Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write descriptive docstrings
- Keep functions focused and small

### Commit Messages
Use conventional commit format:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

Examples:
```bash
git commit -m "feat: add batch processing for multiple videos"
git commit -m "fix: resolve memory leak in frame extraction"
git commit -m "docs: update installation instructions"
```

### Code Review Process
1. Ensure all tests pass
2. Update documentation if needed
3. Create a pull request to `development`
4. Address review feedback
5. Merge after approval

## Testing

### Running Tests
```bash
python -m pytest tests/
```

### Writing Tests
- Write unit tests for new functions
- Include integration tests for major features
- Aim for good test coverage

## Documentation

### Code Documentation
- Use docstrings for all public functions
- Include type hints
- Provide usage examples

### User Documentation
- Update README.md for user-facing changes
- Add/update docstrings for API changes
- Include examples in documentation

## Reporting Issues

### Bug Reports
When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

### Feature Requests
For feature requests, please include:
- Use case description
- Proposed solution
- Alternative solutions considered
- Additional context

## Pull Request Process

1. **Fork and Branch**: Create a feature branch from `development`
2. **Develop**: Make your changes with proper commits
3. **Test**: Ensure all tests pass
4. **Document**: Update documentation as needed
5. **Submit**: Create a pull request to `development`
6. **Review**: Address feedback from maintainers
7. **Merge**: After approval, your changes will be merged

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## Development Environment

### Recommended Tools
- **IDE**: VS Code, PyCharm, or similar
- **Linting**: flake8, black, mypy
- **Testing**: pytest
- **Version Control**: Git with conventional commits

### Environment Variables
Create a `.env` file for local development:
```bash
REPLICATE_API_TOKEN=your_token_here
NOTION_API_TOKEN=your_token_here
KEYFRAMES_DB_ID=your_db_id
STORYBOARD_DB_ID=your_db_id
```

## Getting Help

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Use GitHub Issues for bugs and feature requests
- **Documentation**: Check README.md and inline documentation

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the project's coding standards

Thank you for contributing! 🎉
