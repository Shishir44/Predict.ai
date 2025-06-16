# Contributing to Battery Prediction System

Thank you for considering contributing to our project! We welcome contributions from the community to help improve and enhance the battery prediction system.

## Code of Conduct

Please note that this project is governed by our Code of Conduct. By participating, you are expected to uphold this code. We use a standard Code of Conduct similar to the Contributor Covenant.

## How to Contribute

### 1. Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/battery-prediction.git
   ```
3. Install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

### 2. Development Workflow

1. Create a branch for your feature:
   ```bash
   git checkout -b feature/AmazingFeature
   ```

2. Make your changes
3. Run tests:
   ```bash
   pytest tests/
   ```

4. Run pre-commit checks:
   ```bash
   pre-commit run --all-files
   ```

5. Commit your changes:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```

6. Push to the branch:
   ```bash
   git push origin feature/AmazingFeature
   ```

7. Create a Pull Request

## Pull Request Guidelines

- The pull request should include tests.
- If the pull request adds functionality, the docs should be updated.
- The pull request should work for Python 3.9 and above.
- Follow PEP 8 style guidelines.
- Include proper type hints.
- Add documentation for new features.
- Update the README.md with details of changes.

## Code Style

- Follow PEP 8 style guidelines
- Use type hints
- Write docstrings for all public functions
- Use meaningful variable names
- Keep lines under 79 characters
- Use snake_case for function and variable names
- Use PascalCase for class names

## Testing

- All new features should have unit tests
- Run tests before committing
- Maintain test coverage above 80%
- Use pytest for testing
- Include integration tests for critical paths

## Documentation

- Update README.md for new features
- Add docstrings to all public functions
- Document configuration options
- Include usage examples
- Keep API documentation up to date

## Versioning

We use semantic versioning:
- MAJOR.MINOR.PATCH
- MAJOR version when you make incompatible API changes
- MINOR version when you add functionality in a backwards-compatible manner
- PATCH version when you make backwards-compatible bug fixes

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
