# Contributing to Misestimation from Aggregation

We welcome contributions to this project! This document provides guidelines for contributing.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/Basso42/misestimation_from_aggregation.git
cd misestimation_from_aggregation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where possible
- Write docstrings for all public functions and classes
- Keep functions focused and modular

## Testing

- Write tests for new functionality
- Ensure all tests pass before submitting:
```bash
pytest tests/
```

- Aim for high test coverage:
```bash
pytest --cov=misestimation_from_aggregation tests/
```

## Documentation

- Update docstrings and README.md for new features
- Include examples in docstrings where helpful
- Update the tutorial notebook if adding major functionality

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation as needed
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Include tests for new functionality
- Update documentation as needed
- Keep changes focused and atomic

## Reporting Issues

When reporting issues, please include:

- Python version
- Package version
- Minimal example to reproduce the issue
- Expected vs actual behavior
- Error messages (if any)

## Code of Conduct

Be respectful and inclusive. We want this to be a welcoming environment for all contributors.

## Questions?

Feel free to open an issue for questions about contributing or usage.