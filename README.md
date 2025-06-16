# Corporate Momentum Document API

A FastAPI REST service for document storage and summarization, optimized for large documents with intelligent summarization capabilities. The system uses file-based storage to handle large documents efficiently.

### Summarization Pipeline

1. Text is saved to file with metadata
2. Summarization starts asynchronously as a background task
3. Attempts Hugging Face API first
4. Uses extractive summarization if API unavailable

## Quick Start with Docker Compose

### 🚀 Production

```bash
docker-compose up
```

### 🛠️ Development (with source code mounting)

```bash
docker-compose -f docker-compose.yml -f docker-compose.override.yml up
```

## Local Development (without Docker)

```bash
# Create and activate virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Then replace placeholders with your configuration variables.

### Storage

- Documents are stored in `data/documents/`
- Summaries are stored in `data/summaries/`
- Metadata is stored in `data/metadata/`
- All data persists across container restarts

## 🧪 Testing

The project uses **pytest** for comprehensive testing with proper test organization and fixtures.

### Running Tests

```bash
# Run all tests
python -m pytest test_api.py -v

# Run a specific test class
python -m pytest test_api.py::TestDocumentStorage -v

# Run with detailed output for debugging
python -m pytest test_api.py -v --tb=short
```

### VS Code Integration

Use the VS Code Command Palette (Ctrl+Shift+P) and run:

- **Tasks: Run Task** → **Run Tests** (full suite)
- **Tasks: Run Task** → **Run Tests with Coverage** (detailed output)

### Test Organization

Tests are organized into logical classes:

- `TestHealthAndRoot` - Basic API functionality
- `TestDocumentStorage` - Document storage operations
- `TestDocumentRetrieval` - Document access and streaming
- `TestDocumentListing` - Document enumeration
- `TestSummarization` - Summary generation and status
- `TestIntegration` - End-to-end workflows
- `TestPerformance` - Load and performance testing

### Prerequisites for Testing

1. **Start the API server** in a separate terminal

2. **Install test dependencies**:

   ```bash
   pip install -r requirements-dev.txt
   ```

3. **Run tests** as shown above

The API must be running on `http://localhost:8000` for the tests to pass.
