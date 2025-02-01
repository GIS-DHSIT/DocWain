# DocWain

An intelligent document processing and QA system that supports multiple document formats and sources.

## Features

- Multi-format document support (PDF, Word, Excel, CSV)
- Multiple source options (Local, S3, FTP)
- Parallel document processing
- Real-time progress tracking
- Intelligent document chunking
- Advanced retrieval system
- Interactive chat interface

## Supported File Types

- PDF Documents (.pdf)
- Microsoft Word (.doc, .docx)
- Microsoft Excel (.xls, .xlsx)
- CSV Files (.csv)

## Quick Start

### Using Docker

```bash
# Clone the repository
git clone https://github.com/GIS-DHSIT/DocWain.git
cd DocWain

# Copy and edit environment variables
cp .env.example .env
# Edit .env with your configuration

# Build and run with Docker
docker-compose up -d
```

### Local Installation

```bash
# Clone the repository
git clone https://github.com/GIS-DHSIT/DocWain.git
cd DocWain

# Install dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell

# Run the application
streamlit run DocWain/web/app.py
```

## Configuration

Create a `.env` file with your configuration:

```env
OPENAI_API_KEY=your-api-key
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
AWS_DEFAULT_REGION=your-region
GROQ_API_KEY=your-groq-key
```

## Usage

1. Upload documents through the web interface
2. Or configure remote sources (S3/FTP) in the settings
3. Ask questions about your documents
4. View source references and explanations

## Development

```bash
# Install development dependencies
poetry install --with dev

# Run tests
pytest

# Run linting
flake8 DocWain
black DocWain
isort DocWain

# Run with hot reload
./scripts/run.sh dev
```

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request