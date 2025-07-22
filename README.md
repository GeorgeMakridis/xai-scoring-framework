# XAI Scoring Framework

A comprehensive framework for evaluating and scoring Explainable AI (XAI) methods. This framework provides a simple ML model approach to recommend the most suitable XAI methods and AI models for your datasets and use cases.

## 🚀 Features

### Core Functionality
- **XAI Method Evaluation**: Score and compare SHAP, LIME, PFI, and PDP methods
- **AI Model Recommendations**: Get tailored recommendations for Random Forest, XGBoost, SVM, and Neural Networks
- **Domain-Specific Scoring**: Optimized scoring for healthcare, finance, cybersecurity, and more
- **Interactive Web Interface**: Professional Flask-based UI with modern design
- **REST API**: Full API for external integrations and automation

### Supported XAI Methods
- **SHAP** (SHapley Additive exPlanations): Game theory-based approach
- **LIME** (Local Interpretable Model-agnostic Explanations): Local approximation
- **PFI** (Permutation Feature Importance): Feature importance measurement
- **PDP** (Partial Dependence Plots): Feature-prediction relationships

### Supported AI Models
- Random Forest
- XGBoost
- Support Vector Machines (SVM)
- Neural Networks

### Scoring Metrics
- **Fidelity**: How accurately the XAI method represents the model's behavior
- **Stability**: Consistency of explanations with small input changes
- **User Rating**: User interpretability and trust ratings
- **Simplicity**: Ease of understanding the method's outputs

## 🏗️ Architecture

The framework consists of two main components:

1. **Web UI** (`web_app.py`): Professional Flask-based interface
2. **REST API** (`api/main.py`): FastAPI-based API for external access

Both components share the same core logic from the original `app.py` file.

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Start both web UI and API
docker-compose up -d

# Access the applications
# Web UI: http://localhost:8501
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the web UI
python3 web_app.py
# Access at: http://localhost:8501

# Start the API (in another terminal)
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
# Access at: http://localhost:8000
```

### Option 3: Test Setup

```bash
# Run comprehensive tests
python3 test_setup.py
```

## 📁 Project Structure

```
xai-scoring-framework/
├── app.py                          # Original Streamlit app (core logic)
├── web_app.py                      # New Flask web application
├── api/
│   └── main.py                     # FastAPI REST API
├── templates/
│   └── index.html                  # Web UI template
├── static/
│   ├── css/
│   │   └── style.css              # Custom styles
│   └── js/
│       └── app.js                 # Frontend JavaScript
├── uploads/                        # File upload directory
├── data/                          # Data storage
├── logs/                          # Application logs
├── Dockerfile                      # Docker configuration
├── docker-compose.yml             # Docker Compose setup
├── requirements.txt                # Python dependencies
├── test_setup.py                  # Setup verification script
└── README.md                      # This file
```

## 🔧 Configuration

### Environment Variables

- `FLASK_ENV`: Flask environment (development/production)
- `FLASK_APP`: Flask application entry point
- `API_HOST`: API host address
- `API_PORT`: API port number

### Data Files

The framework requires the following data files:

**Required:**
- `Fame XAI scoring Framework_v2-2.xlsx`: Benchmark data
- `xai_results.csv`: User ratings for XAI methods

**Optional:**
- `JSON_1.docx`, `JSON_2.docx`, `JSON_3.docx`: Additional survey data

## 📊 API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `GET /` - API information

### Data Management
- `POST /load-data` - Upload and load benchmark data
- `POST /score-dataset` - Score an uploaded dataset
- `POST /score-features` - Score based on provided features

### Metadata
- `GET /api/methods` - Get available XAI methods
- `GET /api/domains` - Get available domains

## 🎯 Usage

### Web Interface

1. **Upload Data**: Upload your benchmark Excel file and ratings CSV
2. **Upload Dataset**: Upload your dataset for scoring
3. **Configure**: Select domain and adjust metric weights
4. **Get Results**: View recommendations and detailed scores

### API Usage

```python
import requests

# Load benchmark data
files = {
    'excel_file': open('benchmark.xlsx', 'rb'),
    'ratings_file': open('ratings.csv', 'rb')
}
response = requests.post('http://localhost:8000/load-data', files=files)

# Score a dataset
files = {'dataset_file': open('your_dataset.csv', 'rb')}
data = {
    'domain': 'healthcare',
    'fidelity_weight': 0.3,
    'stability_weight': 0.3,
    'user_rating_weight': 0.2,
    'simplicity_weight': 0.2
}
response = requests.post('http://localhost:8000/score-dataset', files=files, data=data)
```

## 🔍 Domain-Specific Bonuses

The framework applies domain-specific bonuses to XAI scores:

- **Healthcare**: 1.2x bonus (high interpretability requirements)
- **Finance**: 1.15x bonus (regulatory compliance)
- **Cybersecurity**: 1.1x bonus (security considerations)
- **Autonomous Vehicles**: 1.25x bonus (safety-critical)
- **Recommendation Systems**: 1.05x bonus (user experience)
- **General**: 1.0x (no bonus)

## 🧪 Testing

Run the comprehensive test suite:

```bash
python3 test_setup.py
```

This will test:
- ✅ Import functionality
- ✅ Data file availability
- ✅ Docker configuration
- ✅ Web application
- ✅ API functionality

## 🐳 Docker

### Build and Run

```bash
# Build the image
docker build -t xai-scoring-framework .

# Run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Individual Services

```bash
# Web UI only
docker run -p 8501:8501 xai-scoring-framework python3 web_app.py

# API only
docker run -p 8000:8000 xai-scoring-framework python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## 🔧 Development

### Prerequisites

- Python 3.9+
- Docker (optional)
- Git

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd xai-scoring-framework

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python3 test_setup.py
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 Documentation

- **API Documentation**: Available at `http://localhost:8000/docs` when API is running
- **Interactive API**: Swagger UI at `http://localhost:8000/docs`
- **ReDoc**: Alternative API docs at `http://localhost:8000/redoc`

## 🤝 Support

For issues and questions:
1. Check the test results: `python3 test_setup.py`
2. Review the API documentation
3. Check the logs in the `logs/` directory

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📖 Citation

If you use this framework in your research, please cite:

```
@article{xai-scoring-framework,
  title={XAI Scoring Framework: A Comprehensive Approach to Explainable AI Evaluation},
  author={Your Name},
  journal={Journal of Machine Learning Research},
  year={2024}
}
```

---

**Note**: This framework replaces the original Streamlit interface with a professional Flask-based web application while maintaining all core functionality and adding a comprehensive REST API for external integrations. 