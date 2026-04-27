# FairMind - AI Bias Audit & Remediation Platform

FairMind is a comprehensive full-stack platform for auditing machine learning models for bias, explaining predictions through SHAP analysis, and implementing fairness remediation techniques. The platform provides an intuitive interface to upload datasets, configure audit parameters, visualize bias metrics, and apply remediation strategies.

## Overview

FairMind combines cutting-edge fairness libraries (Fairlearn, AIF360) with interactive visualizations and AI-powered insights (Google Gemma) to help data scientists and ML practitioners identify, understand, and mitigate algorithmic bias in their models.

### Key Features

- **Bias Audit**: Comprehensive fairness auditing using Fairlearn and AIF360
- **Model Explanation**: SHAP-based feature importance and bias driver identification
- **Bias Remediation**: Apply fairness-enhancing techniques to training data
- **Interactive Dashboard**: Real-time visualization of bias metrics by demographic groups
- **AI Insights**: Gemma-powered natural language insights on audit findings
- **Report Generation**: Export comprehensive PDF reports with metrics and recommendations

## Project Structure

```
FairMind/
├── backend/              # Python FastAPI backend
│   ├── app/
│   │   ├── main.py      # FastAPI application setup
│   │   ├── config.py    # Configuration management
│   │   ├── database.py  # SQLAlchemy database setup
│   │   ├── models/      # Data models (Job)
│   │   └── routers/     # API endpoints
│   │       ├── upload.py      # CSV upload endpoint
│   │       ├── audit.py       # Bias audit endpoint
│   │       ├── explain.py     # Model explanation endpoint
│   │       ├── remediate.py   # Bias remediation endpoint
│   │       └── report.py      # PDF report generation
│   ├── ml/              # ML engine modules
│   │   ├── audit.py     # AuditEngine for bias detection
│   │   ├── explain.py   # ExplainEngine for SHAP analysis
│   │   ├── remediate.py # RemediationEngine for fairness techniques
│   │   └── preprocessing.py
│   ├── reporting/       # PDF report generation
│   ├── datasets/        # Sample datasets (adult.csv, compas.csv)
│   ├── tests/           # Unit and integration tests
│   └── requirements.txt # Python dependencies
│
├── frontend/            # React + Vite frontend
│   ├── src/
│   │   ├── pages/       # Main pages
│   │   │   ├── LandingPage.jsx    # Welcome page
│   │   │   ├── WizardPage.jsx     # Upload & config wizard
│   │   │   └── DashboardPage.jsx  # Results dashboard
│   │   ├── components/ # React components
│   │   ├── hooks/      # Custom React hooks
│   │   ├── api/        # API client
│   │   └── main.jsx    # React entry point
│   ├── package.json    # Node.js dependencies
│   └── vite.config.js  # Vite configuration
│
├── docker-compose.yml  # Docker orchestration
└── requirements.txt    # Root dependencies (this file)
```

## Tech Stack

### Backend
- **Framework**: FastAPI (Python web framework)
- **API Server**: Uvicorn (ASGI server)
- **Database**: SQLite with SQLAlchemy ORM
- **ML/Fairness**: 
  - Fairlearn (fairness metrics & algorithms)
  - AIF360 (IBM fairness toolkit)
  - Scikit-learn (machine learning)
  - TensorFlow (deep learning)
  - SHAP (model explainability)
- **Data Processing**: Pandas, NumPy
- **Testing**: Pytest
- **Reporting**: ReportLab (PDF generation)

### Frontend
- **Framework**: React 19 with Vite
- **Routing**: React Router v7
- **Styling**: Tailwind CSS 4
- **HTTP Client**: Axios
- **Visualization**: Recharts
- **AI Integration**: Google Generative AI (Gemma)
- **UI Components**: Custom React components with Tailwind styling
- **Code Quality**: ESLint

## Installation

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose (optional, for containerized setup)

### Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend
npm install
```

## Running the Application

### Development Mode

**Terminal 1 - Backend API:**
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

API documentation available at: http://localhost:8000/docs

**Terminal 2 - Frontend Dev Server:**
```bash
cd frontend
npm run dev
```

Frontend available at: http://localhost:5173

### Docker Setup

```bash
docker-compose up --build
```

This starts:
- Backend API on http://localhost:8000
- Frontend on http://localhost:3000

## API Endpoints

### Upload
- `POST /upload` - Upload CSV dataset for analysis

### Audit
- `POST /audit` - Run bias audit on dataset
- `GET /audit/{job_id}` - Get audit results

### Explain
- `GET /explain/{job_id}` - Get SHAP explanations

### Remediate
- `POST /remediate` - Apply remediation techniques
- `GET /remediate/{job_id}` - Get remediation results

### Report
- `GET /report/{job_id}` - Generate PDF report

### Health
- `GET /health` - API health check

## Usage Workflow

1. **Landing Page**: Introduce users to FairMind and its capabilities
2. **Upload Wizard**: 
   - Select and preview CSV dataset
   - Configure audit parameters (protected attributes, target variable)
   - Upload for processing
3. **Dashboard**:
   - View bias audit results with group-level metrics
   - Examine feature importance via SHAP explanations
   - Review remediation options and recommendations
   - Access AI-powered insights via Gemma
   - Download PDF report

## Testing

### Backend Tests

```bash
cd backend
pytest tests/ -v
```

Test files:
- `test_api.py` - API endpoint tests
- `test_audit.py` - Audit engine tests
- `test_remediate.py` - Remediation engine tests
- `test_report.py` - Report generation tests

### Frontend Tests

```bash
cd frontend
npm run lint
```

## Configuration

### Backend Configuration
- **Database**: Configured in `backend/app/database.py`
- **CORS**: Configured in `backend/app/main.py` (allows localhost:5173 and fairlens.vercel.app)
- **Logging**: Configured with Python logging module

### Frontend Configuration
- **API Base URL**: Configured in `frontend/src/api/client.js`
- **Build**: Vite configuration in `frontend/vite.config.js`
- **Styling**: Tailwind configuration in `frontend/tailwind.config.js`

## Sample Datasets

Pre-loaded sample datasets available in `backend/datasets/`:
- `adult.csv` - UCI Adult dataset for income prediction
- `compas.csv` - COMPAS recidivism dataset

## Key Concepts

### Fairness Metrics
- **Equalized Odds Difference**: Measures difference in false positive/negative rates across groups
- **Demographic Parity**: Ensures positive outcome rates are equal across groups
- **Predictive Parity**: Ensures precision is equal across groups

### Remediation Techniques
- Threshold optimization for fairness constraints
- Reweighting samples to improve fairness
- Fairness-aware model training

### Explainability
- SHAP (SHapley Additive exPlanations) values for feature importance
- Per-group feature analysis to identify bias drivers

## Development Roadmap

Future enhancements:
- Support for multi-class classification
- Advanced remediation algorithms
- Custom fairness metric support
- Model comparison and benchmarking
- Audit scheduling and monitoring
- Batch processing capabilities

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests: `pytest tests/ -v` (backend), `npm run lint` (frontend)
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Support

For issues, questions, or feedback, please open an issue on GitHub or contact the development team.

## Acknowledgments

- **Fairlearn**: Microsoft's fairness toolkit
- **AIF360**: IBM's AI Fairness 360 toolkit
- **SHAP**: Lundberg & Lee's explanation framework
- **Google Generative AI**: Gemma model for insights
