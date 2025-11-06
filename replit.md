# Stock Market Closing Price Prediction System

## Overview

This is a production-ready stock market closing price prediction system built with Streamlit. The application provides an interactive web interface for analyzing stock market data and predicting closing prices using various machine learning models.

**Key Features:**
- 7 ML/Deep Learning models: Linear Regression, Random Forest, SVR, MLP, XGBoost, LightGBM, LSTM
- Flexible data upload (CSV/TXT with automatic column detection)
- Customizable hyperparameters for all model types
- Advanced feature engineering (technical indicators: RSI, MACD, Bollinger Bands, moving averages)
- Multi-model comparison dashboard
- PostgreSQL database persistence for training history
- Comprehensive export functionality (CSV, Excel, PDF reports)
- Interactive visualizations with Plotly

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Updates (November 2025)

### Database Integration
- Created PostgreSQL `prediction_history` table for persistent storage
- Automatic saving of all training runs with metrics, predictions, and feature importance
- Historical trend analysis and querying capabilities

### Model Parameter Customization
- All 7 models now support hyperparameter tuning through UI
- Advanced parameter sections with expand/collapse functionality
- Parameters saved with each training run for reproducibility

### Multi-Model Comparison
- Sequential training workflow allowing users to train multiple models
- Comparison page showing side-by-side performance metrics
- Visual comparisons with bar charts and tables
- Each model type stores latest result in session state for quick comparison
- Full historical data accessible in database and History page

### Feature Engineering
- Technical indicators: RSI, MACD, Bollinger Bands, ATR, Stochastic Oscillator
- Moving averages: SMA, EMA with customizable periods
- Momentum and volatility metrics
- Feature engineering page for indicator calculation

### Export Capabilities
- CSV: Raw prediction data export
- Excel: Multi-sheet reports with metrics and predictions
- PDF: Professional reports with ReportLab (charts, tables, metadata)

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit (Python-based web framework)
- **Layout**: Wide layout configuration for better data visualization
- **State Management**: Streamlit session state for persisting user data and selections across reruns
  - Stores uploaded data, selected prediction columns, close price column, manual override flags, and model training status
- **Visualization Libraries**: 
  - Plotly (Express and Graph Objects) for interactive charts
  - Matplotlib and Seaborn for statistical plots
  - Supports subplots and complex visualizations

### Backend Architecture
- **Language**: Python 3.x
- **Data Processing**: Pandas and NumPy for data manipulation and numerical operations
- **Statistical Analysis**: SciPy for statistical computations
- **Machine Learning Pipeline**:
  - Scikit-learn for preprocessing (StandardScaler), model selection (train_test_split), and traditional ML models
  - Multiple model types supported: Linear Regression, Random Forest, SVR, MLP Neural Networks
  - Ensemble methods: XGBoost and LightGBM (with fallback handling)
  - Deep Learning: TensorFlow/Keras for neural network models
  - Performance metrics: MAE, MSE, RMSE, R² score

### Design Patterns
- **Graceful Degradation**: LightGBM is treated as an optional dependency with fallback handling
  - Sets `LIGHTGBM_AVAILABLE` flag to conditionally enable features
  - Prevents application crashes if LightGBM cannot be imported
- **Session State Pattern**: Persistent state management across page interactions
- **Model Factory Pattern**: Multiple ML models can be trained and compared within the same framework
- **Warning Suppression**: Filters out non-critical warnings for cleaner user experience

### Data Flow
1. User uploads stock market data (CSV format expected)
2. Data stored in session state for persistence
3. User selects features and target variable (closing price)
4. Data is split into training and testing sets
5. Multiple models are trained and evaluated
6. Results are visualized and compared
7. Predictions can be generated for new data

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework for building the interactive UI
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Scientific computing and statistical analysis

### Visualization
- **plotly**: Interactive plotting library (both express and graph_objects modules)
- **matplotlib**: Static plotting
- **seaborn**: Statistical data visualization

### Machine Learning
- **scikit-learn**: Traditional ML algorithms and preprocessing utilities
  - LinearRegression, RandomForestRegressor, SVR, MLPRegressor
  - StandardScaler for feature scaling
  - train_test_split for data splitting
  - Evaluation metrics (MAE, MSE, R²)
- **xgboost**: Gradient boosting framework
- **lightgbm**: Light Gradient Boosting Machine (optional, with error handling)
- **tensorflow/keras**: Deep learning framework for neural networks

### Development
- **warnings**: Python standard library for managing warning filters

### Database & Persistence
- **PostgreSQL**: Production database using Replit's built-in Postgres (Neon-backed)
- **SQLAlchemy**: Database ORM for connection management and queries
- **psycopg2-binary**: PostgreSQL adapter for Python
- Schema: `prediction_history` table stores all training runs with:
  - Model metadata (type, parameters, features used)
  - Performance metrics (MAE, RMSE, R²)
  - Predictions and feature importance
  - Dataset information and timestamps

### Export & Reporting
- **openpyxl**: Excel file generation for multi-sheet reports
- **reportlab**: PDF report generation with professional formatting

### Notes on Dependencies
- LightGBM is implemented with optional import handling due to potential installation issues across different platforms
- All other ML libraries are required dependencies
- Database integration uses environment variables (DATABASE_URL) for connection
- No external API integrations identified
- Application works with locally uploaded CSV/TXT files