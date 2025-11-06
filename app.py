import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import os
import json
from datetime import datetime
import io
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch
import sqlalchemy
from sqlalchemy import create_engine, text

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError):
    LIGHTGBM_AVAILABLE = False
    lgb = None

# 設定環境變數以隱藏 TensorFlow GPU 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只顯示錯誤訊息
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用 GPU

from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# 資料庫連接
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
else:
    engine = None

# 設定頁面配置
st.set_page_config(
    page_title="股市收盤價預測系統",
    page_icon="📈",
    layout="wide"
)

# 初始化 session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'prediction_cols' not in st.session_state:
    st.session_state.prediction_cols = []
if 'close_col' not in st.session_state:
    st.session_state.close_col = None
if 'manual_override' not in st.session_state:
    st.session_state.manual_override = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'feature_engineered_data' not in st.session_state:
    st.session_state.feature_engineered_data = None

# 標題
st.title("📈 股市收盤價預測系統")
st.markdown("---")

# 側邊欄
with st.sidebar:
    st.header("📁 數據上傳")
    uploaded_file = st.file_uploader(
        "上傳您的數據檔案 (CSV 或 TXT)",
        type=['csv', 'txt'],
        help="檔案格式：日期,時間,開盤價,最高價,最低價,收盤價,成交量,預測1,預測2,...,預測10"
    )
    
    st.markdown("---")
    st.header("📊 功能選單")
    page = st.radio(
        "選擇功能",
        ["數據分析與模型訓練", "模型比較", "歷史記錄", "特徵工程", "匯出報告"],
        index=0
    )
    
    st.markdown("---")
    st.header("⚙️ 系統設定")

# 函數：讀取並解析數據
def load_data(file):
    """智慧型讀取數據並自動識別欄位"""
    # 嘗試多種編碼方式（包含 UTF-16 用於處理 Excel 匯出的文字檔）
    encodings = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'gbk', 'big5', 'gb2312', 'latin-1', 'iso-8859-1', 'cp1252']
    
    df = None
    last_error = None
    
    for encoding in encodings:
        try:
            file.seek(0)
            df = pd.read_csv(file, sep=None, engine='python', encoding=encoding)
            
            # 驗證數據是否為數值型（除了日期/時間欄位）
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                # 嘗試轉換可能的數值欄位
                for col in df.columns:
                    if col not in ['日期', 'date', 'Date', '時間', 'time', 'Time']:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except:
                            pass
            
            # 成功讀取，返回數據
            return df
            
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue
    
    # 如果所有編碼都失敗，顯示錯誤
    if df is None:
        st.error(f"無法讀取檔案。已嘗試多種編碼方式但都失敗。最後錯誤：{str(last_error)}")
        st.info("建議：請確保您的檔案是標準的 CSV 或 TXT 格式，且包含數值數據。")
        return None
    
    return df

# 函數：智慧識別欄位
def identify_columns(df):
    """自動識別欄位結構並找出預測欄位"""
    columns = df.columns.tolist()
    
    # 基本欄位名稱（可能的變體）
    basic_cols = {
        'date': ['日期', 'date', 'Date', 'DATE', '時間'],
        'time': ['時間', 'time', 'Time', 'TIME'],
        'open': ['開盤價', 'open', 'Open', 'OPEN', '開盤'],
        'high': ['最高價', 'high', 'High', 'HIGH', '最高'],
        'low': ['最低價', 'low', 'Low', 'LOW', '最低'],
        'close': ['收盤價', 'close', 'Close', 'CLOSE', '收盤'],
        'volume': ['成交量', 'volume', 'Volume', 'VOLUME', '成交']
    }
    
    # 找出預測欄位
    prediction_cols = []
    for col in columns:
        # 檢查是否包含"預測"或"prediction"
        if '預測' in str(col) or 'prediction' in str(col).lower() or 'pred' in str(col).lower():
            prediction_cols.append(col)
    
    # 如果沒有明確的預測欄位，嘗試找數字編號的欄位
    if not prediction_cols:
        for col in columns:
            # 檢查是否為 預測1, 預測2 這類格式
            if any(char.isdigit() for char in str(col)) and '預測' in str(col):
                prediction_cols.append(col)
    
    return prediction_cols

# 函數：計算相關性
def calculate_correlations(df, target_col, feature_cols):
    """計算特徵與目標變數的相關性"""
    correlations = {}
    p_values = {}
    
    for col in feature_cols:
        if col in df.columns:
            # 移除缺失值
            valid_data = df[[col, target_col]].dropna()
            if len(valid_data) > 2:
                corr, p_val = stats.pearsonr(valid_data[col], valid_data[target_col])
                correlations[col] = corr
                p_values[col] = p_val
    
    return correlations, p_values

# 函數：創建時序窗口數據（用於 LSTM）
def create_sequences(X, y, time_steps=5):
    """創建 LSTM 時序窗口數據"""
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i + time_steps)])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

# 函數：計算技術指標
def calculate_technical_indicators(df, close_col, open_col=None, high_col=None, low_col=None):
    """計算技術指標並添加到數據框"""
    df_enhanced = df.copy()
    
    if close_col not in df.columns:
        return df_enhanced
    
    # 簡單移動平均 (SMA)
    for window in [5, 10, 20]:
        df_enhanced[f'SMA_{window}'] = df[close_col].rolling(window=window).mean()
    
    # 指數移動平均 (EMA)
    for window in [5, 10, 20]:
        df_enhanced[f'EMA_{window}'] = df[close_col].ewm(span=window, adjust=False).mean()
    
    # 相對強弱指標 (RSI)
    delta = df[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_enhanced['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df[close_col].ewm(span=12, adjust=False).mean()
    exp2 = df[close_col].ewm(span=26, adjust=False).mean()
    df_enhanced['MACD'] = exp1 - exp2
    df_enhanced['MACD_Signal'] = df_enhanced['MACD'].ewm(span=9, adjust=False).mean()
    df_enhanced['MACD_Hist'] = df_enhanced['MACD'] - df_enhanced['MACD_Signal']
    
    # 布林通道 (Bollinger Bands)
    if close_col in df.columns:
        rolling_mean = df[close_col].rolling(window=20).mean()
        rolling_std = df[close_col].rolling(window=20).std()
        df_enhanced['BB_Upper'] = rolling_mean + (rolling_std * 2)
        df_enhanced['BB_Lower'] = rolling_mean - (rolling_std * 2)
        df_enhanced['BB_Width'] = df_enhanced['BB_Upper'] - df_enhanced['BB_Lower']
    
    # 波動率
    df_enhanced['Volatility_10'] = df[close_col].pct_change().rolling(window=10).std()
    df_enhanced['Volatility_20'] = df[close_col].pct_change().rolling(window=20).std()
    
    # 動量指標
    df_enhanced['Momentum_5'] = df[close_col].diff(5)
    df_enhanced['Momentum_10'] = df[close_col].diff(10)
    
    # ROC (變化率)
    df_enhanced['ROC_10'] = ((df[close_col] - df[close_col].shift(10)) / df[close_col].shift(10)) * 100
    
    return df_enhanced

# 函數：保存預測記錄到資料庫
def save_prediction_to_db(model_type, model_params, features_used, test_size, mae, rmse, r2, 
                          dataset_name, dataset_rows, predictions, feature_importance=None):
    """保存預測記錄到資料庫"""
    if engine is None:
        return False
    
    try:
        with engine.connect() as conn:
            query = text("""
                INSERT INTO prediction_history 
                (model_type, model_params, features_used, test_size, mae, rmse, r2_score, 
                 dataset_name, dataset_rows, predictions, feature_importance)
                VALUES 
                (:model_type, :model_params, :features_used, :test_size, :mae, :rmse, :r2_score,
                 :dataset_name, :dataset_rows, :predictions, :feature_importance)
            """)
            
            conn.execute(query, {
                'model_type': model_type,
                'model_params': json.dumps(model_params),
                'features_used': features_used,
                'test_size': test_size,
                'mae': float(mae),
                'rmse': float(rmse),
                'r2_score': float(r2),
                'dataset_name': dataset_name,
                'dataset_rows': int(dataset_rows),
                'predictions': json.dumps(predictions) if predictions else None,
                'feature_importance': json.dumps(feature_importance) if feature_importance else None
            })
            conn.commit()
        return True
    except Exception as e:
        st.warning(f"無法保存到資料庫：{str(e)}")
        return False

# 函數：從資料庫獲取歷史記錄
def get_prediction_history(limit=10):
    """從資料庫獲取歷史記錄"""
    if engine is None:
        return None
    
    try:
        query = text("""
            SELECT * FROM prediction_history 
            ORDER BY created_at DESC 
            LIMIT :limit
        """)
        df = pd.read_sql(query, engine, params={'limit': limit})
        return df
    except Exception as e:
        st.warning(f"無法讀取歷史記錄：{str(e)}")
        return None

# 函數：生成 PDF 報告
def generate_pdf_report(model_results, correlations, dataset_info):
    """生成 PDF 報告"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # 標題
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=1
    )
    story.append(Paragraph("Stock Market Prediction Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # 報告生成時間
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # 數據集資訊
    story.append(Paragraph("Dataset Information", styles['Heading2']))
    data_table = [
        ['Item', 'Value'],
        ['Dataset Name', dataset_info.get('name', 'N/A')],
        ['Total Rows', str(dataset_info.get('rows', 'N/A'))],
        ['Features Used', str(dataset_info.get('features', 'N/A'))]
    ]
    t = Table(data_table, colWidths=[2*inch, 3*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*inch))
    
    # 模型效能比較
    if model_results:
        story.append(Paragraph("Model Performance Comparison", styles['Heading2']))
        perf_data = [['Model', 'MAE', 'RMSE', 'R² Score']]
        for model_name, results in model_results.items():
            perf_data.append([
                model_name,
                f"{results.get('mae', 0):.4f}",
                f"{results.get('rmse', 0):.4f}",
                f"{results.get('r2', 0):.4f}"
            ])
        
        t = Table(perf_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*inch))
    
    # 相關性分析
    if correlations:
        story.append(Paragraph("Correlation Analysis", styles['Heading2']))
        corr_data = [['Feature', 'Correlation']]
        for feature, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
            corr_data.append([feature, f"{corr:.4f}"])
        
        t = Table(corr_data, colWidths=[3*inch, 2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# 主要應用程式邏輯
if uploaded_file is not None:
    # 讀取數據
    df = load_data(uploaded_file)
    
    if df is not None:
        st.session_state.data = df
        
        # 驗證已保存的欄位是否仍然存在於新數據中
        if st.session_state.manual_override:
            saved_close = st.session_state.close_col
            saved_predictions = st.session_state.prediction_cols
            
            # 檢查保存的欄位是否存在
            close_exists = saved_close and saved_close in df.columns
            predictions_exist = all(col in df.columns for col in saved_predictions) if saved_predictions else False
            
            # 如果欄位不存在，重置為自動識別
            if not close_exists or not predictions_exist:
                st.warning("⚠️ 檢測到新數據，已重置欄位設定。請重新選擇欄位。")
                st.session_state.manual_override = False
        
        # 智慧識別預測欄位（僅在未手動設定時）
        if not st.session_state.manual_override:
            prediction_cols = identify_columns(df)
            st.session_state.prediction_cols = prediction_cols
            
            # 找出收盤價欄位（在使用前先定義）
            close_col = None
            for col in df.columns:
                if '收盤價' in str(col) or 'close' in str(col).lower():
                    close_col = col
                    break
            st.session_state.close_col = close_col
        else:
            # 使用已儲存的手動設定
            prediction_cols = st.session_state.prediction_cols
            close_col = st.session_state.close_col
        
        # 顯示數據概覽
        st.header("📊 數據概覽")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("總資料筆數", len(df))
        with col2:
            st.metric("總欄位數", len(df.columns))
        with col3:
            st.metric("預測欄位數", len(prediction_cols))
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
            st.metric("缺失值比例", f"{missing_pct:.2f}%")
        
        # 顯示前幾筆資料
        st.subheader("📋 數據預覽")
        st.dataframe(df.head(10), width='stretch')
        
        # 顯示識別到的預測欄位
        if prediction_cols:
            st.success(f"✅ 成功識別 {len(prediction_cols)} 個預測欄位：{', '.join(prediction_cols)}")
        else:
            st.warning("⚠️ 未自動識別到預測欄位，請手動選擇")
            
        # 手動欄位選擇選項
        with st.expander("🔧 手動調整欄位設定"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) == 0:
                st.error("❌ 無可用的數值欄位")
            else:
                # 使用 session_state 中的值作為預設
                default_close_idx = 0
                if close_col and close_col in numeric_cols:
                    default_close_idx = numeric_cols.index(close_col)
                
                manual_close = st.selectbox(
                    "選擇收盤價欄位",
                    options=numeric_cols,
                    index=default_close_idx
                )
                
                manual_predictions = st.multiselect(
                    "選擇預測欄位",
                    options=[col for col in numeric_cols if col != manual_close],
                    default=prediction_cols if prediction_cols else []
                )
                
                if st.button("套用手動設定"):
                    st.session_state.prediction_cols = manual_predictions
                    st.session_state.close_col = manual_close
                    st.session_state.manual_override = True
                    st.success("✅ 已套用手動設定")
                    st.rerun()
        
        # 統計描述
        st.subheader("📈 統計描述")
        st.dataframe(df.describe(), width='stretch')
        
        # 缺失值分析
        st.subheader("🔍 缺失值分析")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            fig = px.bar(
                x=missing_data.index,
                y=missing_data.values,
                labels={'x': 'Column', 'y': 'Missing Count'},
                title='Missing Values by Column'
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.success("✅ 數據完整，無缺失值")
        
        st.markdown("---")
        
        # 相關性分析
        st.header("🔗 相關性分析")
        
        # 驗證欄位存在性
        close_col_valid = close_col and close_col in df.columns
        predictions_valid = prediction_cols and all(col in df.columns for col in prediction_cols)
        
        if close_col_valid and predictions_valid:
            st.subheader(f"預測值與 {close_col} 的相關性")
            
            # 計算相關性
            correlations, p_values = calculate_correlations(df, close_col, prediction_cols)
            
            if correlations:
                # 創建相關性數據框
                corr_df = pd.DataFrame({
                    'Feature': list(correlations.keys()),
                    'Correlation': list(correlations.values()),
                    'P-value': [p_values[k] for k in correlations.keys()]
                })
                corr_df = corr_df.sort_values('Correlation', ascending=False, key=abs)
                
                # 顯示相關性表格
                st.dataframe(corr_df.style.background_gradient(cmap='RdYlGn', subset=['Correlation']), 
                           width='stretch')
                
                # 相關性條形圖
                fig = px.bar(
                    corr_df,
                    x='Feature',
                    y='Correlation',
                    color='Correlation',
                    color_continuous_scale='RdYlGn',
                    title='Correlation with Close Price',
                    labels={'Correlation': 'Pearson Correlation Coefficient'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, width='stretch')
                
                # 相關性熱力圖
                st.subheader("📊 相關性熱力圖")
                
                # 選擇包含收盤價和預測欄位的子集
                heatmap_cols = [close_col] + prediction_cols
                corr_matrix = df[heatmap_cols].corr()
                
                # 使用 Plotly 創建熱力圖
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdYlGn',
                    zmid=0,
                    text=corr_matrix.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 10},
                    colorbar=dict(title="Correlation")
                ))
                fig.update_layout(
                    title='Correlation Heatmap',
                    xaxis_tickangle=-45,
                    height=600
                )
                st.plotly_chart(fig, width='stretch')
        else:
            if not close_col_valid:
                st.error("❌ 未找到收盤價欄位，請使用手動調整欄位設定")
            if not predictions_valid:
                st.error("❌ 未找到預測欄位，請使用手動調整欄位設定")
        
        st.markdown("---")
        
        # 機器學習模型訓練
        st.header("🤖 機器學習模型訓練")
        
        # 使用之前驗證過的欄位
        if close_col_valid and predictions_valid:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("特徵選擇")
                # 讓用戶選擇要使用的預測欄位
                selected_features = st.multiselect(
                    "選擇用於訓練的特徵欄位",
                    options=prediction_cols,
                    default=prediction_cols,
                    help="選擇您想用來預測收盤價的特徵"
                )
                
                # 其他數值欄位也可以作為特徵
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                other_features = [col for col in numeric_cols if col not in prediction_cols and col != close_col]
                
                if other_features:
                    additional_features = st.multiselect(
                        "選擇其他數值特徵（可選）",
                        options=other_features,
                        help="例如：開盤價、最高價、最低價、成交量等"
                    )
                    selected_features.extend(additional_features)
            
            with col2:
                st.subheader("模型選擇")
                
                # 根據 LightGBM 可用性調整模型選項
                model_options = [
                    "Linear Regression",
                    "Random Forest",
                    "XGBoost",
                    "Support Vector Regression (SVR)",
                    "Multi-layer Perceptron (MLP)",
                    "LSTM Neural Network"
                ]
                
                if LIGHTGBM_AVAILABLE:
                    model_options.insert(3, "LightGBM")
                
                model_type = st.selectbox(
                    "選擇機器學習模型",
                    options=model_options,
                    help="訓練結果將自動保存，可在「模型比較」頁面查看所有已訓練模型的比較"
                )
                
                test_size = st.slider(
                    "測試集比例",
                    min_value=0.1,
                    max_value=0.4,
                    value=0.2,
                    step=0.05,
                    help="用於測試的數據比例"
                )
            
            # 超參數調整
            st.markdown("---")
            st.subheader("⚙️ 模型參數調整")
            
            show_params = st.checkbox("顯示進階參數設定", value=False)
            
            model_params = {}
            if show_params:
                if model_type == "Random Forest":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.slider("樹的數量 (n_estimators)", 50, 500, 100, 50)
                    with col2:
                        max_depth = st.slider("最大深度 (max_depth)", 3, 30, 10, 1)
                    with col3:
                        min_samples_split = st.slider("最小分割樣本數", 2, 20, 2, 1)
                    model_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split}
                
                elif model_type == "XGBoost":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.slider("樹的數量", 50, 500, 100, 50)
                    with col2:
                        learning_rate = st.slider("學習率", 0.01, 0.3, 0.1, 0.01)
                    with col3:
                        max_depth = st.slider("最大深度", 3, 15, 6, 1)
                    model_params = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth}
                
                elif model_type == "LightGBM":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.slider("樹的數量", 50, 500, 100, 50)
                    with col2:
                        learning_rate = st.slider("學習率", 0.01, 0.3, 0.1, 0.01)
                    with col3:
                        num_leaves = st.slider("葉子數量", 20, 200, 31, 5)
                    model_params = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'num_leaves': num_leaves}
                
                elif model_type == "Support Vector Regression (SVR)":
                    col1, col2 = st.columns(2)
                    with col1:
                        C = st.slider("懲罰參數 C", 0.1, 10.0, 1.0, 0.1)
                    with col2:
                        kernel = st.selectbox("核函數", ['rbf', 'linear', 'poly'])
                    model_params = {'C': C, 'kernel': kernel}
                
                elif model_type == "Multi-layer Perceptron (MLP)":
                    col1, col2 = st.columns(2)
                    with col1:
                        hidden_layers = st.text_input("隱藏層結構 (逗號分隔)", "100,50")
                        try:
                            hidden_layer_sizes = tuple(map(int, hidden_layers.split(',')))
                        except:
                            hidden_layer_sizes = (100, 50)
                    with col2:
                        max_iter = st.slider("最大迭代次數", 100, 1000, 500, 50)
                    model_params = {'hidden_layer_sizes': hidden_layer_sizes, 'max_iter': max_iter}
                
                elif model_type == "LSTM Neural Network":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        lstm_units = st.slider("LSTM 單元數", 32, 256, 50, 16)
                    with col2:
                        epochs = st.slider("訓練週期", 10, 100, 50, 10)
                    with col3:
                        batch_size = st.slider("批次大小", 8, 64, 32, 8)
                    model_params = {'lstm_units': lstm_units, 'epochs': epochs, 'batch_size': batch_size}
            else:
                # 使用預設參數
                if model_type == "Random Forest":
                    model_params = {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2}
                elif model_type == "XGBoost":
                    model_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6}
                elif model_type == "LightGBM":
                    model_params = {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31}
                elif model_type == "Support Vector Regression (SVR)":
                    model_params = {'C': 1.0, 'kernel': 'rbf'}
                elif model_type == "Multi-layer Perceptron (MLP)":
                    model_params = {'hidden_layer_sizes': (100, 50), 'max_iter': 500}
                elif model_type == "LSTM Neural Network":
                    model_params = {'lstm_units': 50, 'epochs': 50, 'batch_size': 32}
            
            if selected_features and st.button("🚀 訓練模型", type="primary"):
                with st.spinner("正在訓練模型，請稍候..."):
                    try:
                        # 準備數據
                        df_clean = df[selected_features + [close_col]].dropna()
                        
                        if len(df_clean) < 10:
                            st.error("❌ 數據量不足，請確保至少有 10 筆完整數據")
                        else:
                            X = df_clean[selected_features]
                            y = df_clean[close_col]
                            
                            # 對於 LSTM 使用時序分割，其他模型使用隨機分割
                            if model_type == "LSTM Neural Network":
                                # 時序分割（不打亂順序）
                                split_idx = int(len(X) * (1 - test_size))
                                X_train = X.iloc[:split_idx].copy()
                                X_test = X.iloc[split_idx:].copy()
                                y_train = y.iloc[:split_idx].copy()
                                y_test = y.iloc[split_idx:].copy()
                            else:
                                # 隨機分割
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=test_size, random_state=42, shuffle=True
                                )
                            
                            # 標準化
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            
                            # 初始化變量
                            model = None
                            y_pred_train = None
                            y_pred_test = None
                            
                            # 訓練模型（使用自訂參數）
                            if model_type == "Linear Regression":
                                model = LinearRegression()
                                model.fit(X_train_scaled, y_train)
                                y_pred_train = model.predict(X_train_scaled)
                                y_pred_test = model.predict(X_test_scaled)
                            
                            elif model_type == "Random Forest":
                                model = RandomForestRegressor(
                                    n_estimators=model_params.get('n_estimators', 100),
                                    max_depth=model_params.get('max_depth', None),
                                    min_samples_split=model_params.get('min_samples_split', 2),
                                    random_state=42,
                                    n_jobs=-1
                                )
                                model.fit(X_train_scaled, y_train)
                                y_pred_train = model.predict(X_train_scaled)
                                y_pred_test = model.predict(X_test_scaled)
                            
                            elif model_type == "XGBoost":
                                model = xgb.XGBRegressor(
                                    n_estimators=model_params.get('n_estimators', 100),
                                    learning_rate=model_params.get('learning_rate', 0.1),
                                    max_depth=model_params.get('max_depth', 6),
                                    random_state=42,
                                    n_jobs=-1
                                )
                                model.fit(X_train_scaled, y_train)
                                y_pred_train = model.predict(X_train_scaled)
                                y_pred_test = model.predict(X_test_scaled)
                            
                            elif model_type == "LightGBM":
                                if LIGHTGBM_AVAILABLE and lgb is not None:
                                    model = lgb.LGBMRegressor(
                                        n_estimators=model_params.get('n_estimators', 100),
                                        learning_rate=model_params.get('learning_rate', 0.1),
                                        num_leaves=model_params.get('num_leaves', 31),
                                        random_state=42,
                                        n_jobs=-1,
                                        verbose=-1
                                    )
                                    model.fit(X_train_scaled, y_train)
                                    y_pred_train = model.predict(X_train_scaled)
                                    y_pred_test = model.predict(X_test_scaled)
                                else:
                                    st.error("❌ LightGBM 不可用，請選擇其他模型")
                            
                            elif model_type == "Support Vector Regression (SVR)":
                                kernel_value = model_params.get('kernel', 'rbf')
                                if isinstance(kernel_value, int):
                                    kernel_value = 'rbf'
                                model = SVR(
                                    kernel=str(kernel_value),
                                    C=model_params.get('C', 1.0)
                                )
                                model.fit(X_train_scaled, y_train)
                                y_pred_train = model.predict(X_train_scaled)
                                y_pred_test = model.predict(X_test_scaled)
                            
                            elif model_type == "Multi-layer Perceptron (MLP)":
                                model = MLPRegressor(
                                    hidden_layer_sizes=model_params.get('hidden_layer_sizes', (100, 50)),
                                    max_iter=model_params.get('max_iter', 500),
                                    random_state=42
                                )
                                model.fit(X_train_scaled, y_train)
                                y_pred_train = model.predict(X_train_scaled)
                                y_pred_test = model.predict(X_test_scaled)
                            
                            elif model_type == "LSTM Neural Network":
                                # 設定時序窗口大小（確保訓練集和測試集都能產生序列）
                                max_time_steps = min(5, len(X_train_scaled) - 1, len(X_test_scaled) - 1)
                                time_steps = max(1, max_time_steps)  # 至少為 1
                                
                                # 創建時序窗口數據
                                y_train_array = np.array(y_train) if isinstance(y_train, pd.Series) else y_train
                                y_test_array = np.array(y_test) if isinstance(y_test, pd.Series) else y_test
                                X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_array, time_steps)
                                X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_array, time_steps)
                                
                                # 檢查序列是否為空
                                if len(X_train_seq) < 10 or len(X_test_seq) < 1:
                                    st.warning(f"⚠️ LSTM 需要較多數據。當前時序窗口 {time_steps} 步後僅剩 {len(X_train_seq)} 筆訓練數據。使用 MLP 模型代替。")
                                    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
                                    model.fit(X_train_scaled, y_train)
                                    y_pred_train = model.predict(X_train_scaled)
                                    y_pred_test = model.predict(X_test_scaled)
                                else:
                                    # 建立 LSTM 模型（使用自訂參數）
                                    lstm_units = model_params.get('lstm_units', 50)
                                    epochs = model_params.get('epochs', 50)
                                    batch_size = model_params.get('batch_size', 32)
                                    
                                    model = keras.Sequential([
                                        layers.LSTM(lstm_units, activation='relu', input_shape=(time_steps, X_train_scaled.shape[1])),
                                        layers.Dropout(0.2),
                                        layers.Dense(lstm_units // 2, activation='relu'),
                                        layers.Dense(1)
                                    ])
                                    model.compile(optimizer='adam', loss='mse')
                                    model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.1)
                                    
                                    # 預測（需要對完整數據集進行預測以匹配原始 y 的長度）
                                    y_pred_train_seq = model.predict(X_train_seq, verbose=0).flatten()
                                    y_pred_test_seq = model.predict(X_test_seq, verbose=0).flatten()
                                    
                                    # 調整 y 以匹配序列長度
                                    y_train = pd.Series(y_train_seq)
                                    y_test = pd.Series(y_test_seq)
                                    y_pred_train = y_pred_train_seq
                                    y_pred_test = y_pred_test_seq
                            
                            # 檢查預測是否成功
                            if y_pred_train is None or y_pred_test is None or model is None:
                                if model is None:
                                    st.error("❌ 模型初始化失敗")
                                else:
                                    st.error("❌ 模型訓練失敗")
                            else:
                                # 計算評估指標
                                train_mae = mean_absolute_error(y_train, y_pred_train)
                                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                                train_r2 = r2_score(y_train, y_pred_train)
                                
                                test_mae = mean_absolute_error(y_test, y_pred_test)
                                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                                test_r2 = r2_score(y_test, y_pred_test)
                                
                                # 顯示結果
                                st.success("✅ 模型訓練完成！")
                            
                                st.subheader("📊 模型效能評估")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**訓練集表現**")
                                    metrics_train = pd.DataFrame({
                                        'Metric': ['MAE', 'RMSE', 'R² Score'],
                                        'Value': [train_mae, train_rmse, train_r2]
                                    })
                                    st.dataframe(metrics_train, width='stretch', hide_index=True)
                                
                                with col2:
                                    st.markdown("**測試集表現**")
                                    metrics_test = pd.DataFrame({
                                        'Metric': ['MAE', 'RMSE', 'R² Score'],
                                        'Value': [test_mae, test_rmse, test_r2]
                                    })
                                    st.dataframe(metrics_test, width='stretch', hide_index=True)
                                
                                # 預測 vs 實際值圖表
                                st.subheader("📈 預測結果視覺化")
                                
                                # 創建子圖
                                fig = make_subplots(
                                    rows=1, cols=2,
                                    subplot_titles=('Training Set', 'Test Set')
                                )
                                
                                # 確保 y_train 和 y_test 是數組或 Series
                                y_train_array = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
                                y_test_array = np.array(y_test) if not isinstance(y_test, np.ndarray) else y_test
                                y_pred_train_array = np.array(y_pred_train) if not isinstance(y_pred_train, np.ndarray) else y_pred_train
                                y_pred_test_array = np.array(y_pred_test) if not isinstance(y_pred_test, np.ndarray) else y_pred_test
                            
                                # 訓練集
                                fig.add_trace(
                                    go.Scatter(x=y_train_array, y=y_pred_train_array, mode='markers',
                                             name='Train', marker=dict(color='blue', opacity=0.5)),
                                    row=1, col=1
                                )
                                fig.add_trace(
                                    go.Scatter(x=[float(np.min(y_train_array)), float(np.max(y_train_array))],
                                             y=[float(np.min(y_train_array)), float(np.max(y_train_array))],
                                             mode='lines', name='Perfect Prediction',
                                             line=dict(color='red', dash='dash')),
                                    row=1, col=1
                                )
                                
                                # 測試集
                                fig.add_trace(
                                    go.Scatter(x=y_test_array, y=y_pred_test_array, mode='markers',
                                             name='Test', marker=dict(color='green', opacity=0.5)),
                                    row=1, col=2
                                )
                                fig.add_trace(
                                    go.Scatter(x=[float(np.min(y_test_array)), float(np.max(y_test_array))],
                                             y=[float(np.min(y_test_array)), float(np.max(y_test_array))],
                                             mode='lines', name='Perfect Prediction',
                                             line=dict(color='red', dash='dash')),
                                    row=1, col=2
                                )
                                
                                fig.update_xaxes(title_text="Actual Close Price", row=1, col=1)
                                fig.update_xaxes(title_text="Actual Close Price", row=1, col=2)
                                fig.update_yaxes(title_text="Predicted Close Price", row=1, col=1)
                                fig.update_yaxes(title_text="Predicted Close Price", row=1, col=2)
                                
                                fig.update_layout(height=500, showlegend=True)
                                st.plotly_chart(fig, width='stretch')
                            
                                # 特徵重要性（適用於樹模型）
                                if model_type in ["Random Forest", "XGBoost", "LightGBM"]:
                                    st.subheader("🎯 特徵重要性分析")
                                    
                                    if hasattr(model, 'feature_importances_'):
                                        importance_df = pd.DataFrame({
                                            'Feature': selected_features,
                                            'Importance': model.feature_importances_
                                        }).sort_values('Importance', ascending=False)
                                        
                                        fig = px.bar(
                                            importance_df,
                                            x='Importance',
                                            y='Feature',
                                            orientation='h',
                                            title='Feature Importance',
                                            labels={'Importance': 'Importance Score'}
                                        )
                                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                                        st.plotly_chart(fig, width='stretch')
                                
                                # 儲存模型到 session state
                                st.session_state.model_trained = True
                                st.session_state.model = model
                                st.session_state.scaler = scaler
                                st.session_state.selected_features = selected_features
                                st.session_state.model_type = model_type
                                
                                # 準備數據保存到資料庫
                                feature_importance_dict = None
                                if model_type in ["Random Forest", "XGBoost", "LightGBM"] and hasattr(model, 'feature_importances_'):
                                    feature_importance_dict = {
                                        feat: float(imp) for feat, imp in zip(selected_features, model.feature_importances_)
                                    }
                                
                                predictions_list = y_pred_test_array.tolist() if hasattr(y_pred_test_array, 'tolist') else list(y_pred_test_array)
                                
                                # 保存到資料庫
                                save_result = save_prediction_to_db(
                                    model_type=model_type,
                                    model_params={'test_size': test_size},
                                    features_used=selected_features,
                                    test_size=test_size,
                                    mae=test_mae,
                                    rmse=test_rmse,
                                    r2=test_r2,
                                    dataset_name=uploaded_file.name if uploaded_file else 'Unknown',
                                    dataset_rows=len(df_clean),
                                    predictions=predictions_list[:100],  # 只保存前100個預測
                                    feature_importance=feature_importance_dict
                                )
                                
                                if save_result:
                                    st.success("✅ 預測記錄已保存到歷史資料庫")
                                
                                # 也保存到 model_results 用於比較
                                st.session_state.model_results[model_type] = {
                                    'mae': test_mae,
                                    'rmse': test_rmse,
                                    'r2': test_r2,
                                    'predictions': y_pred_test,
                                    'y_test': y_test,
                                    'feature_importance': feature_importance_dict
                                }
                            
                    except Exception as e:
                        st.error(f"❌ 訓練模型時發生錯誤：{str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        else:
            if not close_col_valid:
                st.error("❌ 未找到收盤價欄位，請使用手動調整欄位設定")
            if not predictions_valid:
                st.error("❌ 未找到預測欄位，請使用手動調整欄位設定")
        
        # === 新增功能區段 ===
        if page == "模型比較" and st.session_state.model_results:
            st.markdown("---")
            st.header("📊 模型效能比較")
            
            # 顯示所有已訓練模型的比較
            if len(st.session_state.model_results) > 0:
                st.subheader("效能指標比較")
                
                # 創建比較表格
                comparison_data = []
                for model_name, results in st.session_state.model_results.items():
                    comparison_data.append({
                        'Model': model_name,
                        'MAE': results['mae'],
                        'RMSE': results['rmse'],
                        'R² Score': results['r2']
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['R² Score'], color='lightgreen')
                            .highlight_min(axis=0, subset=['MAE', 'RMSE'], color='lightgreen'))
                
                # 視覺化比較
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure()
                    for metric in ['MAE', 'RMSE']:
                        fig.add_trace(go.Bar(
                            name=metric,
                            x=comparison_df['Model'],
                            y=comparison_df[metric],
                        ))
                    fig.update_layout(
                        title='Error Metrics Comparison',
                        barmode='group',
                        xaxis_title='Model',
                        yaxis_title='Error Value'
                    )
                    st.plotly_chart(fig, width='stretch')
                
                with col2:
                    fig = px.bar(
                        comparison_df,
                        x='Model',
                        y='R² Score',
                        title='R² Score Comparison',
                        labels={'R² Score': 'R² Score'}
                    )
                    st.plotly_chart(fig, width='stretch')
            else:
                st.info("請先訓練至少一個模型")
        
        elif page == "歷史記錄":
            st.markdown("---")
            st.header("📜 歷史預測記錄")
            
            if engine:
                history_df = get_prediction_history(limit=20)
                
                if history_df is not None and len(history_df) > 0:
                    st.subheader(f"最近 {len(history_df)} 筆記錄")
                    
                    # 顯示歷史記錄
                    display_cols = ['created_at', 'model_type', 'mae', 'rmse', 'r2_score', 'dataset_name', 'dataset_rows']
                    display_df = history_df[display_cols].copy()
                    display_df.columns = ['時間', '模型類型', 'MAE', 'RMSE', 'R² Score', '數據集', '數據量']
                    st.dataframe(display_df, width='stretch')
                    
                    # 效能趨勢圖
                    st.subheader("📈 效能趨勢分析")
                    
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('MAE Trend', 'R² Score Trend')
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=history_df['created_at'], y=history_df['mae'], 
                                 mode='lines+markers', name='MAE'),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=history_df['created_at'], y=history_df['r2_score'], 
                                 mode='lines+markers', name='R² Score'),
                        row=1, col=2
                    )
                    
                    fig.update_xaxes(title_text="Time", row=1, col=1)
                    fig.update_xaxes(title_text="Time", row=1, col=2)
                    fig.update_yaxes(title_text="MAE", row=1, col=1)
                    fig.update_yaxes(title_text="R² Score", row=1, col=2)
                    
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("暫無歷史記錄")
            else:
                st.warning("資料庫未連接")
        
        elif page == "特徵工程":
            st.markdown("---")
            st.header("⚙️ 進階特徵工程")
            
            if close_col_valid:
                st.subheader("技術指標計算")
                
                enable_features = st.checkbox("啟用技術指標特徵", value=False)
                
                if enable_features:
                    with st.spinner("正在計算技術指標..."):
                        # 計算技術指標
                        df_enhanced = calculate_technical_indicators(df, close_col)
                        st.session_state.feature_engineered_data = df_enhanced
                        
                        # 顯示新增的特徵
                        new_features = [col for col in df_enhanced.columns if col not in df.columns]
                        
                        if new_features:
                            st.success(f"✅ 成功生成 {len(new_features)} 個技術指標特徵")
                            
                            # 顯示新特徵列表
                            st.subheader("新增特徵列表")
                            col1, col2, col3 = st.columns(3)
                            
                            features_by_category = {
                                '移動平均': [f for f in new_features if 'SMA' in f or 'EMA' in f],
                                '技術指標': [f for f in new_features if any(x in f for x in ['RSI', 'MACD', 'BB'])],
                                '其他指標': [f for f in new_features if f not in [f for f in new_features if 'SMA' in f or 'EMA' in f or any(x in f for x in ['RSI', 'MACD', 'BB'])]]
                            }
                            
                            with col1:
                                st.write("**移動平均**")
                                for f in features_by_category['移動平均']:
                                    st.write(f"- {f}")
                            
                            with col2:
                                st.write("**技術指標**")
                                for f in features_by_category['技術指標']:
                                    st.write(f"- {f}")
                            
                            with col3:
                                st.write("**其他指標**")
                                for f in features_by_category['其他指標']:
                                    st.write(f"- {f}")
                            
                            # 顯示增強後的數據預覽
                            st.subheader("增強後的數據預覽")
                            st.dataframe(df_enhanced.tail(10), width='stretch')
                            
                            st.info("💡 提示：您現在可以在模型訓練中使用這些新特徵")
                else:
                    st.info("勾選上方選項以啟用技術指標計算")
            else:
                st.error("❌ 需要有效的收盤價欄位才能計算技術指標")
        
        elif page == "匯出報告":
            st.markdown("---")
            st.header("📥 匯出預測報告")
            
            if st.session_state.model_results or (close_col_valid and predictions_valid):
                st.subheader("選擇匯出格式")
                
                export_format = st.radio(
                    "報告格式",
                    ["CSV", "Excel", "PDF"],
                    horizontal=True
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    include_correlations = st.checkbox("包含相關性分析", value=True)
                
                with col2:
                    include_predictions = st.checkbox("包含預測結果", value=True)
                
                if st.button("🎯 生成報告", type="primary"):
                    with st.spinner("正在生成報告..."):
                        try:
                            if export_format == "CSV":
                                # 生成 CSV
                                buffer = io.BytesIO()
                                
                                if st.session_state.model_results:
                                    # 匯出模型比較結果
                                    comparison_data = []
                                    for model_name, results in st.session_state.model_results.items():
                                        comparison_data.append({
                                            'Model': model_name,
                                            'MAE': results['mae'],
                                            'RMSE': results['rmse'],
                                            'R² Score': results['r2']
                                        })
                                    comparison_df = pd.DataFrame(comparison_data)
                                    csv_data = comparison_df.to_csv(index=False)
                                    buffer.write(csv_data.encode())
                                    buffer.seek(0)
                                    
                                    st.download_button(
                                        label="📥 下載 CSV 報告",
                                        data=buffer,
                                        file_name=f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                                    st.success("✅ CSV 報告已生成")
                            
                            elif export_format == "Excel":
                                # 生成 Excel
                                buffer = io.BytesIO()
                                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                    # 模型比較結果
                                    if st.session_state.model_results:
                                        comparison_data = []
                                        for model_name, results in st.session_state.model_results.items():
                                            comparison_data.append({
                                                'Model': model_name,
                                                'MAE': results['mae'],
                                                'RMSE': results['rmse'],
                                                'R² Score': results['r2']
                                            })
                                        comparison_df = pd.DataFrame(comparison_data)
                                        comparison_df.to_excel(writer, sheet_name='Model Comparison', index=False)
                                    
                                    # 相關性分析
                                    if include_correlations and close_col_valid and predictions_valid:
                                        correlations, p_values = calculate_correlations(df, close_col, prediction_cols)
                                        corr_df = pd.DataFrame({
                                            'Feature': list(correlations.keys()),
                                            'Correlation': list(correlations.values()),
                                            'P-Value': [p_values.get(k, None) for k in correlations.keys()]
                                        })
                                        corr_df.to_excel(writer, sheet_name='Correlations', index=False)
                                    
                                    # 數據統計
                                    df.describe().to_excel(writer, sheet_name='Data Statistics')
                                
                                buffer.seek(0)
                                
                                st.download_button(
                                    label="📥 下載 Excel 報告",
                                    data=buffer,
                                    file_name=f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                                st.success("✅ Excel 報告已生成")
                            
                            elif export_format == "PDF":
                                # 生成 PDF
                                correlations_dict = None
                                if include_correlations and close_col_valid and predictions_valid:
                                    correlations_dict, _ = calculate_correlations(df, close_col, prediction_cols)
                                
                                dataset_info = {
                                    'name': uploaded_file.name if uploaded_file else 'Unknown',
                                    'rows': len(df),
                                    'features': len(prediction_cols) if predictions_valid else 0
                                }
                                
                                pdf_buffer = generate_pdf_report(
                                    st.session_state.model_results,
                                    correlations_dict,
                                    dataset_info
                                )
                                
                                st.download_button(
                                    label="📥 下載 PDF 報告",
                                    data=pdf_buffer,
                                    file_name=f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf"
                                )
                                st.success("✅ PDF 報告已生成")
                        
                        except Exception as e:
                            st.error(f"❌ 生成報告時發生錯誤：{str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
            else:
                st.info("請先訓練模型或確保有有效的數據")

else:
    # 首頁說明
    st.info("👈 請從左側欄上傳您的數據檔案開始使用")
    
    st.markdown("""
    ### 📖 使用說明
    
    1. **上傳數據檔案**：支援 CSV 或 TXT 格式
    2. **數據格式**：日期,時間,開盤價,最高價,最低價,收盤價,成交量,預測1,預測2,...,預測10
    3. **自動識別**：系統會自動識別您的預測欄位（預測1到預測10）
    4. **相關性分析**：查看每個預測值與收盤價的相關性
    5. **模型訓練**：選擇機器學習或神經網路模型進行訓練
    6. **效能評估**：檢視模型的預測效能指標
    
    ### 🎯 支援的模型
    
    - **傳統機器學習**：線性回歸、隨機森林、XGBoost、LightGBM、SVR
    - **神經網路**：多層感知機 (MLP)、長短期記憶網路 (LSTM)
    
    ### 📊 分析功能
    
    - 數據統計描述
    - 缺失值分析
    - 皮爾森相關性分析
    - 相關性熱力圖
    - 特徵重要性分析
    - 預測結果視覺化
    """)

# 頁尾
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Stock Price Prediction System | Powered by Streamlit</div>",
    unsafe_allow_html=True
)
