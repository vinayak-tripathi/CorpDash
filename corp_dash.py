"""
Corporate Events Dashboard - Optimized Version
Fetches and displays corporate announcements, board meetings, and corporate actions
from NSE and BSE exchanges with enhanced filtering and visualization.
"""

import streamlit as st
import pandas as pd
import json
import re
import urllib.parse
import os
import requests
from datetime import datetime, date, timedelta
from typing import Tuple, List, Optional
import logging

# Third-party imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from zipfile import ZipFile
from io import BytesIO
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ================================================================================
# CONFIGURATION & CONSTANTS
# ================================================================================

# ADMIN_SECRET = os.getenv("ADMIN_SECRET_CORPORATE_DASHBOARD")
# st.write(ADMIN_SECRET)

# Database configuration

DB_USER = st.secrets["database"]["user"]
DB_PASSWORD = st.secrets["database"]["password"]
DB_HOST = st.secrets["database"]["host"]
DB_PORT = st.secrets["database"]["port"]
DB_NAME = st.secrets["database"]["database"]

INDEX_URLS = {
    'NIFTY 50': 'https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv',
    'NIFTY BANK': 'https://www.niftyindices.com/IndexConstituent/ind_niftybanklist.csv',
    'NIFTY Next 50': 'https://www.niftyindices.com/IndexConstituent/ind_niftynext50list.csv',
    'NIFTY MIDCAP SELECT': 'https://www.niftyindices.com/IndexConstituent/ind_niftymidcapselect_list.csv',
    'NIFTY FINANCE': 'https://www.niftyindices.com/IndexConstituent/ind_niftyfinancelist.csv',
}



# Construct the database URI
DATABASE_URI = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# Create database engine
engine: Engine = create_engine(
    DATABASE_URI
    # connect_args={'host': DATABASE_CONFIG['host'], 'port': DATABASE_CONFIG['port']}
)

# File paths (keeping for potential future use)
CORP_ANN_FILE = st.secrets["paths"]["corp_ann_file"]
CORP_BM_FILE = st.secrets["paths"]["corp_bm_file"]
CORP_ACT_FILE = st.secrets["paths"]["corp_act_file"]

# API URLs
NSE_BASE_URL = st.secrets["urls"]["nse_base_url"]
NSE_CORP_ANN_URL = st.secrets["urls"]["nse_corp_ann_url"]
NSE_BOARD_MEET_URL = st.secrets["urls"]["nse_board_meet_url"]
BSE_API_URL = st.secrets["urls"]["bse_api_url"]
NSE_FNO_URL = st.secrets["urls"]["nse_fno_url"]


# Default date ranges
DEFAULT_LOOKBACK_DAYS = 8
DEFAULT_LOOKAHEAD_DAYS = 90
DEFAULT_FNO_LOOKBACK_DAYS = 5

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================================
# DATA FETCHING FUNCTIONS
# ================================================================================

def get_selenium_driver() -> webdriver.Chrome:
    """Initialize and return a configured Chrome WebDriver."""
    try:
        chromedriver_path = ChromeDriverManager().install()
        if not chromedriver_path.endswith("chromedriver.exe"):
            chromedriver_dir = os.path.dirname(chromedriver_path)
            chromedriver_path = os.path.join(chromedriver_dir, "chromedriver.exe")

        service = Service(chromedriver_path)
        options = webdriver.ChromeOptions()
        
        # Configure Chrome options for better performance
        options.add_argument("--headless")  # Run in headless mode for better performance
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option('prefs', {
            "download.prompt_for_download": False,
            "plugins.always_open_pdf_externally": True,
        })
        
        return webdriver.Chrome(service=service, options=options)
    except Exception as e:
        logger.error(f"Failed to initialize WebDriver: {e}")
        raise


def fetch_nse_announcements() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch NSE announcements and board meetings data."""
    driver = None
    try:
        driver = get_selenium_driver()
        
        # Navigate to NSE corporate filings page
        driver.get(f"{NSE_BASE_URL}/companies-listing/corporate-filings-announcements")
        wait = WebDriverWait(driver, 10)

        # Fetch Announcements
        start_date = (date.today() - timedelta(days=DEFAULT_LOOKBACK_DAYS)).strftime("%d-%m-%Y")
        end_date = date.today().strftime("%d-%m-%Y")
        
        ann_params = {
            "index": "equities",
            "symbol": "",
            "subject": "",
            "from_date": start_date,
            "to_date": end_date,
            "fo_sec": ""
        }
        
        driver.get(f"{NSE_CORP_ANN_URL}?{urllib.parse.urlencode(ann_params)}")
        ann_data = json.loads(driver.find_element(By.TAG_NAME, "pre").text)
        
        df_ann = pd.DataFrame(ann_data)[['symbol', 'desc', 'an_dt', 'attchmntText', 'attchmntFile']]
        df_ann['an_dt'] = pd.to_datetime(df_ann['an_dt'])
        df_ann.rename({'desc': 'description'}, axis=1, inplace=True)

        # Fetch Board Meetings
        driver.get(NSE_BASE_URL)  # Reset session
        
        bm_start_date = date.today().strftime("%d-%m-%Y")
        bm_end_date = (date.today() + timedelta(days=DEFAULT_LOOKAHEAD_DAYS)).strftime("%d-%m-%Y")
        
        bm_params = {
            "index": "equities",
            "symbol": "",
            "subject": "",
            "from_date": bm_start_date,
            "to_date": bm_end_date,
            "fo_sec": ""
        }
        
        driver.get(f"{NSE_BOARD_MEET_URL}?{urllib.parse.urlencode(bm_params)}")
        bm_data = json.loads(driver.find_element(By.TAG_NAME, "pre").text)
        
        df_bm = pd.DataFrame(bm_data)[['bm_symbol', 'bm_date', 'bm_purpose', 'bm_desc', 'bm_timestamp', 'attachment']]
        df_bm['bm_date'] = pd.to_datetime(df_bm['bm_date'])
        df_bm['bm_timestamp'] = pd.to_datetime(df_bm['bm_timestamp'])
        
        logger.info(f"Fetched {len(df_ann)} announcements and {len(df_bm)} board meetings from NSE")
        return df_ann, df_bm
        
    except Exception as e:
        logger.error(f"Error fetching NSE data: {e}")
        return pd.DataFrame(), pd.DataFrame()
    finally:
        if driver:
            driver.quit()


def fetch_bse_corp_actions() -> pd.DataFrame:
    """Fetch BSE corporate actions data using requests."""
    try:
        session = requests.Session()
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.7",
            "Origin": "https://www.bseindia.com",
            "Referer": "https://www.bseindia.com/",
            "Sec-Ch-Ua": '"Brave";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "Sec-Gpc": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        }
        
        # Initialize session
        session.get("https://bseindia.com", headers=headers, timeout=10)
        
        # Fetch corporate actions data
        params = {
            "Fdate": None,
            "Purposecode": None,
            "TDate": None,
            "ddlcategorys": 'E',
            "ddlindustrys": None,
            "scripcode": None,
            "segment": '0',
            "strSearch": 'S',
        }
        
        response = session.get(BSE_API_URL, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = json.loads(response.content)
        df_bse = pd.DataFrame(data)
        
        if not df_bse.empty:
            df_bse['Ex_date'] = pd.to_datetime(df_bse['Ex_date'], errors='coerce')
            df_bse['RD_Date'] = pd.to_datetime(df_bse['RD_Date'], errors='coerce')
            df_bse = df_bse[['short_name', 'Ex_date', 'RD_Date', 'Purpose', 'scrip_code']]
        
        logger.info(f"Fetched {len(df_bse)} corporate actions from BSE")
        return df_bse
        
    except Exception as e:
        logger.error(f"Error fetching BSE data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_index_constituents(index_name: str = None) -> dict:
    """
    Fetch index constituents from NSE.
    
    Args:
        index_name: Specific index name to fetch. If None, fetches all indices.
    
    Returns:
        Dictionary with index names as keys and list of symbols as values
    """
    try:
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'accept-language': 'en,gu;q=0.9,hi;q=0.8',
            'accept-encoding': 'gzip, deflate, br'
        }
        
        session = requests.Session()
        # Initialize session with main page
        session.get("https://www.niftyindices.com", headers=headers, timeout=10)
        
        index_constituents = {}
        
        # If specific index requested, fetch only that
        if index_name and index_name in INDEX_URLS:
            urls_to_fetch = {index_name: INDEX_URLS[index_name]}
        else:
            urls_to_fetch = INDEX_URLS
        
        for idx_name, url in urls_to_fetch.items():
            try:
                response = session.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Parse CSV data
                from io import StringIO
                csv_data = StringIO(response.text)
                df = pd.read_csv(csv_data)
                # st.write(df)
                
                # Extract symbols (assuming 'Symbol' column exists)
                if 'Symbol' in df.columns:
                    symbols = df['Symbol'].dropna().unique().tolist()
                    index_constituents[idx_name] = symbols
                    logger.info(f"Fetched {len(symbols)} stocks for {idx_name}")
                else:
                    logger.warning(f"No Symbol column found for {idx_name}")
                    
            except Exception as e:
                logger.error(f"Error fetching {idx_name}: {e}")
                continue
        
        return index_constituents
        
    except Exception as e:
        logger.error(f"Error fetching index constituents: {e}")
        return {}



@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_fno_stocks() -> List[str]:
    """Fetch list of F&O stocks from NSE."""
    try:
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'accept-language': 'en,gu;q=0.9,hi;q=0.8',
            'accept-encoding': 'gzip, deflate, br'
        }
        
        session = requests.Session()
        session.get(NSE_FNO_URL, headers=headers, timeout=10)
        
        current_date = date.today()
        max_attempts = DEFAULT_FNO_LOOKBACK_DAYS
        
        for _ in range(max_attempts):
            try:
                file_url = f"{NSE_FNO_URL}/content/fo/BhavCopy_NSE_FO_0_0_0_{current_date.strftime('%Y%m%d')}_F_0000.csv.zip"
                response = session.get(file_url, headers=headers, timeout=10)
                response.raise_for_status()
                
                with ZipFile(BytesIO(response.content)) as zip_ref:
                    df_bhavcopy = pd.read_csv(zip_ref.open(zip_ref.namelist()[-1]))
                
                fno_stocks = sorted(df_bhavcopy.loc[df_bhavcopy["FinInstrmTp"] == 'STF', 'TckrSymb'].unique().tolist())
                logger.info(f"Fetched {len(fno_stocks)} F&O stocks")
                return fno_stocks
                
            except requests.exceptions.RequestException:
                current_date -= timedelta(days=1)
                continue
        
        logger.warning("Failed to fetch F&O stocks data")
        return []
        
    except Exception as e:
        logger.error(f"Error fetching F&O stocks: {e}")
        return []

# ================================================================================
# DATABASE FUNCTIONS
# ================================================================================

def insert_ignore_sqlalchemy(df: pd.DataFrame, table_name: str, engine: Engine, unique_cols: List[str]) -> None:
    """Insert data into database with IGNORE clause to avoid duplicates."""
    if df.empty:
        return

    try:
        cols = df.columns.tolist()
        placeholders = ','.join(['%s'] * len(cols))
        columns_str = ','.join(f"`{col}`" for col in cols)
        
        insert_sql = f"INSERT IGNORE INTO `{table_name}` ({columns_str}) VALUES ({placeholders})"
        data = [tuple(x) for x in df.to_numpy()]
        
        with engine.begin() as conn:
            conn.execute(text("SET NAMES utf8mb4;"))
            conn.connection.cursor().executemany(insert_sql, data)
            
        logger.info(f"Inserted {len(data)} records into {table_name}")
        
    except Exception as e:
        logger.error(f"Error inserting data into {table_name}: {e}")


@st.cache_data(show_spinner=False, ttl=300)  # Cache for 5 minutes
def load_data(refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data from database and optionally refresh from external sources."""
    try:
        # Load existing data from database
        bm = pd.read_sql("SELECT * FROM board_meetings", engine, 
                        parse_dates=['bm_date', 'bm_timestamp', 'added'])
        ann = pd.read_sql("SELECT * FROM announcements", engine, 
                         parse_dates=['an_dt', 'added'])
        act = pd.read_sql("SELECT * FROM corp_actions", engine, 
                         parse_dates=['Ex_date', 'RD_Date', 'added'])
        
        if refresh:
            # Fetch fresh data
            ann_new, bm_new = fetch_nse_announcements()
            act_new = fetch_bse_corp_actions()
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for df in [ann_new, bm_new, act_new]:
                if not df.empty:
                    df['added'] = timestamp
            
            # Insert new data
            insert_ignore_sqlalchemy(bm_new, "board_meetings", engine, ["bm_symbol", "bm_date", "bm_purpose"])
            insert_ignore_sqlalchemy(ann_new, "announcements", engine, ["symbol", "an_dt"])
            insert_ignore_sqlalchemy(act_new, "corp_actions", engine, ["scrip_code", "Ex_date", "Purpose"])
            
            # Reload data after refresh
            bm = pd.read_sql("SELECT * FROM board_meetings", engine, 
                            parse_dates=['bm_date', 'bm_timestamp', 'added'])
            ann = pd.read_sql("SELECT * FROM announcements", engine, 
                             parse_dates=['an_dt', 'added'])
            act = pd.read_sql("SELECT * FROM corp_actions", engine, 
                             parse_dates=['Ex_date', 'RD_Date', 'added'])
        
        logger.info(f"Loaded data: {len(ann)} announcements, {len(bm)} board meetings, {len(act)} corporate actions")
        return bm, ann, act
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# ================================================================================
# FILTERING FUNCTIONS
# ================================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_filtered_stocks(index_filter: List[str] = None, fno_only: bool = False) -> List[str]:
    """
    Get filtered list of stocks based on index membership and F&O availability.
    
    Args:
        index_filter: List of selected indices
        fno_only: If True, filter to only F&O stocks
    
    Returns:
        List of filtered stock symbols
    """
    try:
        # Get F&O stocks if needed
        # fno_stocks = set()
        if fno_only and "fno_stocks" not in st.session_state:
            fno_stocks = get_fno_stocks()
        else:
            fno_stocks = st.session_state['fno_stocks']
        
        
        
        # If no index filter, return all stocks (optionally filtered by F&O)
        if not index_filter:
            if fno_only:
                return fno_stocks
            else:
                # Return all unique stocks from data sources
                all_stocks = set()
                for key in ['ann', 'bm', 'bse']:
                    if key in st.session_state:
                        df = st.session_state[key]
                        if key == 'ann':
                            all_stocks.update(df['symbol'].dropna())
                        elif key == 'bm':
                            all_stocks.update(df['bm_symbol'].dropna())
                        elif key == 'bse':
                            all_stocks.update(df['short_name'].dropna())
                return sorted(list(all_stocks))
        
        
        if "index_constituents" not in st.session_state:
            index_constituents = get_index_constituents()
        else:
            index_constituents = st.session_state['index_constituents']

        # Combine stocks from selected indices
        filtered_stocks = set()
        for index_name in index_filter:
            if index_name in index_constituents:
                filtered_stocks.update(index_constituents[index_name])
        
        # Apply F&O filter if requested
        if fno_only:
            filtered_stocks = filtered_stocks.intersection(fno_stocks)
        
        return sorted(list(filtered_stocks))
        
    except Exception as e:
        logger.error(f"Error in get_filtered_stocks: {e}")
        return []

def apply_universal_filter(
    df: pd.DataFrame, 
    stock_filter: List[str], 
    fno_only: bool,
    indices_filter: List[str] = None,
    start_date: Optional[date] = None, 
    end_date: Optional[date] = None,
    keywords: List[str] = None, 
    data_type: str = None
) -> pd.DataFrame:
    """
    Apply comprehensive filters to dataframes based on stocks, dates, and keywords.
    
    Args:
        df: DataFrame to filter
        stock_filter: List of selected stocks
        fno_only: Filter for F&O stocks only
        indices_filter: List of selected indices
        start_date: Start date for filtering
        end_date: End date for filtering
        keywords: List of keywords for text filtering
        data_type: Type of data ('ca', 'ann', or 'bm')
    
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Apply index and F&O filters
    if indices_filter or fno_only:
        filtered_stocks = get_filtered_stocks(indices_filter, fno_only)
        
        if filtered_stocks:
            stock_columns = ['bm_symbol', 'symbol', 'short_name']
            
            for col in stock_columns:
                if col in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[col].isin(filtered_stocks)]
                    break
    
    # Apply specific stock filter
    if stock_filter:
        stock_columns = ['bm_symbol', 'symbol', 'short_name']
        for col in stock_columns:
            if col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[col].isin(stock_filter)]
                break
    
    # Apply date filters based on data type
    if data_type == 'ca':  # Corporate Actions
        if len(stock_filter) <= 1:  # Single stock or no stock selected
            if start_date and end_date:
                filtered_df = filtered_df[
                    (filtered_df['Ex_date'] >= pd.Timestamp(start_date)) &
                    (filtered_df['Ex_date'] <= pd.Timestamp(end_date))
                ]
        else:  # Multiple stocks - show only upcoming
            filtered_df = filtered_df[filtered_df['Ex_date'] >= pd.Timestamp.today()]
            
    elif data_type == 'ann':  # Announcements
        if start_date and end_date:
            filtered_df = filtered_df[
                (filtered_df['an_dt'] >= pd.Timestamp(start_date)) &
                (filtered_df['an_dt'] <= pd.Timestamp(end_date)+timedelta(days=1))
            ]
            
    elif data_type == 'bm':  # Board Meetings
        if start_date:
            filtered_df = filtered_df[filtered_df['bm_date'] >= pd.Timestamp(start_date)]
    
    # Apply keyword filters
    if keywords and any(keywords):
        pattern = '|'.join(re.escape(kw) for kw in keywords if kw.strip())
        
        if 'attchmntText' in filtered_df.columns:  # Announcements
            mask = (
                filtered_df['attchmntText'].str.contains(pattern, case=False, na=False) |
                filtered_df['description'].str.contains(pattern, case=False, na=False)
            )
            filtered_df = filtered_df[mask]
            
        elif 'bm_purpose' in filtered_df.columns:  # Board meetings
            mask = (
                filtered_df['bm_purpose'].str.contains(pattern, case=False, na=False) |
                filtered_df['bm_desc'].str.contains(pattern, case=False, na=False)
            )
            filtered_df = filtered_df[mask]
            
        elif 'Purpose' in filtered_df.columns:  # Corporate actions
            filtered_df = filtered_df[
                filtered_df['Purpose'].str.contains(pattern, case=False, na=False)
            ]
    
    return filtered_df


def is_recent_record(added_timestamp: pd.Timestamp, hours: int = 24) -> bool:
    """Check if a record was added recently."""
    if pd.isna(added_timestamp):
        return False
    return (datetime.now() - pd.to_datetime(added_timestamp)) < timedelta(hours=hours)

# ================================================================================
# STREAMLIT UI COMPONENTS
# ================================================================================

def setup_page_config():
    """Configure Streamlit page settings and styling."""
    st.set_page_config(
        page_title="Corporate Events Dashboard",
        page_icon="📢",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS styling
    st.markdown("""
        <style>
        .stExpander {
            border: 1px solid #e6e6e6;
            border-radius: 8px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stDataFrame {
            font-size: 14px;
        }
        .recent-indicator {
            color: #28a745;
            font-weight: bold;
        }
        .metric-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .filter-section {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)


def create_sidebar_filters():
    """Create and return sidebar filter components with index filtering."""
    st.sidebar.title(" Filters")
    
    # Initialize index constituents in session state
    if "index_constituents" not in st.session_state:
        with st.spinner("Loading index constituents..."):
            st.session_state['index_constituents'] = get_index_constituents()
    
    # Index filter
    st.sidebar.markdown("### 📊 Index Filter")
    available_indices = list(INDEX_URLS.keys())
    selected_indices = st.sidebar.multiselect(
        "Select Indices",
        # default='ALL Stocks',
        options=available_indices,
        help="Filter stocks by index membership"
    )
    
    # F&O filter
    fno_only = st.sidebar.checkbox("F&O Stocks Only", value=True, 
                                  help="Filter to show only Futures & Options stocks")
    
    # Get filtered stock options
    stock_options = get_filtered_stocks(selected_indices, fno_only)
    
    st.sidebar.markdown("----")

    # Corporate Actions Filters
    st.sidebar.markdown("### 📊 Corporate Actions")
    ca_stock_filter = st.sidebar.multiselect(
        "Select Stocks", 
        options=stock_options,
        key="ca_stocks",
        help="Filter corporate actions by specific stocks"
    )
    
    ca_keyword = st.sidebar.text_input(
        "Keywords", 
        value="dividend, bonus", 
        key="ca_keywords",
        help="Comma-separated keywords to filter corporate actions"
    )
    
    ca_col1, ca_col2 = st.sidebar.columns(2)
    with ca_col1:
        ca_start_date = st.date_input("From", value=date.today(), key="ca_start")
    with ca_col2:
        ca_end_date = st.date_input("To", value=date.today() + timedelta(days=90), key="ca_end")

    
    # Announcements Filters
    st.sidebar.markdown("### 📢 Announcements")
    ann_stock_filter = st.sidebar.multiselect(
        "Select Stocks", 
        options=stock_options,
        key="ann_stocks"
    )
    
    ann_keyword = st.sidebar.text_input(
        "Keywords", 
        value="dividend, record date", 
        key="ann_keywords"
    )
    
    ann_col1, ann_col2 = st.sidebar.columns(2)
    with ann_col1:
        ann_start_date = st.date_input("From", value=date.today() - timedelta(days=7), key="ann_start")
    with ann_col2:
        ann_end_date = st.date_input("To", value=date.today(), key="ann_end")
    
    # Board Meetings Filters
    st.sidebar.markdown("### 📅 Board Meetings")
    bm_stock_filter = st.sidebar.multiselect(
        "Select Stocks", 
        options=stock_options,
        key="bm_stocks"
    )
    
    bm_date = st.sidebar.date_input("From Date", value=date.today(), key="bm_date")
    bm_keyword = st.sidebar.text_input("Keywords", value="results, dividend", key="bm_keywords")
    
    return {
        'indices': selected_indices,
        'fno_only': fno_only,
        'ca': {
            'stocks': ca_stock_filter,
            'keywords': ca_keyword,
            'start_date': ca_start_date,
            'end_date': ca_end_date
        },
        'ann': {
            'stocks': ann_stock_filter,
            'keywords': ann_keyword,
            'start_date': ann_start_date,
            'end_date': ann_end_date,
        },
        'bm': {
            'stocks': bm_stock_filter,
            'keywords': bm_keyword,
            'start_date': bm_date
        }
    }

def display_metrics(ann_df: pd.DataFrame, bm_df: pd.DataFrame, ca_df: pd.DataFrame):
    """Display key metrics in columns."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📢 Total Announcements", len(ann_df))
    
    with col2:
        st.metric("📅 Upcoming Board Meetings", len(bm_df))
    
    with col3:
        st.metric("📊 Corporate Actions", len(ca_df))
    
    with col4:
        recent_count = sum([
            len(ann_df[ann_df['added'] > datetime.now() - timedelta(hours=24)]) if 'added' in ann_df.columns else 0,
            len(bm_df[bm_df['added'] > datetime.now() - timedelta(hours=24)]) if 'added' in bm_df.columns else 0,
            len(ca_df[ca_df['added'] > datetime.now() - timedelta(hours=24)]) if 'added' in ca_df.columns else 0
        ])
        st.metric("🆕 Recent Updates", recent_count)


def display_corporate_actions(df: pd.DataFrame):
    """Display corporate actions data with enhanced formatting."""
    if df.empty:
        st.info("No corporate actions found matching the selected filters.")
        return
        
    # Prepare display dataframe
    display_df = df.copy()
    if 'added' in display_df.columns:
        is_recent = display_df['added'].apply(lambda x: is_recent_record(x))
        display_df['Status'] = is_recent.apply(lambda x: '🟢 New' if x else '')
        display_df = display_df.drop(['added'], axis=1)
    
    # Sort by Ex Date
    display_df = display_df.sort_values(by=['Ex_date', 'short_name'], ascending=[True, True])
    
    st.data_editor(
        display_df,
        use_container_width=True,
        hide_index=True,
        disabled=True,
        column_config={
            "Status": st.column_config.TextColumn(
                "Status",
                help="🟢 indicates recently added records",
                width="small"
            ),
            "short_name": st.column_config.TextColumn("Symbol", width="medium"),
            "scrip_code": st.column_config.TextColumn("Code", width="small"),
            "Ex_date": st.column_config.DateColumn(
                "Ex Date",
                format="DD-MMM-YYYY",
                width="medium"
            ),
            "RD_Date": st.column_config.DateColumn(
                "Record Date",
                format="DD-MMM-YYYY",
                width="medium"
            ),
            "Purpose": st.column_config.TextColumn("Purpose", width="large")
        }
    )


def display_announcements(df: pd.DataFrame):#, limit: int):
    """Display announcements data with enhanced formatting."""
    if df.empty:
        st.info("No announcements found matching the selected filters.")
        return
        
    # Prepare display dataframe
    display_df = df.sort_values(by=['an_dt', 'symbol'], ascending=[False, True])#.head()
    
    if 'added' in display_df.columns:
        is_recent = display_df['added'].apply(lambda x: is_recent_record(x))
        display_df['Status'] = is_recent.apply(lambda x: '🟢 New' if x else '')
        display_df = display_df.drop(['added'], axis=1)
    
    st.data_editor(
        display_df,
        use_container_width=True,
        hide_index=True,
        disabled=True,
        column_config={
            "Status": st.column_config.TextColumn(
                "Status",
                help="🟢 indicates recently added records",
                width="small"
            ),
            "symbol": st.column_config.TextColumn("Symbol", width="medium"),
            "description": st.column_config.TextColumn("Subject", width="large"),
            "attchmntText": st.column_config.TextColumn("Details", width="large"),
            "attchmntFile": st.column_config.LinkColumn(
                "Attachment",
                display_text="📎 Link",
                width="small"
            ),
            "an_dt": st.column_config.DatetimeColumn(
                "Date & Time",
                format="DD-MMM-YYYY HH:mm",
                width="medium"
            )
        }
    )


def display_board_meetings(df: pd.DataFrame):
    """Display board meetings data with enhanced formatting."""
    if df.empty:
        st.info("No board meetings found matching the selected filters.")
        return
        
    # Prepare display dataframe
    display_df = df.copy()
    if 'added' in display_df.columns:
        is_recent = display_df['added'].apply(lambda x: is_recent_record(x))
        display_df['Status'] = is_recent.apply(lambda x: '🟢 New' if x else '')
        display_df = display_df.drop(['added'], axis=1)
    
    # Sort by meeting date
    display_df = display_df.sort_values(by=['bm_date'], ascending=True)
    
    st.data_editor(
        display_df,
        use_container_width=True,
        hide_index=True,
        disabled=True,
        column_config={
            "Status": st.column_config.TextColumn(
                "Status",
                help="🟢 indicates recently added records",
                width="small"
            ),
            "bm_symbol": st.column_config.TextColumn("Symbol", width="medium"),
            "bm_purpose": st.column_config.TextColumn("Purpose", width="large"),
            "bm_desc": st.column_config.TextColumn("Description", width="large"),
            "attachment": st.column_config.LinkColumn(
                "Attachment",
                display_text="📎 Link",
                width="small"
            ),
            "bm_date": st.column_config.DateColumn(
                "Meeting Date",
                format="DD-MMM-YYYY",
                width="medium"
            ),
            "bm_timestamp": st.column_config.DatetimeColumn(
                "Announced",
                format="DD-MMM-YYYY HH:mm",
                width="medium"
            )
        }
    )


# ================================================================================
# MAIN APPLICATION
# ================================================================================

def main():
    """Main application function."""
    # Setup page configuration
    setup_page_config()
    
    # Page header
    st.title("📈 Corporate Events Dashboard")
    # st.markdown("*Real-time corporate announcements, board meetings, and corporate actions from NSE & BSE*")
    
    # Initialize session state for F&O stocks
    if "fno_stocks" not in st.session_state:
        with st.spinner("Loading F&O stocks list..."):
            st.session_state['fno_stocks'] = get_fno_stocks()
            # Get index constituents
            st.session_state['index_constituents'] = get_index_constituents()
    
    # # Data refresh button
    # col1, col2 = st.columns([1, 9])
    # with col1:
    refresh_clicked=False #= st.button("🔄 Refresh Data", help="Fetch latest data from NSE and BSE")

    
    # with col2:
    #     if st.button("🗑️ Clear Cache", help="Clear cached data"):
    #         st.cache_data.clear()
    #         st.success("Cache cleared!")
    
    # Load or refresh data
    if refresh_clicked:
        with st.spinner("Fetching latest data from NSE and BSE..."):
            bm, ann, act = load_data(refresh=True)
            st.session_state.update({'ann': ann, 'bm': bm, 'bse': act})
            st.success("✅ Data refreshed successfully!")
    else:
        # Load existing data if not in session state
        if not all(key in st.session_state for key in ['ann', 'bm', 'bse']):
            with st.spinner("Loading data..."):
                bm, ann, act = load_data(refresh=False)
                st.session_state.update({'ann': ann, 'bm': bm, 'bse': act})
        else:
            ann = st.session_state['ann']
            bm = st.session_state['bm']
            act = st.session_state['bse']
    
    # Create sidebar filters
    filters = create_sidebar_filters()
    
    # Apply filters to data
    filtered_ann = apply_universal_filter(
        ann.copy(),
        stock_filter=filters['ann']['stocks'],
        fno_only=filters['fno_only'],
        indices_filter=filters['indices'],
        start_date=filters['ann']['start_date'],
        end_date=filters['ann']['end_date'],
        keywords=[kw.strip() for kw in filters['ann']['keywords'].split(',') if kw.strip()],
        data_type='ann'
    )
    
    filtered_bm = apply_universal_filter(
        bm.copy(),
        stock_filter=filters['bm']['stocks'],
        fno_only=filters['fno_only'],
        indices_filter=filters['indices'],
        start_date=filters['bm']['start_date'],
        keywords=[kw.strip() for kw in filters['bm']['keywords'].split(',') if kw.strip()],
        data_type='bm'
    )
    
    filtered_act = apply_universal_filter(
        act.copy(),
        stock_filter=filters['ca']['stocks'],
        fno_only=filters['fno_only'],
        indices_filter=filters['indices'],
        start_date=filters['ca']['start_date'],
        end_date=filters['ca']['end_date'],
        keywords=[kw.strip() for kw in filters['ca']['keywords'].split(',') if kw.strip()],
        data_type='ca'
    )
    
    # Display metrics
    display_metrics(filtered_ann, filtered_bm, filtered_act)
    st.markdown("---")
    
    # Display data sections
    with st.expander("🏛️ Corporate Actions", expanded=True):
        st.markdown("*Dividend, bonus, rights, and other corporate actions*")
        display_corporate_actions(filtered_act)
    
    with st.expander("📢 Announcements", expanded=True):
        st.markdown("*Latest corporate announcements and filings*")
        display_announcements(filtered_ann)#, filters['ann']['limit'])
    
    with st.expander("📅 Board Meetings", expanded=True):
        st.markdown("*Upcoming board meetings*")
        display_board_meetings(filtered_bm)
    
    # Footer
    st.markdown("---")
    # st.markdown(
    #     "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
    #     "📊 Data sourced from NSE and BSE | "
    #     "</div>",
    #     unsafe_allow_html=True
    # )


# ================================================================================
# APPLICATION ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)
        
        # Provide troubleshooting information
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **Common issues and solutions:**
            
            1. **Database Connection Error**: Check if MySQL service is running and credentials are correct
            2. **Web Scraping Issues**: NSE/BSE websites might be temporarily unavailable
            3. **Chrome Driver Issues**: Ensure ChromeDriver is properly installed
            4. **Network Issues**: Check internet connectivity for API calls
            
            **Quick fixes:**
            - Try refreshing the page
            - Clear cache using the button above
            - Check database connectivity
            - Restart the application
            """)
