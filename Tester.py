import os
import hashlib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st
from prophet import Prophet
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

import math

MAX_FLOAT = 1.797e+308
MAX_FILE_SIZE_MB = 50  # Maximum file size in MB

def safe_bound(value, default=0.0):
    """
    Ensure slider bounds are within Streamlit's supported range.
    Replaces infinities/NaNs with a safe default or max float.
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return default
    if value > MAX_FLOAT:
        return MAX_FLOAT
    if value < -MAX_FLOAT:
        return -MAX_FLOAT
    return float(value)

def validate_file_extension(uploaded_file):
    """
    Validate that the uploaded file has a valid extension.
    Returns True if valid, False otherwise.
    """
    if uploaded_file is None:
        return True  
    
    valid_extensions = ['.csv']
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    return file_extension in valid_extensions

def validate_file_size(uploaded_file):
    """
    Validate that the uploaded file is within size limits.
    Returns True if valid, False otherwise.
    """
    if uploaded_file is None:
        return True  
    
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    return file_size_mb <= MAX_FILE_SIZE_MB

def validate_csv_columns(df):
    """
    Validate that the CSV contains required columns and at least one data column.
    Returns (is_valid, error_message, suggested_columns)
    """
    # Required columns
    required_columns = ['Country', 'Year']
    
    # Expected data columns (at least one should be present)
    expected_data_columns = [
        'CO2', 'co2', 'CO2_emissions', 'co2_emissions', 'CO2 ', ' CO2'
        'Renewables_equivalent_primary_energy', 'renewables', 'renewable_energy',
        'Total_GHG', 'total_ghg', 'ghg_emissions',
        'CO2_per_capita', 'co2_per_capita', 'emissions_per_capita'
    ]
    
    # Check if DataFrame is empty
    if df is None or df.empty:
        return False, "The uploaded file is empty or could not be read.", []
    
    # Get available columns (case-insensitive check)
    available_columns = df.columns.tolist()
    available_columns_lower = [col.lower().strip() for col in available_columns]
    
    # Check for required columns
    missing_required = []
    for req_col in required_columns:
        if not any(req_col.lower() in col_lower for col_lower in available_columns_lower):
            missing_required.append(req_col)
    
    if missing_required:
        return False, f"Missing required columns: {missing_required}", required_columns + expected_data_columns
    
    # Check for at least one data column
    has_data_column = any(
        any(expected.lower() in col_lower for col_lower in available_columns_lower)
        for expected in expected_data_columns
    )
    
    if not has_data_column:
        return False, "No recognized data columns found. Please ensure your CSV contains at least one emissions or renewables data column.", expected_data_columns
    
    # Check if Year column contains numeric data
    year_columns = [col for col in available_columns if 'year' in col.lower()]
    if year_columns:
        year_col = year_columns[0]
        try:
            pd.to_numeric(df[year_col], errors='coerce')
        except:
            return False, f"Year column '{year_col}' does not contain valid numeric data.", []
    
    return True, "", []

def display_validation_error(error_type, message, suggested_columns=None, file_size_mb=None):
    """
    Display appropriate error messages and suggestions based on validation failure.
    """
    if error_type == "extension":
        st.error("❌ **Invalid File Extension**")
        st.error(f"**Error:** {message}")
        st.info("📋 **Solution:** Please upload a CSV file (.csv extension only)")
        st.markdown("""
        **Supported formats:**
        - ✅ .csv (Comma Separated Values)
        
        **Not supported:**
        - ❌ .xlsx, .xls (Excel files)
        - ❌ .txt, .json, .xml
        - ❌ Other file formats
        """)
        
    elif error_type == "size":
        st.error("❌ **File Too Large**")
        st.error(f"**Error:** File size ({file_size_mb:.1f} MB) exceeds the maximum limit of {MAX_FILE_SIZE_MB} MB")
        st.info("📋 **Solutions:**")
        st.markdown(f"""
        - Reduce the file size to under {MAX_FILE_SIZE_MB} MB
        - Remove unnecessary columns or rows
        - Consider splitting the data into smaller files
        - Compress or optimize your CSV file
        """)
        
    elif error_type == "columns":
        st.error("❌ **Invalid CSV Structure**")
        st.error(f"**Error:** {message}")
        
        if suggested_columns:
            st.info("📋 **Required Column Structure:**")
            st.markdown("**Your CSV file must contain these columns:**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Required Columns:**")
                st.markdown("- `Country` (country names)")
                st.markdown("- `Year` (numeric years)")
                
            with col2:
                st.markdown("**Data Columns (at least one):**")
                for col in suggested_columns[2:8]:  # Show first few expected columns
                    st.markdown(f"- `{col}`")
        
        st.markdown("---")
        st.info("💡 **Example of valid CSV structure:**")
        st.code("""
Country,Year,CO2,Renewables_equivalent_primary_energy
United States,2020,5000,1200
China,2020,10000,800
Germany,2020,800,400
        """)
        
    elif error_type == "read":
        st.error("❌ **File Reading Error**")
        st.error(f"**Error:** {message}")
        st.info("📋 **Possible solutions:**")
        st.markdown("""
        - Ensure the file is a valid CSV format
        - Check that the file is not corrupted
        - Verify the file encoding (UTF-8 recommended)
        - Make sure the file is properly saved as CSV
        """)

def clean_data_for_forecast(df, target_column):
    """
    Clean data specifically for forecasting, removing infinities and extreme outliers
    """
    # Remove infinities and NaN values
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_column])
    
    # Remove extreme outliers (beyond 3 standard deviations)
    if len(df_clean) > 3:
        mean_val = df_clean[target_column].mean()
        std_val = df_clean[target_column].std()
        if std_val > 0:
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            df_clean = df_clean[
                (df_clean[target_column] >= lower_bound) & 
                (df_clean[target_column] <= upper_bound)
            ]
    
    return df_clean

def calculate_derived_features(df):
    """
    Calculate derived features like growth rates and efficiency metrics
    """
    df = df.copy()
    
    # Sort by Country and Year to ensure proper calculation
    df = df.sort_values(['Country', 'Year'])
    
    # Calculate year-over-year growth rates by country
    for country in df['Country'].unique():
        country_mask = df['Country'] == country
        country_data = df[country_mask].copy()
        
        # CO2 growth rate
        if 'CO2' in df.columns:
            df.loc[country_mask, 'CO2_grow_yoy'] = country_data['CO2'].pct_change() * 100
            # Calculate lagged values
            df.loc[country_mask, 'CO2_lag1'] = country_data['CO2'].shift(1)
            df.loc[country_mask, 'CO2_lag2'] = country_data['CO2'].shift(2)
        
        # Renewables growth rate
        if 'Renewables_equivalent_primary_energy' in df.columns:
            df.loc[country_mask, 'Renewables_grow_yoy'] = country_data['Renewables_equivalent_primary_energy'].pct_change() * 100
            # Calculate lagged values
            df.loc[country_mask, 'Renewables_lag1'] = country_data['Renewables_equivalent_primary_energy'].shift(1)
            df.loc[country_mask, 'Renewables_lag2'] = country_data['Renewables_equivalent_primary_energy'].shift(2)
    
    # Calculate CO2 per renewable unit (efficiency metric)
    if 'CO2' in df.columns and 'Renewables_equivalent_primary_energy' in df.columns:
        # Avoid division by zero
        renewable_mask = df['Renewables_equivalent_primary_energy'] > 0
        df.loc[renewable_mask, 'CO2_per_renewable_unit'] = (
            df.loc[renewable_mask, 'CO2'] / df.loc[renewable_mask, 'Renewables_equivalent_primary_energy']
        )
    
    # Clean up infinite and extreme values
    derived_cols = ['CO2_grow_yoy', 'Renewables_grow_yoy', 'CO2_per_renewable_unit', 
                   'CO2_lag1', 'CO2_lag2', 'Renewables_lag1', 'Renewables_lag2']
    
    for col in derived_cols:
        if col in df.columns:
            # Replace infinities with NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Cap extreme growth rates at reasonable levels
            if 'grow_yoy' in col:
                df[col] = df[col].clip(-100, 500)  # Cap at -100% to 500% growth
    
    return df

# --------------------------
# Region Mapping
# --------------------------
UN_TO_SUBREG = {
    'Algeria':'Northern Africa','Argentina':'South America','Australia':'Australia and New Zealand','Austria':'Western Europe',
    'Azerbaijan':'Western Asia','Bangladesh':'Southern Asia','Belarus':'Eastern Europe','Belgium':'Western Europe','Brazil':'South America',
    'Bulgaria':'Eastern Europe','Canada':'Northern America','Chile':'South America','China':'Eastern Asia','Colombia':'South America',
    'Croatia':'Southern Europe','Cyprus':'Southern Europe','Czechia':'Eastern Europe','Denmark':'Northern Europe','Ecuador':'South America',
    'Egypt':'Northern Africa','Estonia':'Northern Europe','Finland':'Northern Europe','France':'Western Europe','Germany':'Western Europe',
    'Greece':'Southern Europe','Hong Kong':'Eastern Asia','Hungary':'Eastern Europe','Iceland':'Northern Europe','India':'Southern Asia',
    'Indonesia':'South-eastern Asia','Iran':'Southern Asia','Iraq':'Western Asia','Ireland':'Northern Europe','Israel':'Western Asia',
    'Italy':'Southern Europe','Japan':'Eastern Asia','Kazakhstan':'Central Asia','Kuwait':'Western Asia','Latvia':'Northern Europe',
    'Lithuania':'Northern Europe','Luxembourg':'Western Europe','Malaysia':'South-eastern Asia','Mexico':'Central America',
    'Morocco':'Northern Africa','Netherlands':'Western Europe','New Zealand':'Australia and New Zealand','North Macedonia':'Southern Europe',
    'Norway':'Northern Europe','Oman':'Western Asia','Pakistan':'Southern Asia','Peru':'South America','Philippines':'South-eastern Asia',
    'Poland':'Eastern Europe','Portugal':'Southern Europe','Qatar':'Western Asia','Romania':'Eastern Europe','Russia':'Eastern Europe',
    'Saudi Arabia':'Western Asia','Singapore':'South-eastern Asia','Slovakia':'Eastern Europe','Slovenia':'Southern Europe',
    'South Africa':'Southern Africa','South Korea':'Eastern Asia','Spain':'Southern Europe','Sri Lanka':'Southern Asia','Sweden':'Northern Europe',
    'Switzerland':'Western Europe','Taiwan':'Eastern Asia','Thailand':'South-eastern Asia','Trinidad and Tobago':'Latin America and the Caribbean',
    'Turkey':'Western Asia','Turkmenistan':'Central Asia','Ukraine':'Eastern Europe','United Arab Emirates':'Western Asia',
    'United Kingdom':'Northern Europe','United States':'Northern America','Uzbekistan':'Central Asia','Venezuela':'South America','Vietnam':'South-eastern Asia'
}

SUBREG_TO_REGION = {
    'Northern Africa':'Africa','Southern Africa':'Africa','Middle Africa':'Africa','Western Africa':'Africa','Eastern Africa':'Africa',
    'Southern Europe':'Europe','Northern Europe':'Europe','Western Europe':'Europe','Eastern Europe':'Europe',
    'Central Asia':'Asia','Eastern Asia':'Asia','South-eastern Asia':'Asia','Southern Asia':'Asia','Western Asia':'Asia',
    'Central America':'North America','Northern America':'North America',
    'South America':'South America','Latin America and the Caribbean':'South America',
    'Australia and New Zealand':'Australia','Oceania':'Australia','Other':'Other'
}

# --------------------------
# Load data
# --------------------------
DEFAULT_CSV_PATH = os.environ.get("EMISSIONS_CSV", "Renewable.csv")

st.sidebar.title("📁 Data Source")

# File upload with validation
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File", 
    type=["csv"],
    help=f"Upload a CSV file (max {MAX_FILE_SIZE_MB} MB) with emissions/renewables data"
)

csv_path = st.sidebar.text_input("Or use default CSV path", value=DEFAULT_CSV_PATH)

# Initialize validation flags
file_is_valid = True
use_uploaded = False
validation_error_type = None
error_message = ""
suggested_columns = []

# Validate uploaded file if present
if uploaded_file is not None:
    # Extension validation
    if not validate_file_extension(uploaded_file):
        file_is_valid = False
        validation_error_type = "extension"
        error_message = f"Invalid file extension. Only .csv files are supported. Uploaded: {os.path.splitext(uploaded_file.name)[1]}"
        
    # Size validation
    elif not validate_file_size(uploaded_file):
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        file_is_valid = False
        validation_error_type = "size" 
        error_message = f"File size ({file_size_mb:.1f} MB) exceeds maximum limit"
        
    # CSV structure validation
    else:
        try:
            # Try to read the CSV
            uploaded_file.seek(0)  # Reset file pointer
            temp_df = pd.read_csv(uploaded_file)
            temp_df.columns = [c.strip() for c in temp_df.columns]  # Clean column names
            
            is_valid, csv_error, suggested_cols = validate_csv_columns(temp_df)
            
            if not is_valid:
                file_is_valid = False
                validation_error_type = "columns"
                error_message = csv_error
                suggested_columns = suggested_cols
            else:
                use_uploaded = True
                st.sidebar.success(f"✅ Valid CSV uploaded: {uploaded_file.name}")
                
        except Exception as e:
            file_is_valid = False
            validation_error_type = "read"
            error_message = f"Could not read CSV file: {str(e)}"

# Display validation results
if uploaded_file is not None and not file_is_valid:
    if validation_error_type == "size":
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        display_validation_error(validation_error_type, error_message, file_size_mb=file_size_mb)
    else:
        display_validation_error(validation_error_type, error_message, suggested_columns)
    
    st.sidebar.warning("⚠️ Using default data due to file validation errors")
    st.sidebar.info(f"📂 Default file: {csv_path}")

elif uploaded_file is None:
    st.sidebar.info(f"📂 Using default file: {csv_path}")

else:
    st.sidebar.success(f"✅ Using uploaded file: {uploaded_file.name}")

# Determine data source for caching
if use_uploaded and file_is_valid:
    data_source = "uploaded"
    data_identifier = uploaded_file.name
else:
    data_source = "default"
    data_identifier = csv_path
    uploaded_file = None  # Reset to use default

@st.cache_data
def load_raw(data_source_type, file_identifier, uploaded_file_content=None):
    """
    Load data with proper cache invalidation for uploaded files
    """
    if data_source_type == "uploaded" and uploaded_file_content is not None:
        # Reset file pointer to beginning
        uploaded_file_content.seek(0)
        df = pd.read_csv(uploaded_file_content)
    else:
        try:
            df = pd.read_csv(file_identifier)
        except FileNotFoundError:
            st.error(f"❌ Default file not found: {file_identifier}")
            st.error("Please check the file path or upload a valid CSV file.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error loading default file: {str(e)}")
            st.stop()
    
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    
    # Flexible column renaming - handle different naming conventions
    column_mapping = {
        "co2": "CO2",
        "co2_per_capita": "CO2_per_capita", 
        "total_ghg": "Total_GHG",
        "renewables_equivalent_primary_energy": "Renewables_equivalent_primary_energy",
        "renewables": "Renewables_equivalent_primary_energy",
        "renewable_energy": "Renewables_equivalent_primary_energy",
        "country": "Country",
        "year": "Year"
    }
    
    # Apply column mapping with case-insensitive matching
    for old_name, new_name in column_mapping.items():
        matching_cols = [col for col in df.columns if col.lower() == old_name.lower()]
        if matching_cols:
            df = df.rename(columns={matching_cols[0]: new_name})
    
    # Identify numeric columns dynamically
    potential_numeric_cols = ["Renewables_equivalent_primary_energy", "CO2", "CO2_per_capita", 
                             "Total_GHG", "Year"]
    
    numeric_cols = [col for col in potential_numeric_cols if col in df.columns]
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Replace infinities with NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Check for required columns
    required_cols = ["Country", "Year"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.error(f"Available columns: {list(df.columns)}")
        st.stop()
    
    # Drop rows with missing essential data
    df = df.dropna(subset=required_cols)
    
    # Add region mapping if not present
    if 'Region_Canonical' not in df.columns:
        df["UN_Region"] = df["Country"].map(UN_TO_SUBREG).fillna("Other")
        df["Region_Canonical"] = df["UN_Region"].map(SUBREG_TO_REGION).fillna("Other")
    
    # Fill missing values for key columns
    if "Renewables_equivalent_primary_energy" in df.columns:
        df["Renewables_equivalent_primary_energy"] = df["Renewables_equivalent_primary_energy"].fillna(0)
    
    if "Total_GHG" in df.columns and "CO2" in df.columns:
        df["Total_GHG"] = df["Total_GHG"].fillna(df["CO2"])
    
    # Calculate derived features AFTER basic cleaning
    df = calculate_derived_features(df)
    
    return df

# Load data with proper cache key
if use_uploaded and file_is_valid and uploaded_file is not None:
    # Create a unique cache key based on file content
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    df = load_raw("uploaded", f"{uploaded_file.name}_{file_hash}", uploaded_file)
else:
    df = load_raw("default", csv_path)

# Add data validation summary
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Data Summary")
st.sidebar.write(f"**Rows:** {len(df):,}")
st.sidebar.write(f"**Columns:** {len(df.columns)}")
st.sidebar.write(f"**Countries:** {df['Country'].nunique()}")
if 'Year' in df.columns:
    st.sidebar.write(f"**Years:** {df['Year'].min():.0f} - {df['Year'].max():.0f}")

# Show column mapping for uploaded files
if use_uploaded and file_is_valid:
    with st.sidebar.expander("🔍 Column Details"):
        st.write("**Base Columns:**")
        base_cols = [col for col in df.columns if col not in ['CO2_grow_yoy', 'Renewables_grow_yoy', 'CO2_per_renewable_unit', 'CO2_lag1', 'CO2_lag2', 'Renewables_lag1', 'Renewables_lag2', 'UN_Region', 'Region_Canonical']]
        for col in sorted(base_cols):
            st.write(f"• {col}")
        
        st.write("**Derived Columns:**")
        derived_cols = ['CO2_grow_yoy', 'Renewables_grow_yoy', 'CO2_per_renewable_unit']
        for col in derived_cols:
            if col in df.columns:
                st.write(f"• {col}")

# Add file validation help
with st.sidebar.expander("ℹ️ File Requirements"):
    st.markdown("""
    **Required:**
    - ✅ .csv format only
    - ✅ Max file size: 50 MB
    - ✅ 'Country' column
    - ✅ 'Year' column (numeric)
    
    **Expected data columns (≥1):**
    - CO2, Renewables
    - Total_GHG, CO2_per_capita
    - Any emissions/energy data
    
    **Example structure:**
    ```
    Country,Year,CO2,Renewables
    USA,2020,5000,1200
    China,2020,10000,800
    ```
    """)

# --------------------------
# Enhanced Aggregation helper
# --------------------------
def aggregate_data(region=None, country=None):
    agg_dict = {
        "CO2": "sum",
        "Renewables_equivalent_primary_energy": "sum", 
        "Total_GHG": "sum",
        "CO2_per_capita": "mean",
        "Renewables_grow_yoy": "mean",
        "CO2_grow_yoy": "mean",
        "CO2_per_renewable_unit": "mean"
    }
    
    # Only aggregate columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    if region == "Global":
        df_f = df.groupby("Year", as_index=False).agg(agg_dict)
        df_f["Label"] = "Global"
    elif country and country != "All":
        df_f = df[df["Country"] == country].groupby("Year", as_index=False).agg(agg_dict)
        df_f["Label"] = f"{region} - {country}"
    else:
        df_f = df[df["Region_Canonical"] == region].groupby("Year", as_index=False).agg(agg_dict)
        df_f["Label"] = f"{region} - All"
    
    # Clean any remaining infinities
    numeric_columns = df_f.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df_f[col] = df_f[col].replace([np.inf, -np.inf], np.nan)
    
    return df_f

# --------------------------
# Enhanced Insights Generator
# --------------------------
def generate_insights(df_filtered, forecast, target_feature, horizon, label):
    """Generate comprehensive AI insights from forecast data"""
    
    historical_data = df_filtered[target_feature].dropna()
    if len(historical_data) < 2:
        return "⚠️ Insufficient historical data for trend analysis."
    
    # Calculate key metrics with safety checks
    latest_historical = historical_data.iloc[-1]
    earliest_historical = historical_data.iloc[0]
    forecast_final = forecast["yhat"].iloc[-1]
    
    # Safe percentage calculation
    if latest_historical != 0 and not np.isnan(latest_historical) and not np.isinf(latest_historical):
        forecast_change = ((forecast_final - latest_historical) / abs(latest_historical)) * 100
    else:
        forecast_change = 0
    
    if earliest_historical != 0 and not np.isnan(earliest_historical) and not np.isinf(earliest_historical):
        historical_change = ((latest_historical - earliest_historical) / abs(earliest_historical)) * 100
    else:
        historical_change = 0
    
    # Advanced metrics with safety checks
    recent_volatility = historical_data.iloc[-5:].std() if len(historical_data) >= 5 else 0
    forecast_volatility = forecast["yhat"].iloc[-horizon:].std()
    
    if np.isnan(recent_volatility) or np.isinf(recent_volatility):
        recent_volatility = 0
    if np.isnan(forecast_volatility) or np.isinf(forecast_volatility):
        forecast_volatility = 0
    
    insights = []
    
    # 1. Overall forecast direction with enhanced context
    if target_feature in ["Renewables_grow_yoy", "CO2_grow_yoy"]:
        if forecast_change > 10:
            insights.append(f"📈 **Accelerating Growth Rate**: {target_feature.replace('_', ' ').title()} is projected to increase by **{forecast_change:.1f}%**, indicating faster year-over-year changes ahead.")
        elif forecast_change < -10:
            insights.append(f"📉 **Decelerating Growth**: {target_feature.replace('_', ' ').title()} expected to slow by **{abs(forecast_change):.1f}%**, suggesting stabilizing trends.")
        else:
            insights.append(f"➡️ **Stable Growth Pattern**: {target_feature.replace('_', ' ').title()} projected to remain steady with **{forecast_change:.1f}%** change over {horizon} years.")
    else:
        if forecast_change > 5:
            insights.append(f"📈 **Strong Growth Expected**: {target_feature} is projected to increase by **{abs(forecast_change):.1f}%** over the next {horizon} years.")
        elif forecast_change > 1:
            insights.append(f"📊 **Moderate Growth**: {target_feature} is expected to grow by **{forecast_change:.1f}%** in the next {horizon} years.")
        elif forecast_change < -5:
            insights.append(f"📉 **Significant Decline**: {target_feature} is forecasted to decrease by **{abs(forecast_change):.1f}%** over {horizon} years.")
        elif forecast_change < -1:
            insights.append(f"📊 **Moderate Decline**: {target_feature} is projected to fall by **{abs(forecast_change):.1f}%** in the next {horizon} years.")
        else:
            insights.append(f"➡️ **Stable Outlook**: {target_feature} is expected to remain relatively stable with only **{forecast_change:.1f}%** change over {horizon} years.")

    # 2. Volatility and stability analysis (only if both values are valid)
    if recent_volatility > 0 and forecast_volatility > 0:
        if forecast_volatility < recent_volatility * 0.8:
            volatility_reduction = ((recent_volatility - forecast_volatility)/recent_volatility*100)
            insights.append(f"📊 **Increasing Stability**: Forecast shows **{volatility_reduction:.0f}%** reduction in volatility, suggesting more predictable patterns ahead.")
        elif forecast_volatility > recent_volatility * 1.2:
            volatility_increase = ((forecast_volatility - recent_volatility)/recent_volatility*100)
            insights.append(f"⚠️ **Increasing Volatility**: Model predicts **{volatility_increase:.0f}%** higher variability, indicating potential market instability.")

    # 3. Advanced feature-specific insights
    if target_feature == "CO2_per_renewable_unit":
        if forecast_change < -10:
            insights.append(f"🌱 **Improving Carbon Efficiency**: CO₂ per renewable unit declining by **{abs(forecast_change):.1f}%** indicates significant improvement in clean energy effectiveness.")
        elif forecast_change > 10:
            insights.append(f"⚠️ **Efficiency Concerns**: Rising CO₂ per renewable unit suggests declining effectiveness of renewable investments.")
    
    elif target_feature == "Renewables_grow_yoy":
        if abs(latest_historical) > 15:
            insights.append(f"⚡ **High Growth Momentum**: Current **{latest_historical:.1f}%** annual renewable growth rate indicates strong clean energy acceleration.")
        elif abs(latest_historical) < 5:
            insights.append(f"🔋 **Growth Opportunity**: Low **{latest_historical:.1f}%** renewable growth suggests potential for policy intervention.")

    # 4. Regional context and benchmarking
    if "Global" in label:
        if target_feature == "CO2" and forecast_change > 0:
            insights.append(f"🌍 **Global Climate Challenge**: Rising global CO₂ emissions require urgent international coordination for climate targets.")
        elif target_feature.startswith("Renewables") and forecast_change > 10:
            insights.append(f"🚀 **Global Energy Transition**: Strong renewable growth worldwide signals successful shift toward sustainable energy systems.")
    else:
        region_name = label.split(" - ")[0]
        if target_feature == "CO2_grow_yoy":
            if forecast_change < -5:
                insights.append(f"🏆 **Regional Leadership**: {region_name} showing strong CO₂ growth rate reduction, setting example for other regions.")

    # 5. Uncertainty and confidence
    forecast_final_upper = forecast["yhat_upper"].iloc[-1]
    forecast_final_lower = forecast["yhat_lower"].iloc[-1]
    
    if forecast_final != 0 and not np.isnan(forecast_final) and not np.isinf(forecast_final):
        uncertainty_range = ((forecast_final_upper - forecast_final_lower) / abs(forecast_final)) * 100
        if np.isnan(uncertainty_range) or np.isinf(uncertainty_range):
            uncertainty_range = 50  # Default moderate uncertainty
    else:
        uncertainty_range = 50
    
    if uncertainty_range > 50:
        insights.append(f"⚠️ **High Uncertainty**: Wide confidence interval ({uncertainty_range:.0f}% range) suggests multiple possible scenarios - consider scenario planning.")
    elif uncertainty_range < 20:
        insights.append(f"✅ **High Confidence**: Narrow confidence interval ({uncertainty_range:.0f}% range) indicates reliable projections for planning purposes.")

    # 6. Key numbers summary
    final_insight = f"📊 **Executive Summary**: Current: **{latest_historical:.1f}** → {horizon}-year projection: **{forecast_final:.1f}** (**{forecast_change:+.1f}%** change) with **±{uncertainty_range:.0f}%** confidence range."
    insights.append(final_insight)
    
    return "\n\n".join(insights)

# --------------------------
# Advanced Analytics Functions
# --------------------------
def calculate_correlation_matrix(df_filtered):
    """Calculate correlation matrix for numeric variables"""
    numeric_cols = ['CO2', 'Renewables_equivalent_primary_energy', 'Total_GHG', 'CO2_per_capita',
                   'Renewables_grow_yoy', 'CO2_grow_yoy', 'CO2_per_renewable_unit']
    
    available_cols = [col for col in numeric_cols if col in df_filtered.columns]
    
    if len(available_cols) < 2:
        return pd.DataFrame()  # Return empty dataframe if insufficient columns
    
    # Clean data for correlation calculation
    corr_df = df_filtered[available_cols].copy()
    for col in available_cols:
        corr_df[col] = corr_df[col].replace([np.inf, -np.inf], np.nan)
    
    # Drop columns with all NaN values
    corr_df = corr_df.dropna(axis=1, how='all')
    
    if corr_df.empty or len(corr_df.columns) < 2:
        return pd.DataFrame()
    
    corr_data = corr_df.corr()
    
    return corr_data

def create_correlation_heatmap(corr_matrix):
    """Create correlation heatmap using plotly"""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Variable Correlation Matrix",
        height=600,
        xaxis_tickangle=-45
    )
    
    return fig

def create_efficiency_analysis(df_filtered):
    """Analyze CO2 per renewable unit over time"""
    if 'CO2_per_renewable_unit' not in df_filtered.columns:
        return None
    
    efficiency_data = df_filtered[['Year', 'CO2_per_renewable_unit']].copy()
    efficiency_data['CO2_per_renewable_unit'] = efficiency_data['CO2_per_renewable_unit'].replace([np.inf, -np.inf], np.nan)
    efficiency_data = efficiency_data.dropna()
    
    if len(efficiency_data) == 0:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=efficiency_data['Year'],
        y=efficiency_data['CO2_per_renewable_unit'],
        mode='lines+markers',
        name='CO₂ per Renewable Unit',
        line=dict(width=3, color='orange'),
        marker=dict(size=8)
    ))
    
    # Add trend line
    if len(efficiency_data) > 1:
        z = np.polyfit(efficiency_data['Year'], efficiency_data['CO2_per_renewable_unit'], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=efficiency_data['Year'],
            y=p(efficiency_data['Year']),
            mode='lines',
            name='Trend',
            line=dict(dash='dash', color='red')
        ))
    
    fig.update_layout(
        title="Carbon Efficiency: CO₂ per Renewable Unit Over Time",
        xaxis_title="Year",
        yaxis_title="CO₂ per Renewable Unit",
        height=500
    )
    
    return fig

def create_growth_analysis(df_filtered):
    """Create dual-axis plot for growth rates"""
    if 'CO2_grow_yoy' not in df_filtered.columns or 'Renewables_grow_yoy' not in df_filtered.columns:
        return None
    
    growth_data = df_filtered[['Year', 'CO2_grow_yoy', 'Renewables_grow_yoy']].copy()
    
    # Clean infinities
    for col in ['CO2_grow_yoy', 'Renewables_grow_yoy']:
        growth_data[col] = growth_data[col].replace([np.inf, -np.inf], np.nan)
    
    growth_data = growth_data.dropna()
    
    if len(growth_data) == 0:
        return None
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # CO2 growth rate
    fig.add_trace(
        go.Scatter(
            x=growth_data['Year'], 
            y=growth_data['CO2_grow_yoy'],
            mode='lines+markers',
            name='CO₂ Growth Rate (%)',
            line=dict(color='red', width=2)
        ),
        secondary_y=False,
    )
    
    # Renewables growth rate
    fig.add_trace(
        go.Scatter(
            x=growth_data['Year'], 
            y=growth_data['Renewables_grow_yoy'],
            mode='lines+markers',
            name='Renewables Growth Rate (%)',
            line=dict(color='green', width=2)
        ),
        secondary_y=True,
    )
    
    # Add horizontal line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="CO₂ Growth Rate (%)", secondary_y=False)
    fig.update_yaxes(title_text="Renewables Growth Rate (%)", secondary_y=True)
    
    fig.update_layout(
        title="Year-over-Year Growth Rates Comparison",
        height=500,
        hovermode='x unified'
    )
    
    return fig

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Advanced Emissions & Renewables Analytics", layout="wide")
st.title("🌍 Advanced Emissions & Renewables Analytics Platform")

# Enhanced sidebar
st.sidebar.title("🎛️ Analysis Controls")

regions = ["Global"] + sorted(df["Region_Canonical"].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("Select Region", regions)

if selected_region == "Global":
    countries = ["All"]
else:
    countries = ["All"] + sorted(df[df["Region_Canonical"] == selected_region]["Country"].unique())

selected_country = st.sidebar.selectbox("Select Country", countries)

df_filtered = aggregate_data(selected_region, selected_country if selected_country != "All" else None)

# --------------------------
# Enhanced Tabs
# --------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Core Dashboard", 
    "🔮 Advanced Forecasting", 
    "📈 Growth & Efficiency",
    "🔗 Correlation Analysis",
    "📋 Data Explorer"
])

with tab1:
    st.subheader("Core Emissions & Renewables Dashboard")
    
    # Enhanced metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    latest_year = df_filtered["Year"].max()
    latest_data = df_filtered[df_filtered["Year"] == latest_year].iloc[0]
    
    with col1:
        st.metric("Latest CO₂ (Mt)", f"{latest_data['CO2']:.1f}")
    with col2:
        st.metric("Latest Renewables", f"{latest_data['Renewables_equivalent_primary_energy']:.1f}")
    with col3:
        if 'CO2_grow_yoy' in latest_data:
            co2_growth = latest_data['CO2_grow_yoy']
            if not (np.isnan(co2_growth) or np.isinf(co2_growth)):
                st.metric("CO₂ Growth Rate", f"{co2_growth:.2f}%", delta=f"{co2_growth:.2f}%")
            else:
                st.metric("Latest GHG (Mt)", f"{latest_data['Total_GHG']:.1f}")
        else:
            st.metric("Latest GHG (Mt)", f"{latest_data['Total_GHG']:.1f}")
    with col4:
        if 'Renewables_grow_yoy' in latest_data:
            ren_growth = latest_data['Renewables_grow_yoy']
            if not (np.isnan(ren_growth) or np.isinf(ren_growth)):
                st.metric("Renewables Growth", f"{ren_growth:.2f}%", delta=f"{ren_growth:.2f}%")
            else:
                st.metric("CO₂ per Capita", f"{latest_data['CO2_per_capita']:.2f}")
        else:
            st.metric("CO₂ per Capita", f"{latest_data['CO2_per_capita']:.2f}")
    
    # Enhanced 4-panel dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("CO₂ Emissions (Mt)", "Renewables (Primary Energy Eq.)",
                        "Total GHG (Mt CO₂eq)", "CO₂ per Capita (tons)"),
        horizontal_spacing=0.12, vertical_spacing=0.15
    )

    trace_names = ["CO2", "Renewables_equivalent_primary_energy", "Total_GHG", "CO2_per_capita"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, col in enumerate(trace_names, 1):
        row, colpos = divmod(i-1, 2)
        fig.add_trace(go.Scatter(
            x=df_filtered["Year"], y=df_filtered[col],
            mode="lines+markers", name=col,
            line=dict(color=colors[i-1], width=3),
            marker=dict(size=8)
        ), row=row+1, col=colpos+1)

    fig.update_layout(
        title=f"Core Metrics: {df_filtered['Label'].iloc[0]}",
        height=900, showlegend=False
    )

    fig.update_xaxes(rangeslider_visible=True, row=2, col=1)
    fig.update_xaxes(rangeslider_visible=True, row=2, col=2)

with tab2:
    st.subheader("🔮 Advanced Predictive Analytics")
    
    # Enhanced forecast options
    forecast_options = {
        "Core Metrics": ["CO2", "Renewables_equivalent_primary_energy", "Total_GHG", "CO2_per_capita"],
        "Growth Rates": ["Renewables_grow_yoy", "CO2_grow_yoy"],
        "Efficiency": ["CO2_per_renewable_unit"]
    }
    
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Forecast Category", list(forecast_options.keys()))
    with col2:
        available_features = [f for f in forecast_options[category] if f in df_filtered.columns]
        target_feature = st.selectbox("Select Variable", available_features)

    horizon = st.slider("Forecast Horizon (years)", 5, 30, 10)

    if target_feature not in df_filtered.columns:
        st.error(f"Variable '{target_feature}' not found in data")
    else:
        # Clean data before forecasting
        df_clean = clean_data_for_forecast(df_filtered, target_feature)
        
        # Prepare data for Prophet
        df_forecast = df_clean.rename(columns={"Year":"ds", target_feature:"y"})[["ds","y"]].copy()
        df_forecast["ds"] = pd.to_datetime(df_forecast["ds"], format="%Y")
        df_forecast = df_forecast.dropna()

        if len(df_forecast) < 3:
            st.warning("Not enough clean data points for forecasting. Need at least 3 years of data without infinities.")
        else:
            # Enhanced Prophet model with seasonality detection
            seasonality_mode = 'additive'
            if target_feature in ['CO2_grow_yoy', 'Renewables_grow_yoy']:
                seasonality_mode = 'multiplicative'
            
            model = Prophet(
                yearly_seasonality=False,
                daily_seasonality=False, 
                weekly_seasonality=False,
                interval_width=0.95,
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=0.05
            )
            
            with st.spinner('Training advanced forecasting model...'):
                try:
                    model.fit(df_forecast)
                    
                    # Generate predictions
                    future = model.make_future_dataframe(periods=horizon, freq="Y")
                    forecast = model.predict(future)

                    # Enhanced forecast visualization
                    fig_pred = go.Figure()
                    
                    # Historical data
                    fig_pred.add_trace(go.Scatter(
                        x=df_forecast["ds"], y=df_forecast["y"],
                        mode="lines+markers", name="Historical",
                        line=dict(color='blue', width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Forecast line
                    forecast_start = len(df_forecast) - 1
                    fig_pred.add_trace(go.Scatter(
                        x=forecast["ds"][forecast_start:], 
                        y=forecast["yhat"][forecast_start:],
                        mode="lines", name="Forecast",
                        line=dict(color='red', width=3, dash='dash')
                    ))
                    
                    # Enhanced confidence interval
                    fig_pred.add_trace(go.Scatter(
                        x=forecast["ds"][forecast_start:], 
                        y=forecast["yhat_upper"][forecast_start:],
                        mode="lines", line=dict(width=0), 
                        showlegend=False, hoverinfo='skip'
                    ))
                    
                    fig_pred.add_trace(go.Scatter(
                        x=forecast["ds"][forecast_start:], 
                        y=forecast["yhat_lower"][forecast_start:],
                        mode="lines", fill="tonexty", 
                        fillcolor='rgba(255,0,0,0.2)',
                        line=dict(width=0), 
                        name="95% Confidence Interval"
                    ))

                    fig_pred.update_layout(
                        title=f"Advanced Forecast: {target_feature} for {df_filtered['Label'].iloc[0]}",
                        xaxis_title="Year", yaxis_title=target_feature,
                        height=700, hovermode='x unified'
                    )

                    st.plotly_chart(fig_pred, use_container_width=True)

                    # Enhanced AI Insights
                    st.markdown("---")
                    st.subheader("🤖 AI-Generated Strategic Insights")
                    
                    insights = generate_insights(df_filtered, forecast, target_feature, horizon, df_filtered['Label'].iloc[0])
                    
                    st.markdown(
                        f"""
                        <div style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 25px;
                            border-radius: 15px;
                            border-left: 8px solid #4CAF50;
                            margin: 15px 0;
                            color: white;
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        ">
                            {insights.replace(chr(10), '<br>')}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                    # Executive Dashboard
                    st.markdown("### 📊 Executive Summary")
                    
                    latest_val = df_clean[target_feature].iloc[-1]
                    forecast_val = forecast["yhat"].iloc[-1]
                    
                    # Safe change calculation
                    if latest_val != 0 and not np.isnan(latest_val) and not np.isinf(latest_val):
                        change_pct = ((forecast_val - latest_val) / abs(latest_val)) * 100
                    else:
                        change_pct = 0
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: 
                        st.metric(
                            label=f"Current {target_feature}",
                            value=f"{latest_val:.2f}",
                            help="Latest historical value"
                        )
                    with col2:
                        st.metric(
                            label=f"Projected ({horizon}y)",
                            value=f"{forecast_val:.2f}",
                            delta=f"{change_pct:+.1f}%",
                            help="Forecasted value with percentage change"
                        )
                    with col3:
                        trend_emoji = "📈" if change_pct > 1 else "📉" if change_pct < -1 else "➡️"
                        trend_text = "Growing" if change_pct > 1 else "Declining" if change_pct < -1 else "Stable"
                        st.metric(
                            label="Trend Direction",
                            value=f"{trend_emoji} {trend_text}",
                            help="Overall trend direction"
                        )
                    with col4:
                        volatility = forecast["yhat"].std()
                        if np.isnan(volatility) or np.isinf(volatility):
                            volatility = 0
                        st.metric(
                            label="Forecast Volatility",
                            value=f"{volatility:.2f}",
                            help="Standard deviation of forecast"
                        )

                    # Detailed forecast table with enhanced metrics
                    with st.expander("📋 Comprehensive Forecast Analysis"):
                        forecast_future = forecast[forecast["ds"] > df_forecast["ds"].max()][["ds", "yhat", "yhat_lower", "yhat_upper"]]
                        forecast_future["ds"] = forecast_future["ds"].dt.year
                        forecast_future.columns = ["Year", "Forecast", "Lower_Bound", "Upper_Bound"]
                        
                        # Enhanced calculations with safety checks
                        if latest_val != 0 and not np.isnan(latest_val) and not np.isinf(latest_val):
                            forecast_future["Change_from_Current_%"] = ((forecast_future["Forecast"] - latest_val) / abs(latest_val) * 100).round(2)
                        else:
                            forecast_future["Change_from_Current_%"] = 0
                            
                        forecast_future["YoY_Change_%"] = forecast_future["Forecast"].pct_change().fillna(0).round(4) * 100
                        
                        # Safe confidence range calculation
                        confidence_range = []
                        for _, row in forecast_future.iterrows():
                            if row["Forecast"] != 0 and not np.isnan(row["Forecast"]) and not np.isinf(row["Forecast"]):
                                conf_range = ((row["Upper_Bound"] - row["Lower_Bound"]) / abs(row["Forecast"]) * 100)
                            else:
                                conf_range = 0
                            confidence_range.append(round(conf_range, 1))
                        
                        forecast_future["Confidence_Range_%"] = confidence_range
                        
                        # Round all values
                        forecast_future = forecast_future.round(2)
                        
                        st.dataframe(forecast_future, use_container_width=True)
                        
                        # Enhanced download functionality
                        csv_data = forecast_future.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Complete Forecast Analysis",
                            data=csv_data,
                            file_name=f"{target_feature}_comprehensive_forecast_{df_filtered['Label'].iloc[0].replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"Forecasting failed: {str(e)}")
                    st.info("This may be due to insufficient data or data quality issues. Try selecting a different variable or region.")

# Tab 3 - Growth & Efficiency Analysis
with tab3:
    st.subheader("📈 Growth Rates & Efficiency Analysis")
    
    # Growth rates analysis
    if 'CO2_grow_yoy' in df_filtered.columns and 'Renewables_grow_yoy' in df_filtered.columns:
        st.markdown("### Year-over-Year Growth Analysis")
        
        growth_fig = create_growth_analysis(df_filtered)
        if growth_fig:
            st.plotly_chart(growth_fig, use_container_width=True)
            
            # Growth insights with safety checks
            co2_growth_data = df_filtered['CO2_grow_yoy'].replace([np.inf, -np.inf], np.nan).dropna()
            ren_growth_data = df_filtered['Renewables_grow_yoy'].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(co2_growth_data) > 0 and len(ren_growth_data) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_co2_growth = co2_growth_data.mean()
                    st.metric("Avg CO₂ Growth Rate", f"{avg_co2_growth:.2f}%")
                
                with col2:
                    avg_ren_growth = ren_growth_data.mean()
                    st.metric("Avg Renewables Growth", f"{avg_ren_growth:.2f}%")
                
                with col3:
                    co2_volatility = co2_growth_data.std()
                    if np.isnan(co2_volatility) or np.isinf(co2_volatility):
                        co2_volatility = 0
                    st.metric("CO₂ Growth Volatility", f"{co2_volatility:.2f}%")
                
                with col4:
                    ren_volatility = ren_growth_data.std()
                    if np.isnan(ren_volatility) or np.isinf(ren_volatility):
                        ren_volatility = 0
                    st.metric("Renewables Volatility", f"{ren_volatility:.2f}%")
        else:
            st.info("Growth rate analysis not available - insufficient clean data.")
    
    # Carbon efficiency analysis
    st.markdown("### Carbon Efficiency Analysis")
    if 'CO2_per_renewable_unit' in df_filtered.columns:
        efficiency_fig = create_efficiency_analysis(df_filtered)
        if efficiency_fig:
            st.plotly_chart(efficiency_fig, use_container_width=True)
            
            # Efficiency insights with safety checks
            efficiency_data = df_filtered['CO2_per_renewable_unit'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(efficiency_data) > 1:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    current_efficiency = efficiency_data.iloc[-1]
                    st.metric("Current Efficiency", f"{current_efficiency:.3f}")
                
                with col2:
                    if efficiency_data.iloc[0] != 0:
                        efficiency_change = ((efficiency_data.iloc[-1] - efficiency_data.iloc[0]) / abs(efficiency_data.iloc[0]) * 100)
                    else:
                        efficiency_change = 0
                    st.metric("Total Change", f"{efficiency_change:+.1f}%", delta=f"{efficiency_change:+.1f}%")
                
                with col3:
                    efficiency_trend = np.polyfit(range(len(efficiency_data)), efficiency_data, 1)[0]
                    trend_direction = "Improving" if efficiency_trend < 0 else "Worsening"
                    st.metric("Trend Direction", trend_direction)
        else:
            st.info("Carbon efficiency analysis not available - insufficient clean data.")
    else:
        st.info("CO₂ per renewable unit data not available for detailed efficiency analysis.")
    
    # Advanced growth pattern analysis
    st.markdown("### Growth Pattern Analysis")
    
    if 'CO2_grow_yoy' in df_filtered.columns or 'Renewables_grow_yoy' in df_filtered.columns:
        pattern_fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Growth Rate Distribution", "Growth Momentum Analysis"),
            specs=[[{"type": "xy"}, {"type": "xy"}]]
        )
        
        # Distribution analysis
        if 'CO2_grow_yoy' in df_filtered.columns:
            co2_growth_clean = df_filtered['CO2_grow_yoy'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(co2_growth_clean) > 0:
                pattern_fig.add_trace(
                    go.Histogram(x=co2_growth_clean, name="CO₂ Growth Rate", 
                               opacity=0.7, nbinsx=10),
                    row=1, col=1
                )
        
        if 'Renewables_grow_yoy' in df_filtered.columns:
            ren_growth_clean = df_filtered['Renewables_grow_yoy'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(ren_growth_clean) > 0:
                pattern_fig.add_trace(
                    go.Histogram(x=ren_growth_clean, name="Renewables Growth Rate", 
                               opacity=0.7, nbinsx=10),
                    row=1, col=1
                )
        
        # Momentum analysis (rolling averages)
        if 'CO2_grow_yoy' in df_filtered.columns and len(df_filtered) > 3:
            df_temp = df_filtered.copy()
            df_temp['CO2_grow_yoy'] = df_temp['CO2_grow_yoy'].replace([np.inf, -np.inf], np.nan)
            rolling_co2 = df_temp.set_index('Year')['CO2_grow_yoy'].rolling(window=3, center=True).mean()
            rolling_co2_clean = rolling_co2.dropna()
            if len(rolling_co2_clean) > 0:
                pattern_fig.add_trace(
                    go.Scatter(x=rolling_co2_clean.index, y=rolling_co2_clean.values,
                              mode='lines', name='CO₂ 3-Year Rolling Avg'),
                    row=1, col=2
                )
        
        if 'Renewables_grow_yoy' in df_filtered.columns and len(df_filtered) > 3:
            df_temp = df_filtered.copy()
            df_temp['Renewables_grow_yoy'] = df_temp['Renewables_grow_yoy'].replace([np.inf, -np.inf], np.nan)
            rolling_ren = df_temp.set_index('Year')['Renewables_grow_yoy'].rolling(window=3, center=True).mean()
            rolling_ren_clean = rolling_ren.dropna()
            if len(rolling_ren_clean) > 0:
                pattern_fig.add_trace(
                    go.Scatter(x=rolling_ren_clean.index, y=rolling_ren_clean.values,
                              mode='lines', name='Renewables 3-Year Rolling Avg'),
                    row=1, col=2
                )
        
        pattern_fig.update_layout(height=500, title="Advanced Growth Pattern Analysis")
        pattern_fig.update_xaxes(title_text="Growth Rate (%)", row=1, col=1)
        pattern_fig.update_xaxes(title_text="Year", row=1, col=2)
        pattern_fig.update_yaxes(title_text="Frequency", row=1, col=1)
        pattern_fig.update_yaxes(title_text="Growth Rate (%)", row=1, col=2)
        
        st.plotly_chart(pattern_fig, use_container_width=True)
        
        # Growth insights summary
        st.markdown("#### Growth Pattern Summary")
        
        growth_insights = []
        
        if 'CO2_grow_yoy' in df_filtered.columns:
            co2_growth_clean = df_filtered['CO2_grow_yoy'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(co2_growth_clean) > 0:
                co2_mean = co2_growth_clean.mean()
                co2_volatility = co2_growth_clean.std()
                if np.isnan(co2_volatility) or np.isinf(co2_volatility):
                    co2_volatility = 0
                growth_insights.append(f"**CO₂ Growth**: Average {co2_mean:.1f}% per year with {co2_volatility:.1f}% volatility")
        
        if 'Renewables_grow_yoy' in df_filtered.columns:
            ren_growth_clean = df_filtered['Renewables_grow_yoy'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(ren_growth_clean) > 0:
                ren_mean = ren_growth_clean.mean()
                ren_volatility = ren_growth_clean.std()
                if np.isnan(ren_volatility) or np.isinf(ren_volatility):
                    ren_volatility = 0
                growth_insights.append(f"**Renewables Growth**: Average {ren_mean:.1f}% per year with {ren_volatility:.1f}% volatility")
        
        for insight in growth_insights:
            st.write(f"• {insight}")
    
    else:
        st.info("Growth rate columns not available in the dataset.")

# Tab 4 - Correlation Analysis
with tab4:
    st.subheader("🔗 Correlation & Relationship Analysis")
    
    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(df_filtered)
    
    if not corr_matrix.empty:
        # Display correlation heatmap
        col1, col2 = st.columns([2, 1])
        
        with col1:
            corr_fig = create_correlation_heatmap(corr_matrix)
            st.plotly_chart(corr_fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Key Correlations")
            
            # Find strongest correlations (excluding self-correlations)
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    var1 = corr_matrix.columns[i]
                    var2 = corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_val) and not np.isinf(corr_val):
                        corr_pairs.append((var1, var2, abs(corr_val), corr_val))
            
            # Sort by absolute correlation value
            corr_pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Display top correlations
            for i, (var1, var2, abs_corr, corr) in enumerate(corr_pairs[:5]):
                emoji = "🔴" if corr > 0.7 else "🟡" if corr > 0.4 else "🟢" if corr > 0.2 else "⚪"
                direction = "📈" if corr > 0 else "📉"
                st.write(f"{emoji} {direction} **{var1}** vs **{var2}**: {corr:.3f}")
        
        # Advanced correlation insights
        st.markdown("### 🧠 Correlation Insights")
        
        correlation_insights = []
        
        # Check key relationships
        if 'CO2' in corr_matrix.columns and 'Renewables_equivalent_primary_energy' in corr_matrix.columns:
            co2_ren_corr = corr_matrix.loc['CO2', 'Renewables_equivalent_primary_energy']
            if not np.isnan(co2_ren_corr):
                if co2_ren_corr < -0.5:
                    correlation_insights.append("🌱 **Strong Negative Correlation**: Higher renewable energy is strongly associated with lower CO₂ emissions - excellent decoupling!")
                elif co2_ren_corr > 0.5:
                    correlation_insights.append("⚠️ **Coupling Effect**: CO₂ and renewables are positively correlated - may indicate energy transition challenges.")
                else:
                    correlation_insights.append("🔄 **Weak Relationship**: CO₂ and renewables show weak correlation - other factors may dominate emissions.")
        
        if 'CO2_grow_yoy' in corr_matrix.columns and 'Renewables_grow_yoy' in corr_matrix.columns:
            growth_corr = corr_matrix.loc['CO2_grow_yoy', 'Renewables_grow_yoy']
            if not np.isnan(growth_corr):
                if growth_corr < -0.3:
                    correlation_insights.append("📉 **Growth Rate Decoupling**: Faster renewable growth is associated with slower CO₂ growth - positive trend!")
                elif growth_corr > 0.3:
                    correlation_insights.append("📈 **Synchronized Growth**: CO₂ and renewable growth rates move together - may indicate economic coupling.")
        
        if 'CO2_per_renewable_unit' in corr_matrix.columns and 'Year' in df_filtered.columns:
            # Calculate correlation with time
            efficiency_time_data = df_filtered[['Year', 'CO2_per_renewable_unit']].dropna()
            if len(efficiency_time_data) > 3:
                time_corr, _ = pearsonr(efficiency_time_data['Year'], efficiency_time_data['CO2_per_renewable_unit'])
                if not np.isnan(time_corr):
                    if time_corr < -0.3:
                        correlation_insights.append("⚡ **Improving Efficiency Over Time**: CO₂ per renewable unit is decreasing - technology advancement visible!")
                    elif time_corr > 0.3:
                        correlation_insights.append("🔋 **Efficiency Challenges**: CO₂ per renewable unit is increasing over time - need for technology improvements.")
        
        for insight in correlation_insights:
            st.info(insight)
        
        # Statistical significance testing
        with st.expander("📊 Statistical Significance Analysis"):
            st.markdown("**P-values for Key Correlations** (< 0.05 indicates statistical significance)")
            
            significance_results = []
            for var1 in corr_matrix.columns:
                for var2 in corr_matrix.columns:
                    if var1 != var2:
                        data1 = df_filtered[var1].replace([np.inf, -np.inf], np.nan).dropna()
                        data2 = df_filtered[var2].replace([np.inf, -np.inf], np.nan).dropna()
                        
                        # Align the data
                        common_indices = data1.index.intersection(data2.index)
                        if len(common_indices) > 3:
                            aligned_data1 = data1.loc[common_indices]
                            aligned_data2 = data2.loc[common_indices]
                            
                            try:
                                _, p_value = pearsonr(aligned_data1, aligned_data2)
                                if not np.isnan(p_value):
                                    significance = "✅ Significant" if p_value < 0.05 else "❌ Not Significant"
                                    significance_results.append({
                                        'Variable 1': var1,
                                        'Variable 2': var2,
                                        'P-value': f"{p_value:.4f}",
                                        'Significance': significance
                                    })
                            except:
                                pass
            
            if significance_results:
                significance_df = pd.DataFrame(significance_results)
                # Remove duplicate pairs (A-B and B-A)
                significance_df = significance_df[significance_df['Variable 1'] < significance_df['Variable 2']]
                st.dataframe(significance_df, use_container_width=True)
    
    else:
        st.warning("Insufficient numeric data for correlation analysis.")

with tab5:
    st.subheader("📋 Data Explorer & Quality Assessment")
    
    # Data quality dashboard
    st.markdown("### 🔍 Data Quality Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(df_filtered)
        st.metric("Total Records", f"{total_records:,}")
    
    with col2:
        complete_records = df_filtered.dropna().shape[0]
        completeness = (complete_records / total_records * 100) if total_records > 0 else 0
        st.metric("Complete Records", f"{complete_records:,}", f"{completeness:.1f}%")
    
    with col3:
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        st.metric("Numeric Variables", len(numeric_cols))
    
    with col4:
        year_range = df_filtered['Year'].max() - df_filtered['Year'].min()
        st.metric("Year Coverage", f"{year_range:.0f} years")
    
    # Missing data analysis
    st.markdown("### 📊 Missing Data Analysis")
    
    missing_data = df_filtered.isnull().sum()
    missing_pct = (missing_data / len(df_filtered) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Variable': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percentage': missing_pct.values
    }).sort_values('Missing_Percentage', ascending=False)
    
    # Create missing data visualization
    fig_missing = go.Figure(data=[
        go.Bar(
            x=missing_df['Variable'],
            y=missing_df['Missing_Percentage'],
            text=missing_df['Missing_Count'],
            textposition='auto',
            marker_color='lightcoral'
        )
    ])
    
    fig_missing.update_layout(
        title="Missing Data by Variable",
        xaxis_title="Variables",
        yaxis_title="Missing Percentage (%)",
        height=400,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig_missing, use_container_width=True)
    
    # Data quality issues detection
    st.markdown("### ⚠️ Data Quality Issues")
    
    quality_issues = []
    
    for col in df_filtered.select_dtypes(include=[np.number]).columns:
        # Check for infinities
        inf_count = np.isinf(df_filtered[col]).sum()
        if inf_count > 0:
            quality_issues.append(f"🔢 **{col}**: {inf_count} infinite values detected")
        
        # Check for extreme outliers (beyond 3 standard deviations)
        clean_data = df_filtered[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_data) > 3:
            mean_val = clean_data.mean()
            std_val = clean_data.std()
            if std_val > 0:
                outliers = np.abs((clean_data - mean_val) / std_val) > 3
                outlier_count = outliers.sum()
                if outlier_count > 0:
                    quality_issues.append(f"📊 **{col}**: {outlier_count} extreme outliers (>3σ)")
        
        # Check for negative values where they shouldn't exist
        if col in ['CO2', 'Renewables_equivalent_primary_energy', 'Total_GHG']:
            negative_count = (clean_data < 0).sum()
            if negative_count > 0:
                quality_issues.append(f"⚠️ **{col}**: {negative_count} negative values (unexpected)")
    
    if quality_issues:
        for issue in quality_issues:
            st.warning(issue)
    else:
        st.success("✅ No major data quality issues detected!")
    
    # Interactive data explorer
    st.markdown("### 🔍 Interactive Data Explorer")
    
    # Column selector
    display_cols = st.multiselect(
        "Select columns to display",
        df_filtered.columns.tolist(),
        default=df_filtered.columns[:6].tolist()
    )
    
    if display_cols:
        # Year range filter
        min_year, max_year = int(df_filtered['Year'].min()), int(df_filtered['Year'].max())
        year_range_filter = st.slider(
            "Filter by Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
        
        # Apply filters
        filtered_display = df_filtered[
            (df_filtered['Year'] >= year_range_filter[0]) & 
            (df_filtered['Year'] <= year_range_filter[1])
        ][display_cols]
        
        st.dataframe(filtered_display, use_container_width=True, height=400)
        
        # Download filtered data
        csv_download = filtered_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv_download,
            file_name=f"filtered_data_{selected_region}_{selected_country}.csv",
            mime="text/csv"
        )
    
    # Statistical summary
    st.markdown("### 📈 Statistical Summary")
    
    with st.expander("Detailed Statistics"):
        numeric_summary = df_filtered.select_dtypes(include=[np.number]).describe()
        
        # Clean the summary (replace inf/nan)
        for col in numeric_summary.columns:
            numeric_summary[col] = numeric_summary[col].replace([np.inf, -np.inf], np.nan)
        
        st.dataframe(numeric_summary.round(3), use_container_width=True)

# Add footer with app information
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 20px; color: #666;">
        <h4>🌍 Advanced Emissions & Renewables Analytics Platform</h4>
        <p>Powered by Streamlit • Prophet Forecasting • Interactive Visualizations</p>
        <p><i>Built for comprehensive climate data analysis and strategic insights</i></p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Display current selection info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Current Selection")
st.sidebar.info(f"**Region:** {selected_region}\n\n**Country:** {selected_country}\n\n**Data Points:** {len(df_filtered)}")

# Additional sidebar help
with st.sidebar.expander("❓ Help & Tips"):
    st.markdown("""
    **Navigation:**
    - 📊 **Core Dashboard**: Overview metrics
    - 🔮 **Advanced Forecasting**: AI predictions
    - 📈 **Growth & Efficiency**: Trend analysis
    - 🔗 **Correlation Analysis**: Relationships
    - 📋 **Data Explorer**: Raw data & quality
    
    **Tips:**
    - Use the region/country filters above
    - Hover over charts for detailed data
    - Download forecast results as CSV
    - Check data quality in Explorer tab
    """)

# Error handling for edge cases
try:
    # Validate that we have some data to work with
    if df_filtered.empty:
        st.error("No data available for the selected region/country combination.")
        st.info("Please try selecting a different region or country.")
    
    # Check for essential columns
    required_cols = ['Year', 'CO2', 'Renewables_equivalent_primary_energy']
    missing_essential = [col for col in required_cols if col not in df_filtered.columns]
    if missing_essential:
        st.warning(f"Some features may be limited due to missing columns: {missing_essential}")

except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
    st.info("Please refresh the page or contact support if the issue persists.")