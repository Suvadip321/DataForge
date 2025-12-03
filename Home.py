import streamlit as st
import pathlib
from dataforge.data_loader import load_csv 

st.set_page_config(
    page_title="DataForge", 
    page_icon="📊", 
    layout="wide"
)

st.title("📊 DataForge:ML Workflow Engine")
st.markdown("### Welcome! Upload your data to get started.")

# Session State Init
if 'df_full' not in st.session_state:
    st.session_state.df_full = None 
if 'X' not in st.session_state:
    st.session_state.X = None       
if 'y' not in st.session_state:
    st.session_state.y = None       
if 'file_name' not in st.session_state:
    st.session_state.file_name = None 

# Sidebar: Data Ingestion
with st.sidebar:
    st.header("1. Data Ingestion")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    data_path = None # Default state is no path

    if uploaded_file:
        # Save to temp directory so backend can read it via path
        temp_dir = pathlib.Path("./temp_upload")
        temp_dir.mkdir(exist_ok=True)
        data_path = temp_dir / uploaded_file.name
        data_path.write_bytes(uploaded_file.read())
        
        # Reset state if a new file is uploaded
        if st.session_state.file_name != uploaded_file.name:
            st.session_state.df_full = None
            st.session_state.X = None
            st.session_state.y = None
            st.session_state.file_name = uploaded_file.name
    else:
        # Inform user to upload instead of loading sample data
        st.info("Please upload a CSV file to proceed.")

# Load Logic
# Check if data_path is set before trying to load
if st.session_state.df_full is None and data_path is not None and data_path.exists():
    try:
        # Load the full dataframe
        st.session_state.df_full, _ = load_csv(data_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Main Dashboard: Data Preview
if st.session_state.df_full is not None:
    st.success(f"Data Loaded: `{st.session_state.file_name}`")
    st.markdown("#### Raw Data Preview")
    st.dataframe(st.session_state.df_full.head(10), use_container_width=True)
    
    st.divider()
    st.info("Use the **Sidebar** to navigate to **EDA** or **Cleaning** to process this data.")
else:
    # Show a welcome/instruction message when no data is loaded
    st.write("---")
    st.warning("No dataset loaded. Upload a file in the sidebar to unlock the workflow.")
