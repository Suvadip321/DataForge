import streamlit as st
import pandas as pd
from dataforge import preprocessing as prep

st.set_page_config(page_title="Preprocessing", page_icon="⚙️", layout="wide")

if 'df_full' not in st.session_state or st.session_state.df_full is None:
    st.warning("⚠️ No data loaded! Please upload a dataset on the **Home** page first.")
    st.stop()

st.title("⚙️ Feature Engineering & Preprocessing")

df = st.session_state.df_full
st.markdown(f"**Current Data Shape:** `{df.shape[0]} Rows`, `{df.shape[1]} Columns`")

keys_to_init = ['X_train', 'X_test', 'y_train', 'y_test', 'scaler_model', 'encoder_model']
for key in keys_to_init:
    if key not in st.session_state:
        st.session_state[key] = None

def reset_pipeline_state():
    """Invalidate downstream steps (split/scale) if upstream data (features) changes."""
    for key in keys_to_init:
        st.session_state[key] = None
    st.toast("Data modified. Pipeline steps reset.", icon="🔄")

tab_create, tab_encode, tab_split, tab_scale = st.tabs([
    "1. Feature Creation", 
    "2. Encoding", 
    "3. Train/Test Split", 
    "4. Scaling"
])

# Tab 1: Feature Engg
with tab_create:
    st.subheader("Create New Features")
    c1, c2 = st.columns(2)
    
    # Date Extraction
    with c1:
        st.markdown("#### 📅 Date Components")
        date_cols = df.select_dtypes(include=['datetime', 'object']).columns.tolist()
        
        if not date_cols:
            st.info("No date-like columns detected.")
        else:
            date_col = st.selectbox("Select Date Column:", date_cols, key="feat_date_col")
            feats = st.multiselect("Extract:", ['year', 'month', 'day', 'weekday', 'quarter'], default=['year', 'month'])
            
            if st.button("Extract Features", key="btn_date"):
                try:
                    st.session_state.df_full = prep.extract_date_features(df, date_col, feats)
                    reset_pipeline_state()
                    st.success(f"Extracted {len(feats)} features from '{date_col}'!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Extraction failed: {e}")

    # Numeric Interactions
    with c2:
        st.markdown("#### ➗ Interaction")
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(num_cols) < 2:
            st.info("Need 2+ numeric columns.")
        else:
            col1 = st.selectbox("Col A:", num_cols, key="feat_c1")
            col2 = st.selectbox("Col B:", num_cols, key="feat_c2")
            op = st.selectbox("Op:", ['add', 'subtract', 'multiply', 'divide'], key="feat_op")
            
            if st.button("Create Interaction", key="btn_interact"):
                try:
                    st.session_state.df_full = prep.create_interaction_features(df, col1, col2, op)
                    reset_pipeline_state()
                    st.success(f"Created feature: {col1}_{op}_{col2}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Interaction failed: {e}")

# Tab 2: Encoding
with tab_encode:
    st.subheader("Categorical Encoding")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not cat_cols:
        st.info("No categorical columns found.")
    else:
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### One-Hot Encoding")
            cols_ohe = st.multiselect("Select Columns:", cat_cols, key="ohe_cols")
            drop_first = st.checkbox("Drop First?", value=True, help="Recommended for Linear Models to avoid multicollinearity.")
            
            if st.button("Apply OHE", key="btn_ohe"):
                try:
                    # We ignore the fitted encoder for this UI preview step, as we are transforming the whole dataset
                    new_df, encoder = prep.one_hot_encode(df, cols_ohe, drop_first)
                    st.session_state.df_full = new_df
                    st.session_state.ohe_encoder = encoder
                    reset_pipeline_state()
                    st.success(f"One-Hot Encoding Applied to {len(cols_ohe)} columns!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Encoding failed: {e}")

        with c2:
            st.markdown("#### Ordinal Encoding")
            cols_le = st.multiselect("Select Columns:", cat_cols, key="le_cols")
            
            if st.button("Apply Ordinal", key="btn_le"):
                try:
                    new_df, encoder = prep.label_encode(df, cols_le)
                    st.session_state.df_full = new_df
                    st.session_state.le_encoder = encoder
                    reset_pipeline_state()
                    st.success(f"Ordinal Encoding Applied to {len(cols_le)} columns!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Encoding failed: {e}")

# Tab 3: Data Splitting
with tab_split:
    st.subheader("Train / Test Split")
    
    all_cols = df.columns.tolist()
    # Auto-detect target if previously set or named 'Target'
    def_idx = all_cols.index('Target') if 'Target' in all_cols else 0
    if st.session_state.y is not None and getattr(st.session_state.y, 'name', '') in all_cols:
        def_idx = all_cols.index(st.session_state.y.name)
        
    c_split_1, c_split_2, c_split_3 = st.columns(3)
    with c_split_1:
        target_col = st.selectbox("Target Variable (y):", all_cols, index=def_idx, key="split_target")
    with c_split_2:
        test_size = st.slider("Test Size:", 0.1, 0.5, 0.2, 0.05)
    with c_split_3:
        stratify = st.checkbox("Stratify Split?", value=False, help="Useful for classification problems.")
        st.write("") 
        if st.button("Perform Split", type="primary", use_container_width=True):
            try:
                # Call Backend
                X_tr, X_te, y_tr, y_te = prep.split_data(
                    df, target_col, test_size=test_size, stratify=stratify
                )
                # Save to Session State
                st.session_state.X_train = X_tr
                st.session_state.X_test = X_te
                st.session_state.y_train = y_tr
                st.session_state.y_test = y_te
                st.session_state.y = y_tr # Update global target reference
                
                st.success(f"✅ Data Split! Train: {X_tr.shape[0]}, Test: {X_te.shape[0]}")
                st.rerun()
            except Exception as e:
                st.error(f"Split Failed: {e}")

    # Show Split Confirmation
    if st.session_state.X_train is not None:
        st.info("✅ Data is split and ready for scaling.")

# Scaling
with tab_scale:
    st.subheader("Feature Scaling")
    
    if st.session_state.X_train is None:
        st.warning("⚠️ You must split the data in **Tab 3** before scaling to prevent data leakage.")
    else:
        X_tr = st.session_state.X_train
        X_te = st.session_state.X_test
        
        # Only show numerical columns present in the training set
        num_cols_train = X_tr.select_dtypes(include=['number']).columns.tolist()
        
        c1, c2 = st.columns([3, 1])
        with c1:
            cols_to_scale = st.multiselect("Columns to Scale:", num_cols_train, default=num_cols_train)
        with c2:
            method = st.selectbox("Method:", ['standard', 'minmax', 'robust'])
            st.write("")
            if st.button("Apply Scaling", key="btn_scale", use_container_width=True):
                try:
                    # 1. Fit & Transform Train
                    X_tr_sc, scaler = prep.scale_numerical_columns(X_tr, cols_to_scale, method)
                    
                    # 2. Transform Test (using fitted scaler from Train)
                    X_te_sc, _ = prep.scale_numerical_columns(X_te, cols_to_scale, method, fitted_scaler=scaler)
                    
                    # 3. Save State
                    st.session_state.X_train = X_tr_sc
                    st.session_state.X_test = X_te_sc
                    st.session_state.scaler_model = scaler
                    
                    st.success(f"Scaled {len(cols_to_scale)} columns using {method}.")
                    st.caption("Note: Scaler fitted on Train set and applied to Test set to prevent leakage.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Scaling failed: {e}")

# Preview & Download
st.divider()
st.subheader("Preview & Export")

if st.session_state.X_train is not None:
    col1, col2 = st.columns(2)
    # Helper function to safely combine X and y
    def safe_combine(X, y):
        df_export = X.copy()
        # Robustness: Ensure target has a name, default to 'Target' if None
        target_name = y.name if y.name else 'Target'
        df_export[target_name] = y
        return df_export

    with col1:
        st.markdown("##### Training Data (Processed)")
        st.dataframe(st.session_state.X_train.head(), use_container_width=True)
        
        # Combine safely
        train_export = safe_combine(st.session_state.X_train, st.session_state.y_train)
        csv_train = train_export.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Processed Train Set",
            data=csv_train,
            file_name="train_processed.csv",
            mime="text/csv",
            key='dl_train'
        )
    with col2:
        st.markdown("##### Test Data (Processed)")
        st.dataframe(st.session_state.X_test.head(), use_container_width=True)
        
        # Combine safely
        test_export = safe_combine(st.session_state.X_test, st.session_state.y_test)
        csv_test = test_export.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Processed Test Set",
            data=csv_test,
            file_name="test_processed.csv",
            mime="text/csv",
            key='dl_test'
        )
else:
    st.info("Complete the **Split** and **Scaling** steps to enable download.")
    st.subheader("Current Full Data")
    st.dataframe(st.session_state.df_full.head(), use_container_width=True)


    