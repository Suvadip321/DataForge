import streamlit as st
import pandas as pd
from dataforge import clean

st.set_page_config(page_title="Cleaning", page_icon="🧼", layout="wide")

if 'df_full' not in st.session_state or st.session_state.df_full is None:
    st.warning("⚠️ No data loaded! Please upload a dataset on the **Home** page first.")
    st.stop()

st.title("🧹 Data Cleaning Studio")

df = st.session_state.df_full
st.markdown(f"**Current Data Shape:** `{df.shape[0]} Rows`, `{df.shape[1]} Columns`")

# 1. Structural fixes
with st.expander("1. Structural Fixes (Duplicates, Drop, Rename)", expanded=True):
    tab_dup, tab_drop, tab_rename = st.tabs(["Duplicates", "Drop Columns", "Rename Columns"])
    
    # Remove Duplicates
    with tab_dup:
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            st.warning(f"Found **{dup_count}** duplicate rows.")
            if st.button("Remove Duplicates", key="btn_dedup"):
                st.session_state.df_full = clean.remove_duplicates(df)
                st.success(f"Removed {dup_count} duplicates!")
                st.rerun()
        else:
            st.success("No duplicate rows found.")

    # Drop Columns
    with tab_drop:
        st.write("Select columns to remove permanently.")
        cols_to_drop = st.multiselect("Select Columns:", df.columns)
        if cols_to_drop:
            if st.button(f"Drop {len(cols_to_drop)} Columns", key="btn_drop"):
                st.session_state.df_full = clean.drop_columns(df, cols_to_drop)
                st.success(f"Dropped: {', '.join(cols_to_drop)}")
                st.rerun()

    # Rename Columns
    with tab_rename:
        st.write("Rename specific columns.")
        c1, c2 = st.columns(2)
        with c1:
            col_to_rename = st.selectbox("Select Column:", df.columns, key="rename_src")
        with c2:
            new_name = st.text_input("New Name:", key="rename_dest")
            
        if st.button("Rename", key="btn_rename"):
            if new_name in df.columns:
                st.error(f"Error: A column named '{new_name}' already exists.")
            elif new_name and new_name != col_to_rename:
                st.session_state.df_full = clean.rename_columns(df, {col_to_rename: new_name})
                st.success(f"Renamed '{col_to_rename}' to '{new_name}'")
                st.rerun()
            else:
                st.warning("Please enter a valid new name.")

# 2. Type conversion
with st.expander("2. Data Type Conversion"):
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        col_type = st.selectbox("Select Column:", df.columns, key="type_col")
        col_data = df[col_type]
        if isinstance(col_data, pd.DataFrame):
            current_type = col_data.dtypes.iloc[0]
            st.warning(f"⚠️ Multiple columns named '{col_type}'.")
        else:
            current_type = col_data.dtype
        st.caption(f"Current Type: `{current_type}`")
    
    with col2:
        new_type = st.selectbox("Convert To:", ["int", "float", "str", "bool", "datetime"], key="type_target")
    
    with col3:
        st.write("")
        st.write("")
        if st.button("Convert", key="btn_convert"):
            if new_type == "datetime":
                new_df = clean.convert_to_datetime(df, col_type)
            else:
                new_df = clean.change_column_dtype(df, col_type, new_type)

            new_col_data = new_df[col_type]
            if isinstance(new_col_data, pd.DataFrame):
                new_actual_type = new_col_data.dtypes.iloc[0]
            else:
                new_actual_type = new_col_data.dtype

            if str(new_actual_type) == str(current_type):
                 st.warning(f"⚠️ Conversion failed or was not needed. Column '{col_type}' is still `{current_type}`. Check if the data is compatible with '{new_type}'.")
            else:
                st.session_state.df_full = new_df
                st.success(f"Converted '{col_type}' to {new_type}")
                st.rerun()

# 3. Missing values
with st.expander("3. Handle Missing Values"):
    missing_cols = [c for c in df.columns if df[c].isnull().sum() > 0]
    
    if not missing_cols:
        st.success("No missing values found!")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            target_miss = st.selectbox("Select Column:", missing_cols, key="miss_col")
        with c2:
            target_data = df[target_miss]
            if isinstance(target_data, pd.DataFrame):
                 is_num = pd.api.types.is_numeric_dtype(target_data.iloc[:, 0])
            else:
                 is_num = pd.api.types.is_numeric_dtype(target_data)

            opts = ["drop_rows", "mean", "median", "constant"] if is_num else ["drop_rows", "mode", "constant"]
            method = st.selectbox("Imputation Method:", opts, key="miss_method")
        with c3:
            fill_val = None
            if method == 'constant':
                if is_num:
                    fill_val = st.number_input("Fill Value:", value=0.0)
                else:
                    fill_val = st.text_input("Fill Value:", value="Missing")
            else:
                st.write("") 
                st.write("*(Auto-calculated)*")

        if st.button("Apply Imputation", key="btn_impute"):
            st.session_state.df_full = clean.handle_missing_values(
                df, strategy=method, column=target_miss, fill_value=fill_val
            )
            st.success(f"Fixed '{target_miss}' using {method}")
            st.rerun()

# 4. Text cleaning
with st.expander("4. Text Cleaning"):
    text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if not text_cols:
        st.info("No text columns available.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            text_col = st.selectbox("Select Text Column:", text_cols, key="txt_col")
        with c2:
            op = st.selectbox("Operation:", ["strip", "lower", "upper", "title", "replace"], key="txt_op")
        
        old_val_txt = ""
        new_val_txt = ""
        
        with c3:
            if op == "replace":
                old_val_txt = st.text_input("Old Value (Substring):")
                new_val_txt = st.text_input("New Value:")
            else:
                st.write("*(No extra params)*")
            
        if st.button("Apply Text Fix", key="btn_text"):
            if op == "replace" and not old_val_txt:
                st.error("Please provide the substring to replace.")
            else:
                st.session_state.df_full = clean.process_text_column(
                    df, text_col, op, old_val=old_val_txt, new_val=new_val_txt
                )
                st.success(f"Applied '{op}' to '{text_col}'")
                st.rerun()

# 5. Find & replace
with st.expander("5. Find & Replace"):
    st.write("Replace specific values globally or in a specific column.")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        replace_scope = st.selectbox("Scope:", ["Global (All Columns)"] + df.columns.tolist(), key="rep_scope")
    
    with c2:
        find_val = st.text_input("Find Value (Exact Match):", key="val_find")
        
    with c3:
        replace_val = st.text_input("Replace With (Type 'NaN' for null):", key="val_replace")
        
    if st.button("Apply Replacement", key="btn_replace_global"):
        scope_col = None if replace_scope == "Global (All Columns)" else replace_scope
        
        st.session_state.df_full = clean.find_and_replace(df, find_val, replace_val, scope_col)
        
        msg = f"Replaced '{find_val}' with '{replace_val}'"
        if scope_col:
            msg += f" in column '{scope_col}'"
        else:
            msg += " globally"
            
        st.success(msg)
        st.rerun()


# Preview & Download
st.divider()
st.subheader("Final Data Preview")
st.dataframe(st.session_state.df_full.head(), use_container_width=True)

csv_data = st.session_state.df_full.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Cleaned CSV",
    data=csv_data,
    file_name="cleaned_data.csv",
    mime="text/csv"
)
