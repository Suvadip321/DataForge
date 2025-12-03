import streamlit as st
import matplotlib.pyplot as plt
from dataforge.eda import (
    get_head_tail,
    get_column_info, 
    get_descriptive_stats, 
    get_data_quality_report, 
    get_value_counts,
    plot_univariate_distribution,
    plot_bivariate_analysis,
    plot_correlation_matrix
)

st.set_page_config(page_title="EDA", page_icon="🔍", layout="wide")

if 'df_full' not in st.session_state or st.session_state.df_full is None:
    st.warning("Please upload data on the Home page first!")
    st.stop()

st.title("🔍 Exploratory Data Analysis")

df = st.session_state.df_full
all_columns = df.columns.tolist()

tab_report, tab_uni, tab_bi = st.tabs(["Data Reports & Quality", "Univariate Plotting", "Bivariate Plotting"])

# Tab 1
with tab_report:
    st.subheader("1. Structural & Quality Reports")
            
    st.markdown("#### Head/Tail View")
    head_or_tail = st.radio("View:", ('Head', 'Tail'))
    n_rows_view = st.slider("Rows:", min_value=1, max_value=10, value=5)

    if head_or_tail == 'Head':
        st.dataframe(get_head_tail(st.session_state.df_full, n_rows_view, True))
    else:
        st.dataframe(get_head_tail(st.session_state.df_full, n_rows_view, False))

    st.markdown("#### Column Data Types & Missingness")
    info_df = get_column_info(st.session_state.df_full)
    st.dataframe(info_df, use_container_width=True)

    st.markdown("#### Descriptive Statistics (Numerical)")
    stats_df = get_descriptive_stats(st.session_state.df_full)
    st.dataframe(stats_df, use_container_width=True)

    st.markdown("#### Value Counts Report")
    value_col = st.selectbox("Select column for Value Counts:", all_columns)
    if value_col:
        st.dataframe(get_value_counts(st.session_state.df_full, value_col))

    st.markdown("#### Data Quality Overview")
    quality_report = get_data_quality_report(st.session_state.df_full)
    st.json(quality_report)

# Tab 2
with tab_uni:
    st.subheader("2. Univariate Distributions")

    col1, col2 = st.columns([2, 1])

    with col1:
        plot_col_uni = st.selectbox("Select column to plot:", all_columns, key='uni_col')

    with col2:
        hue_col_uni = st.selectbox(
            "Hue (optional):", 
            ["None"] + all_columns, 
            key='uni_hue'
        )
        hue_col_uni = None if hue_col_uni == "None" else hue_col_uni

    def get_univariate_plot(df, col, hue):
        return plot_univariate_distribution(df, col, hue)

    if plot_col_uni:
        fig = get_univariate_plot(st.session_state.df_full, plot_col_uni, hue_col_uni)
        if fig:
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        else:
            st.warning(f"Cannot generate plot for column {plot_col_uni}.")

# Tab 3
with tab_bi:
    st.subheader("3. Bivariate Relationships & Correlation")
    st.markdown("#### Feature Correlation Matrix")
    
    fig_corr = plot_correlation_matrix(st.session_state.df_full)
    if fig_corr:
        st.pyplot(fig_corr, clear_figure=True)
        plt.close(fig_corr) 
    else:
        st.info("Not enough numerical data to generate correlation matrix.")
    
    st.markdown("#### Customizable Bivariate Plot")
    
    plot_type = st.selectbox("Plot Type:", ('scatter', 'boxplot', 'violin'), key='bi_type')
    col_x = st.selectbox("X-Axis Column:", all_columns, key='bi_x')
    col_y = st.selectbox("Y-Axis Column:", all_columns, key='bi_y')
    
    all_columns_plus_none = ['None'] + all_columns
    col_hue = st.selectbox("Hue/Color Grouping:", all_columns_plus_none, index=0, key='bi_hue')
    
    col_hue = None if col_hue == "None" else col_hue

    if st.button("Generate Bivariate Plot"):
        fig_bi = plot_bivariate_analysis(st.session_state.df_full, col_x, col_y, plot_type, col_hue)
        if fig_bi:
            st.pyplot(fig_bi, clear_figure=True)
            plt.close(fig_bi)
        else:
            st.warning("Could not generate plot. Check column selection and plot type compatibility.")
