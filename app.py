import streamlit as st
import pandas as pd
import numpy as np

st.title("Production Data Analyzer")
st.header("Production Snapshots")
uploaded_file = st.file_uploader("Upload a Production Snapshot Excel file", type=["xlsx"])

# New: Pace Reports uploader
st.header("Pace Reports")
pace_files = st.file_uploader("Upload one or more Pace Report Excel files", type=["xlsx"], accept_multiple_files=True)

if uploaded_file:

    # Read Production Snapshot
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    st.write(f"Found {len(sheet_names)} sheets: {sheet_names}")
    data = {}
    for sheet in sheet_names:
        df = xls.parse(sheet, header=None)
        header_row_idx = None
        for i, row in df.iterrows():
            if any([pd.notnull(cell) and str(cell).strip() != '' for cell in row]):
                header_row_idx = i
                break
        if header_row_idx is None:
            continue
        raw_cols = df.iloc[header_row_idx].tolist()
        new_cols = []
        used = set()
        for idx, col in enumerate(raw_cols):
            col_str = str(col).strip() if pd.notnull(col) else ''
            if idx == 0 and not col_str:
                col_str = "job number"
            elif idx == 1 and not col_str:
                col_str = "client"
            elif not col_str:
                col_str = f"Unnamed_{idx}"
            orig_col_str = col_str
            count = 1
            while col_str in used:
                col_str = f"{orig_col_str}_{count}"
                count += 1
            used.add(col_str)
            new_cols.append(col_str)
        df = df[(header_row_idx+1):]
        df.columns = new_cols
        df = df.reset_index(drop=True)
        data[sheet] = df



    # Attempt to convert all columns to numeric where possible (except __sheet__)
    for sheet in data:
        data[sheet]['__sheet__'] = sheet
        for col in data[sheet].columns:
            if col != '__sheet__':
                data[sheet][col] = pd.to_numeric(data[sheet][col], errors='ignore')
    combined_df = pd.concat(data.values(), ignore_index=True)

    # Utility: robust column finder
    def find_col(df, name):
        name = name.strip().lower()
        for col in df.columns:
            if col.strip().lower() == name:
                return col
        return None

    # --- Pace Reports logic ---
    if pace_files:
        pace_dfs = []
        for pace_file in pace_files:
            pace_xls = pd.ExcelFile(pace_file)
            for sheet in pace_xls.sheet_names:
                pace_df = pace_xls.parse(sheet)
                # Normalize job number column name to 'job number'
                pace_df.columns = [str(c).strip().lower() for c in pace_df.columns]
                if 'job' in pace_df.columns:
                    pace_df = pace_df.rename(columns={'job': 'job number'})
                # Only keep rows with job numbers in the Production Snapshot
                if 'job number' in pace_df.columns:
                    pace_df = pace_df[pace_df['job number'].isin(combined_df['job number'])]
                    pace_dfs.append(pace_df)
        if pace_dfs:
            pace_combined = pd.concat(pace_dfs, ignore_index=True)

            # Merge with combined_df on 'job number'
            merged_df = pd.merge(combined_df, pace_combined, on='job number', how='left', suffixes=('', '_pace'))
            st.subheader('Combined Production Snapshot + Pace Report Data')
            st.dataframe(merged_df)
            # Optionally, allow download
            csv_merged = merged_df.to_csv(index=False).encode('utf-8')
            st.download_button('Download Combined Data as CSV', csv_merged, 'combined_data.csv', 'text/csv')

            # --- CSR Average Days Report ---
            # Find CSR column case-insensitively
            csr_col = None
            for col in pace_combined.columns:
                if col.strip().lower() == 'csr':
                    csr_col = col
                    break
            if csr_col and 'job number' in combined_df.columns:
                st.subheader('Average Number of Days per CSR (from Pace Reports)')
                # Get mapping of job number to CSR from pace_combined
                job_to_csr = pace_combined[['job number', csr_col]].drop_duplicates()
                # Get number of days each job appears in production snapshots
                job_days = combined_df.groupby('job number')['__sheet__'].nunique().reset_index(name='days_on_snapshots')
                # Merge to associate each job with its CSR
                job_csr_days = pd.merge(job_to_csr, job_days, on='job number', how='inner')
                # Group by CSR and calculate average and total jobs
                csr_group = job_csr_days.groupby(csr_col)
                csr_avg_days = csr_group['days_on_snapshots'].mean().reset_index()
                csr_avg_days = csr_avg_days.rename(columns={'days_on_snapshots': 'avg_days_on_snapshots'})
                csr_avg_days['total_jobs'] = csr_group['job number'].nunique().values
                st.dataframe(csr_avg_days)
            elif not csr_col:
                st.warning("No 'CSR' column found in Pace Reports. Please check your file.")

    # Sheet selector and preview
    selected_sheet = st.selectbox("Select a sheet to view", sheet_names)
    st.write(f"Preview of {selected_sheet}")
    st.dataframe(data[selected_sheet].head())

    # Column selector for visualizations
    numeric_cols = combined_df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.subheader("Visualizations")
        col_to_plot = st.selectbox("Select a numeric column to visualize", numeric_cols)
        chart_type = st.radio("Chart type", ["Line", "Bar", "Boxplot", "Heatmap (correlation)"])

        if chart_type == "Line":
            st.line_chart(combined_df[[col_to_plot, '__sheet__']].groupby('__sheet__').mean())
        elif chart_type == "Bar":
            st.bar_chart(combined_df[[col_to_plot, '__sheet__']].groupby('__sheet__').mean())
        elif chart_type == "Boxplot":
            import matplotlib.pyplot as plt
            import seaborn as sns
            fig, ax = plt.subplots()
            sns.boxplot(x='__sheet__', y=col_to_plot, data=combined_df, ax=ax)
            st.pyplot(fig)
        elif chart_type == "Heatmap (correlation)":
            import seaborn as sns
            import matplotlib.pyplot as plt
            corr = combined_df[numeric_cols].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

    # Pattern detection: highlight outliers in selected sheet
    st.subheader("Pattern Detection: Outliers")
    if numeric_cols:
        outlier_col = st.selectbox("Select column for outlier detection", numeric_cols, key='outlier')
        df = data[selected_sheet]
        q1 = df[outlier_col].quantile(0.25)
        q3 = df[outlier_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[outlier_col] < lower) | (df[outlier_col] > upper)]
        st.write(f"Number of outliers in {outlier_col}: {len(outliers)}")
        if not outliers.empty:
            st.dataframe(outliers)

    # Cross-sheet trend analysis
    st.subheader("Cross-Sheet Trend Analysis")
    if numeric_cols:
        trend_col = st.selectbox("Select column for trend analysis", numeric_cols, key='trend')
        trend = combined_df.groupby('__sheet__')[trend_col].mean()
        st.line_chart(trend)


    # Pareto Chart for Causes of Delays/Downtime
    st.subheader("Pareto Chart: Causes of Delays/Downtime")
    # Select a categorical column (e.g., reason/cause)
    cat_cols = combined_df.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        pareto_cat_col = st.selectbox("Select categorical column (e.g., downtime reason)", cat_cols, key='pareto_cat')
        # Optionally select a numeric column to sum (e.g., downtime minutes)
        pareto_num_col = st.selectbox("Select numeric column to sum (optional, else counts)", [None] + numeric_cols, key='pareto_num')
        pareto_df = combined_df.copy()
        if pareto_num_col:
            pareto_data = pareto_df.groupby(pareto_cat_col)[pareto_num_col].sum().sort_values(ascending=False)
        else:
            pareto_data = pareto_df[pareto_cat_col].value_counts()
        pareto_data = pareto_data[pareto_data.index.notnull()]
        pareto_cum = pareto_data.cumsum() / pareto_data.sum() * 100

        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots()
        pareto_data.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_ylabel('Total' + (f' {pareto_num_col}' if pareto_num_col else ' Count'))
        ax2 = ax1.twinx()
        pareto_cum.plot(ax=ax2, color='red', marker='o', linewidth=2)
        ax2.set_ylabel('Cumulative %')
        ax2.set_ylim(0, 110)
        ax2.axhline(80, color='gray', linestyle='dashed', linewidth=1)
        plt.title(f'Pareto Chart for {pareto_cat_col}')
        plt.tight_layout()
        st.pyplot(fig)
        st.write('The bars show the most common causes. The red line shows the cumulative percentage (80/20 rule).')

    # Custom filter/search
    st.subheader("Custom Filter/Search")
    filter_col = st.selectbox("Select column to filter", combined_df.columns)
    unique_vals = combined_df[filter_col].dropna().unique()
    selected_val = st.selectbox("Select value to filter by", unique_vals)
    filtered = combined_df[combined_df[filter_col] == selected_val]
    st.write(f"Filtered rows for {filter_col} = {selected_val}:")
    st.dataframe(filtered)

    # Job number tracking across all days
    st.subheader("Job Number Tracking Across Days")
    if 'job number' in combined_df.columns:
        job_tracking = combined_df.groupby('job number')['__sheet__'].agg(['nunique', list])
        job_tracking = job_tracking.rename(columns={'nunique': 'days_present', 'list': 'days_list'})
        # Last day seen (assuming sheet names are sortable as dates or in order)
        job_tracking['last_day'] = job_tracking['days_list'].apply(lambda x: x[-1] if x else None)
        st.write("Summary of job numbers across all days:")
        st.dataframe(job_tracking[['days_present', 'last_day']])
        # Optionally, let user search for a job number
        job_search = st.text_input("Enter a job number to view its days:")
        # Show all job numbers as strings for user reference
        st.caption(f"All job numbers: {list(job_tracking.index.map(str))}")
        if job_search:
            # Debug: show types and repr
            st.write(f"You entered: {job_search} (type: {type(job_search)}, repr: {repr(job_search)})")
            st.write(f"Index as strings: {list(job_tracking.index.map(str))}")
            def normalize_jobnum(val):
                s = str(val)
                return s[:-2] if s.endswith('.0') else s
            job_index_str = job_tracking.index.map(normalize_jobnum)
            job_index_str_list = list(job_index_str)
            job_search_norm = normalize_jobnum(job_search)
            if job_search_norm in job_index_str_list:
                idx = job_index_str_list.index(job_search_norm)
                real_idx = job_tracking.index[idx]
                st.write(f"Job {job_search} appeared on: {job_tracking.loc[real_idx, 'days_list']}")
                st.write(f"Last day seen: {job_tracking.loc[real_idx, 'last_day']}")
            else:
                st.warning("Job number not found.")

    # --- Production Station Flow ---
    st.subheader('Production Station Flow')
    if 'Production Station' in combined_df.columns:
        station_counts = combined_df['Production Station'].value_counts().sort_values(ascending=False)
        if not station_counts.empty:
            st.bar_chart(station_counts)
            st.write('Number of jobs at each production station.')
        else:
            st.info('No jobs found at any production station.')
    else:
        st.info('Column "Production Station" not found.')

    # --- Job Aging ---
    st.subheader('Job Aging (Longest Open/In Progress)')
    import pandas as pd
    from datetime import datetime
    if 'Admin Status' in combined_df.columns and 'Entered Date' in combined_df.columns:
        # Parse dates
        df_aging = combined_df.copy()
        df_aging['Entered Date'] = pd.to_datetime(df_aging['Entered Date'], errors='coerce')
        today = pd.Timestamp.today()
        df_aging['days_open'] = (today - df_aging['Entered Date']).dt.days
        aging_jobs = df_aging[df_aging['Admin Status'].isin(['Open', 'In Progress'])]
        aging_jobs = aging_jobs[['job number', 'Admin Status', 'Entered Date', 'days_open', 'CSR', 'Production Station']]
        aging_jobs = aging_jobs.sort_values('days_open', ascending=False).head(20)
        if not aging_jobs.empty:
            st.dataframe(aging_jobs)
            st.write('Top 20 jobs that have been open or in progress the longest.')
        else:
            st.info('No open or in progress jobs found.')
    else:
        st.info('Required columns "Admin Status" and/or "Entered Date" not found.')

    # --- Revenue Forecast ---
    st.subheader('Revenue Forecast (Next 30 Days)')
    if 'Expected Production Completion Date' in combined_df.columns and 'Amount to Invoice' in combined_df.columns:
        df_rev = combined_df.copy()
        df_rev['Expected Production Completion Date'] = pd.to_datetime(df_rev['Expected Production Completion Date'], errors='coerce')
        df_rev['Amount to Invoice'] = pd.to_numeric(df_rev['Amount to Invoice'], errors='coerce')
        next_30 = pd.Timestamp.today() + pd.Timedelta(days=30)
        mask = (df_rev['Expected Production Completion Date'] >= pd.Timestamp.today()) & (df_rev['Expected Production Completion Date'] <= next_30)
        forecast = df_rev[mask]['Amount to Invoice'].sum()
        st.metric('Expected Revenue (next 30 days)', f"${forecast:,.2f}")
        st.caption(f"Jobs considered: {df_rev[mask].shape[0]}")
    else:
        st.info('Required columns "Expected Production Completion Date" and/or "Amount to Invoice" not found.')

    # --- Status Distribution ---
    st.subheader('Job Status Distribution')
    if 'Admin Status' in combined_df.columns:
        status_counts = combined_df['Admin Status'].value_counts()
        if not status_counts.empty:
            st.bar_chart(status_counts)
            st.write('Distribution of jobs by status.')
        else:
            st.info('No jobs found for status distribution.')
    else:
        st.info('Column "Admin Status" not found.')

    # --- Custom Alerts: Likely to Pass Promise Date ---
    st.subheader('Custom Alerts: Likely to Pass Promise Date')
    # Robust column matching
    col_promise = find_col(combined_df, 'Promise Date')
    col_due = find_col(combined_df, 'Expected Production Completion Date')
    display_cols = ['job number', col_promise, col_due, 'CSR', 'Production Station', 'Customer', 'Description']
    if col_promise and col_due:
        df_alerts = combined_df.copy()
        df_alerts[col_promise] = pd.to_datetime(df_alerts[col_promise], errors='coerce')
        df_alerts[col_due] = pd.to_datetime(df_alerts[col_due], errors='coerce')
        today = pd.Timestamp.today().normalize()
        # Exclude jobs with promise dates before today
        mask = (df_alerts[col_due] > df_alerts[col_promise]) & (df_alerts[col_promise] >= today)
        likely_late_jobs = df_alerts[mask]
        # Only keep columns that exist in the DataFrame
        display_cols_final = [c for c in display_cols if c and c in df_alerts.columns]
        if not likely_late_jobs.empty:
            st.warning(f"{len(likely_late_jobs)} job(s) are forecasted to miss their Promise Date:")
            st.dataframe(likely_late_jobs[display_cols_final])
        else:
            st.success('No jobs are currently forecasted to miss their Promise Date!')
        st.caption(f"Jobs checked: {df_alerts.shape[0]}")
    else:
        st.info(f'Required columns not found. Columns present: {list(combined_df.columns)}')
else:
    st.info("Please upload a .xlsx file with multiple sheets.")
