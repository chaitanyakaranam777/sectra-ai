from pathlib import Path

path = Path("streamlit_app.py")
source = path.read_text(encoding="utf-8")

helper = '''

# ============================================================
# SAFE DATAFRAME CONVERSION - PYARROW COMPATIBILITY
# ============================================================

def safe_display_df(frame):
    """
    Prevent mixed Python objects in dataframe columns from
    causing PyArrow conversion errors in Streamlit.
    Preserve numeric columns; normalize mixed object columns.
    """
    frame = frame.copy()

    for column in frame.columns:
        series = frame[column]

        if series.dtype == "object":
            non_null = series.dropna()
            types = {type(value) for value in non_null}

            if len(types) > 1:
                frame[column] = series.map(
                    lambda value: str(value) if pd.notna(value) else ""
                )

    return frame

'''

marker = '# ============================================================\n# LOAD PROJECT DATA\n# ============================================================'

if 'def safe_display_df(frame):' not in source:
    if marker not in source:
        raise RuntimeError("Could not find the data-loading section.")
    source = source.replace(marker, helper + '\n' + marker, 1)

# Normalize the training-run table before displaying it.
old = '''        run_df = pd.DataFrame(runs)

        a, b, c = st.columns(3)'''

new = '''        run_df = pd.DataFrame(runs)
        run_df = safe_display_df(run_df)

        a, b, c = st.columns(3)'''

if old in source:
    source = source.replace(old, new, 1)

# Normalize comparison table.
old = '''        comparison_df = pd.DataFrame(records)

        if not comparison_df.empty:'''

new = '''        comparison_df = pd.DataFrame(records)
        comparison_df = safe_display_df(comparison_df)

        if not comparison_df.empty:'''

if old in source:
    source = source.replace(old, new, 1)

# Normalize dataset preview.
old = '''                st.dataframe(df.head(5), width="stretch", hide_index=True)'''

new = '''                st.dataframe(
                    safe_display_df(df.head(5)),
                    width="stretch",
                    hide_index=True
                )'''

if old in source:
    source = source.replace(old, new, 1)

# Normalize full dataset table.
old = '''        st.dataframe(selected, width="stretch", hide_index=True)'''

new = '''        st.dataframe(
            safe_display_df(selected),
            width="stretch",
            hide_index=True
        )'''

if old in source:
    source = source.replace(old, new, 1)

# Normalize column information table.
old = '''        st.dataframe(info, width="stretch", hide_index=True)'''

new = '''        st.dataframe(
            safe_display_df(info),
            width="stretch",
            hide_index=True
        )'''

if old in source:
    source = source.replace(old, new, 1)

# Normalize training run table.
old = '''        st.dataframe(run_df, width="stretch", hide_index=True)'''

new = '''        st.dataframe(
            safe_display_df(run_df),
            width="stretch",
            hide_index=True
        )'''

if old in source:
    source = source.replace(old, new, 1)

path.write_text(source, encoding="utf-8")

compile(source, str(path), "exec")
print("SUCCESS: Streamlit file patched and Python syntax verified.")
