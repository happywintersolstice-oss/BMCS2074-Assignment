"""
Thin Streamlit entry point.

What this file does:
- starts the Streamlit app
- sets the browser page title and layout
- hands control to the main UI module

How it works:
- `main()` configures Streamlit once
- then it calls `render_app()` from the e-wallet UI module
- keeping this file small makes the project easier to understand
"""

import streamlit as st

from src.ewallet_review_ui import render_app


def main() -> None:
    """
    Start the Streamlit app.
    """
    # Configure the browser page once, then hand rendering to the UI module.
    # Set the top-level Streamlit page settings before any UI is drawn.
    st.set_page_config(
        page_title="E-Wallet Review Classifier",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Render the full application from the dedicated UI module.
    render_app()


if __name__ == "__main__":
    main()
