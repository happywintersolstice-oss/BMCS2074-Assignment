"""
Thin Streamlit entry point.

This file stays intentionally small so the project remains easy to maintain.
"""

import streamlit as st

from src.ui import render_app


def main() -> None:
    """
    Start the Streamlit app.
    """
    st.set_page_config(
        page_title="Scam Message Detector",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_app()


if __name__ == "__main__":
    main()
