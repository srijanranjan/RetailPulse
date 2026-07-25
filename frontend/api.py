
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL")
if not API_URL:
    try:
        API_URL = st.secrets["API_URL"]
    except Exception:
        API_URL = "http://localhost:8000"
API_URL = API_URL.rstrip("/")
TIMEOUT = 30


@st.cache_data(ttl=300, show_spinner=False)
def get(path: str, **params):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def post(path: str, payload: dict):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def health() -> bool:
    try:
        requests.get(f"{API_URL}/", timeout=5).raise_for_status()
        return True
    except Exception:
        return False