#!/bin/zsh
cd "$(dirname "$0")"
if [[ -x .venv/bin/python ]]; then
 .venv/bin/python -m streamlit run app.py --server.port 8506
elif [[ -x ../../work/solar-venv/bin/python ]]; then
 ../../work/solar-venv/bin/python -m streamlit run app.py --server.port 8506
else
 echo "Follow README.md to install the environment first."
 read
fi
