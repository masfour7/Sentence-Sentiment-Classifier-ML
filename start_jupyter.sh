#!/bin/bash
jupyter notebook \
  --ip=0.0.0.0 \
  --port=5000 \
  --no-browser \
  --NotebookApp.token='' \
  --NotebookApp.password='' \
  --NotebookApp.allow_origin='*' \
  --NotebookApp.disable_check_xsrf=True \
  --NotebookApp.allow_remote_access=True
