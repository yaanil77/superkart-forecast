# SuperKart deployment files

The Flask API and Streamlit app are in `backend_files/` and `frontend_files/`. The trained model is intentionally not included in Git; it must be available at `backend_files/superkart_model.joblib` before starting the containers.

## Prepare and run in Codespaces

After training the final model in the notebook, save it to the expected location:

```python
joblib.dump(final_model, "backend_files/superkart_model.joblib")
```

From the repository root, build and start both services:

```bash
docker compose up --build
```

In the Codespaces **Ports** tab, make port `7860` public and copy its forwarded URL. The API health endpoint is `/`; predictions use `/v1/predict` and `/v1/predictbatch`. The Streamlit UI is on port `8501`.

## Retrieve the files from Google Colab

Colab runs in a separate environment and cannot read the Codespace filesystem directly. Publish this workspace to a GitHub repository first, then clone that repository from a Colab cell:

```python
!git clone https://github.com/<owner>/<repository>.git
%cd <repository>
```

The deployment files will then be available in `backend_files/` and `frontend_files/` in the Colab runtime. For a private repository, authenticate with GitHub before cloning. Do not put GitHub tokens or other secrets in a notebook cell.

To call the running Codespaces API from Colab, set `model_root_url` to the public forwarded URL copied from the Codespaces Ports tab, then append `/v1/predict` or `/v1/predictbatch`. Colab must use the forwarded Codespaces URL, not the Docker-only address `http://backend:7860`.