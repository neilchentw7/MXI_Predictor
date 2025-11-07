# MXI Predictor

This project contains a Streamlit application for predicting stock market closing prices.

## Running the app

From the project root run:

```bash
streamlit run app.py
```

The included `.streamlit/config.toml` binds the server to `0.0.0.0` on port `5000`, which is compatible with hosted environments such as Replit. If you need a different port locally, either edit the config file or override it on the command line, for example:

```bash
streamlit run app.py --server.port 8501
```

After the server starts, open the URL shown in the terminal.

- **Hosted (e.g., Replit):** use the full URL Streamlit prints, such as `https://<your-domain>.replit.app`.
- **Local machine:** when Streamlit shows `http://0.0.0.0:<port>`, open `http://localhost:<port>` instead. The `0.0.0.0` host only signals that the server listens on every interface; browsers expect `localhost` or your machine's IP address.
