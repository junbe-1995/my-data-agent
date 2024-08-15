import multiprocessing

from my_data_backend.config import config as c

from my_data_backend.utils.env_loader import load_env

bind = "0.0.0.0:9001"
workers = load_env(
    "WSGI_NUM_PROCESS", str(multiprocessing.cpu_count() * 2 + 1), as_type=int
)

wsgi_app = "my_data_backend:app"
worker_class = "uvicorn.workers.UvicornWorker"
loglevel = c.LOG_LEVEL

timeout = load_env("WSGI_TIMEOUT", "60", as_type=int)
graceful_timeout = load_env("WSGI_GRACEFUL_TIMEOUT", "30", as_type=int)
