import uvicorn

from my_data_backend.config import config


if __name__ == "__main__":
    if config.DEBUG:
        uvicorn.run(
            "my_data_backend:app",
            host="0.0.0.0",
            port=9001,
            log_level="debug",
            workers=1,
            loop="asyncio",
        )
    else:
        uvicorn.run("my_data_backend:app", host="0.0.0.0", port=9001)
