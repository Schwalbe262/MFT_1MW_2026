"""Run with: python -m regression_260707.monitoring"""

import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "regression_260707.monitoring.app:app",
        host="127.0.0.1",
        port=int(os.environ.get("MFT_MONITOR_PORT", "8010")),
        reload=False,
    )
