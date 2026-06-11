FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir .
ENV AUTOPILOT_AUTOSTART=1
EXPOSE 8200
CMD ["uvicorn", "cost_autopilot.api:app", "--host", "0.0.0.0", "--port", "8200"]
