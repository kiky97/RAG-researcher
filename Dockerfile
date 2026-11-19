FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-eval.txt ./
ARG WITH_DEEPEVAL=false
RUN if [ "$WITH_DEEPEVAL" = "true" ]; then pip install --no-cache-dir -r requirements-eval.txt; else pip install --no-cache-dir -r requirements.txt; fi
COPY raglab ./raglab
COPY sample_data ./sample_data
RUN useradd --uid 10001 --create-home raglab && mkdir -p /app/data && chown raglab:raglab /app/data
USER raglab
ENV RAGLAB_DATA=/app/data DEEPEVAL_TELEMETRY_OPT_OUT=YES
EXPOSE 8000
CMD ["uvicorn", "raglab.main:app", "--host", "0.0.0.0", "--port", "8000"]
