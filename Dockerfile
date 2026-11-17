FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
WORKDIR /app
ENV PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BASELINE_CPU_THREADS=1
COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
COPY submission/official_sdk_1_0_2 /opt/contest-harness-sdk
RUN pip install --no-cache-dir --no-build-isolation --no-deps /opt/contest-harness-sdk
COPY src/zhixun/__init__.py src/zhixun/data.py src/zhixun/metrics.py src/zhixun/model_extension.py src/zhixun/tracking.py src/zhixun/contest_runtime.py /app/src/zhixun/
COPY src/zhixun/vendor /app/src/zhixun/vendor
COPY submission/official_sdk_1_0_2/template/harness.py /app/harness.py
COPY submission/contest_submission.py /app/contest_submission.py
COPY NOTICE.md SDK_SOURCE.json /app/
USER 65534:65534
ENTRYPOINT ["python3", "/app/harness.py"]
CMD ["describe"]
