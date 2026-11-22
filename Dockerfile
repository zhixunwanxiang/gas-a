FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 BASELINE_CPU_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
# 赛事镜像仅包含宿主 RPC 运行模块；本地 PyTRiO HTTP
# 和 SwanLab 工具保留在项目源码中，不复制进该镜像。
COPY src/gas_a/__init__.py src/gas_a/__main__.py src/gas_a/cli.py \
     src/gas_a/contest.py src/gas_a/data.py src/gas_a/io.py \
     src/gas_a/model_records.py src/gas_a/monitor.py src/gas_a/optimize.py \
     src/gas_a/controller_policy.py \
     src/gas_a/planning.py \
     src/gas_a/candidates.py src/gas_a/quality.py \
     src/gas_a/robustness.py \
     src/gas_a/blind.py src/gas_a/blind_worker.py \
     src/gas_a/architecture.py src/gas_a/response.py \
     src/gas_a/resilience.py \
     src/gas_a/adaptive.py \
     src/gas_a/pipeline.py src/gas_a/redaction.py src/gas_a/registry.py \
     src/gas_a/review.py src/gas_a/service.py /app/src/gas_a/
COPY src/gas_a/reference/__init__.py src/gas_a/reference/data.py \
     src/gas_a/reference/model.py /app/src/gas_a/reference/
COPY vendor/contest-harness-sdk-1.0.4.tar.gz /opt/contest-harness-sdk-1.0.4.tar.gz
COPY scripts/prune_image.py /tmp/prune_image.py
RUN pip install --no-cache-dir --no-compile numpy==1.26.4 /opt/contest-harness-sdk-1.0.4.tar.gz /app \
    && python3 /tmp/prune_image.py --apply \
    && rm /tmp/prune_image.py
COPY configs/a.json /app/configs/a.json
COPY configs/a.architecture.json /app/configs/a.architecture.json
COPY configs/evidence/response_controls.json /app/configs/evidence/response_controls.json
COPY harness.py contest_submission.py /app/

# 将清理后的文件系统复制到新镜像，避免已删除的构建文件
# 仍留在导出的镜像层中；使用 docker save 导出最终镜像。
FROM scratch
COPY --from=runtime / /
ENV PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 BASELINE_CPU_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
USER 65534:65534
ENTRYPOINT ["python3", "/app/harness.py"]

