FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# 可选 SOTA 后端（失败不影响系统）
RUN pip install --no-cache-dir river || true

CMD ["python", "-m", "onlineforge.cli"]
