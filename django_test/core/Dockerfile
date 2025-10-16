FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev netcat-traditional curl \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y locales

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN mkdir -p /vol/static /vol/media

RUN useradd -m webuser || true
RUN chown -R webuser:webuser /app /vol
USER webuser

ENV DJANGO_SETTINGS_MODULE=core.settings
ENV PORT=8000

EXPOSE 8000

CMD ["gunicorn", "core.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "3", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]

