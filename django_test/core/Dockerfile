# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# рабочая папка
WORKDIR /app

# системные зависимости (psycopg2 build deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev netcat-traditional curl \
 && rm -rf /var/lib/apt/lists/*

# копируем зависимости
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# копируем проект
COPY . /app

# создаём папки для статик/медиа, которые будут смонтированы в compose
RUN mkdir -p /vol/static /vol/media

# создаём пользователя (не root)
RUN useradd -m webuser || true
RUN chown -R webuser:webuser /app /vol
USER webuser

# переменные окружения по-умолчанию (можно переопределить в docker-compose)
ENV DJANGO_SETTINGS_MODULE=core.settings
ENV PORT=8000

EXPOSE 8000

# запуск gunicorn
CMD ["gunicorn", "core.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "3", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]

