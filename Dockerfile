# Вказуємо базовий образ
FROM python:3.10-slim

# Встановлюємо робочий каталог
WORKDIR /app

# Копіюємо файли проекту в контейнер
COPY . .

# Встановлюємо залежності
RUN pip install --no-cache-dir -r requirements.txt

# Відкриваємо порт
EXPOSE 8000

# Команда для запуску сервера
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
