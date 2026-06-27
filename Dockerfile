FROM python:3.12-slim

WORKDIR /app

# Copy packaging files first for layer caching.
COPY pyproject.toml README.md ./

# Install dependencies
RUN pip install --upgrade pip && pip install .

# Now copy the rest of your code
COPY . .

# Run the app
CMD ["python3", "app.py"]
