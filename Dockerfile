FROM python:3.12-slim

WORKDIR /app

# Copy setup.py and requirements.txt first (for layer caching)
COPY setup.py requirements.txt ./

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Now copy the rest of your code
COPY . .

# Run the app
CMD ["python3", "app.py"]
