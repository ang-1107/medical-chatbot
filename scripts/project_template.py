import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

LIST_OF_FILES = {
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    "src/indexing.py",
    "src/webapp.py",
    ".env",
    "app.py",
    "notebooks/trials.ipynb",
    "scripts/project_template.py",
}


def main() -> None:
    for filepath in LIST_OF_FILES:
        file_path = Path(filepath)
        filedir, filename = os.path.split(file_path)

        if filedir:
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Creating directory {filedir} for the file: {filename}")

        if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
            with open(file_path, "w", encoding="utf-8"):
                logging.info(f"Creating empty file: {file_path}")
        else:
            logging.info(f"{filename} already exists")


if __name__ == "__main__":
    main()
