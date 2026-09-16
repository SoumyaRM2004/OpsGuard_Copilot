from pathlib import Path

# List of folders
folders = [
    "src",
    "documents",
    "templates",
    "static",
    "uploads",
    "data",
]

# List of files
files = [
    # Root files
    "app.py",
    "data_ingestion.py",
    "requirements.txt",
    "Dockerfile",
    "docker_compose.yml",
    ".env.example",

    # Source files
    "src/__init__.py",
    "src/config.py",
    "src/db.py",
    "src/ingestion.py",
    "src/models.py",
    "src/self_rag.py",
    "src/vectorstore.py",

    # Frontend
    "templates/index.html",
    "static/style.css",
    "static/app.js",
]


def create_project_structure():
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

    for file in files:
        file_path = Path(file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch(exist_ok=True)

    print("Project folders and files created successfully!")


if __name__ == "__main__":
    create_project_structure()