import sys
from src.config import get_settings


def validate_config() -> bool:
    settings = get_settings()

    checks = {
        "Pinecone API Key": bool(settings.pinecone_api_key and settings.pinecone_api_key.strip()),
        "Groq API Key": bool(settings.groq_api_key and settings.groq_api_key.strip()),
        "Tavily API Key": bool(settings.tavily_api_key and settings.tavily_api_key.strip()),
    }

    all_valid = True
    for label, is_configured in checks.items():
        status = "configured" if is_configured else "missing"
        print(f"{label}: {status}")
        if not is_configured:
            all_valid = False

    return all_valid


if __name__ == "__main__":
    if not validate_config():
        sys.exit(1)
