import os
from dotenv import load_dotenv
from pathlib import Path

# Find the folder where this test_key.py file is located
BASE_DIR = Path(__file__).resolve().parent

# Explicitly load .env from the same folder
env_path = BASE_DIR / ".env"

print("Looking for .env at:")
print(env_path)

print("\nDoes .env exist?")
print(env_path.exists())

# Load the .env file
load_dotenv(dotenv_path=env_path)

# Get API key
api_key = os.getenv("GOOGLE_API_KEY")

print("\nAPI Key found:")

if api_key:
    print("YES!")
    print(api_key[:10] + "...")
else:
    print("NO - API KEY NOT FOUND")