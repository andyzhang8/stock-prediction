# test that .env file is where it should be
from dotenv import load_dotenv
import os

# Specify the path to the .env file
env_path = os.path.join(os.path.dirname(__file__), '../env/.env')
print(os.path.abspath(env_path))
