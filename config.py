import dotenv
import os

dotenv.load_dotenv()

class Config:

    def __init__(self):
        self.OPEN_ROUTER_KEY = os.getenv('OPEN_ROUTER_KEY')
        self.EMBEDDING_SOURCE = os.getenv('EMBEDDING_SOURCE')
        self.EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')
