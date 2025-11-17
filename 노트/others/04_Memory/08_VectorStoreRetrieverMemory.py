import json
import os
from pathlib import Path
 
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import VectorStoreRetrieverMemory
from langchain_core.load import load
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
 
load_dotenv()