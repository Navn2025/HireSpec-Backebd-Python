"""
Fix Pinecone dimension mismatch by recreating the index with correct dimension (512)
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone

try:
    from pinecone import ServerlessSpec
except Exception:
    ServerlessSpec = None

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX", "face-auth-index")
CORRECT_DIMENSION = 512  # FaceNet embeddings are 512-dimensional
METRIC = os.getenv("PINECONE_METRIC", "cosine")

api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise RuntimeError("Missing PINECONE_API_KEY in environment")

pc = Pinecone(api_key=api_key)

# Check if index exists
existing = [idx["name"] for idx in pc.list_indexes()]
if INDEX_NAME in existing:
    print(f"Deleting existing index '{INDEX_NAME}' with incorrect dimension...")
    pc.delete_index(INDEX_NAME)
    print("✓ Index deleted")

# Create index with correct dimension
print(f"Creating new index '{INDEX_NAME}' with dimension {CORRECT_DIMENSION}...")
cloud = os.getenv("PINECONE_CLOUD", "aws")
region = os.getenv("PINECONE_REGION", "us-east-1")

if ServerlessSpec is not None:
    pc.create_index(
        name=INDEX_NAME,
        dimension=CORRECT_DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
else:
    pc.create_index(
        name=INDEX_NAME,
        dimension=CORRECT_DIMENSION,
        metric=METRIC,
    )

print(f"✓ Index '{INDEX_NAME}' created successfully with dimension {CORRECT_DIMENSION}")
print("\nYou can now restart your backend server.")
