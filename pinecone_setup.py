import os
from dotenv import load_dotenv
from pinecone import Pinecone

try:
    from pinecone import PodSpec
except Exception:
    PodSpec = None

try:
    from pinecone import ServerlessSpec
except Exception:
    ServerlessSpec = None

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX", "face-auth-index")
DIMENSION = int(os.getenv("PINECONE_DIMENSION", "512"))
METRIC = os.getenv("PINECONE_METRIC", "cosine")

api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise RuntimeError("Missing PINECONE_API_KEY in environment")

pc = Pinecone(api_key=api_key)

existing = [idx["name"] for idx in pc.list_indexes()] if hasattr(pc, "list_indexes") else []
if INDEX_NAME not in existing:
    use_pod = os.getenv("PINECONE_USE_POD", "true").lower() == "true"
    if use_pod and PodSpec is not None:
        environment = os.getenv("PINECONE_ENV", "us-west1-gcp")
        pod_type = os.getenv("PINECONE_POD_TYPE", "p1.x1")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=PodSpec(environment=environment, pod_type=pod_type),
        )
    elif ServerlessSpec is not None:
        cloud = os.getenv("PINECONE_CLOUD", "aws")
        region = os.getenv("PINECONE_REGION", "us-east-1")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
    else:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
        )

print(f"✓ Pinecone index ready: {INDEX_NAME}")
