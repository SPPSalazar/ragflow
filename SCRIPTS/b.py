import os, json, boto3
from dotenv import load_dotenv
from botocore.config import Config
from botocore.exceptions import ClientError

load_dotenv()

REGION = os.getenv("AWS_REGION")
AK     = os.getenv("AWS_ACCESS_KEY_ID")
SK     = os.getenv("AWS_SECRET_ACCESS_KEY")
MODEL  = os.getenv("AWS_BEDROCK_EMBEDDINGS_ID", "amazon.titan-embed-text-v2:0")  # must be literal :0

print(f"[DEBUG] region={REGION} model={MODEL}")

sess = boto3.Session(aws_access_key_id=AK, aws_secret_access_key=SK, aws_session_token=SK, region_name=REGION)
sts  = sess.client("sts", region_name=REGION)
print("STS:", sts.get_caller_identity())

br = sess.client("bedrock-runtime", region_name=REGION, config=Config(retries={"max_attempts": 3, "mode": "standard"}))

payload = {
    "inputText": "hola mundo",
    "dimensions": int(os.getenv("EMBED_DIMS", "1024")),
    "normalize": True,
    "embeddingTypes": ["float"]
}

try:
    resp = br.invoke_model(
        modelId=MODEL,
        accept="application/json",
        contentType="application/json",
        body=json.dumps(payload),
    )
    data = json.loads(resp["body"].read().decode("utf-8"))
    print("[OK] Titan response keys:", list(data.keys()))
    if "embedding" in data:
        print("[OK] Embedding length:", len(data["embedding"]))
    else:
        print("[WARN] No 'embedding' key. Full:", data)
except ClientError as e:
    # Print the full AWS error for diagnosis
    print("[AWS ClientError]", e.response.get("Error", {}))
    # Helpful specifics:
    print("[RequestId]", e.response.get("ResponseMetadata", {}).get("RequestId"))
    # If service returned a JSON body with details:
    try:
        print("[RawBody]", e.response["body"].read().decode("utf-8"))
    except Exception:
        pass