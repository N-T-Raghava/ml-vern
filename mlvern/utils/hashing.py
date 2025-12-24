import hashlib
import json


def hash_object(obj) -> str:
    serialized = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()
