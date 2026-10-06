import hashlib
import hmac
import secrets
from app.config import settings

def mint_agent_token() -> str:
    return "sops_" + secrets.token_hex(32)

def digest_token(token: str) -> str:
    return hmac.new(settings.credential_hmac_key.encode(), token.encode(), hashlib.sha256).hexdigest()

def token_matches(token: str, digest: str) -> bool:
    return hmac.compare_digest(digest_token(token), digest)
