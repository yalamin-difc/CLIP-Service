from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet

from fastapi import HTTPException, Request


JWT_ALGORITHM = "HS256"
JWT_ISSUER = os.environ.get("INTERNAL_JWT_ISSUER", "dubai-festival-backend")
JWT_AUDIENCE = os.environ.get("INTERNAL_JWT_AUDIENCE", "clip-service")
INTERNAL_JWT_SECRET = os.environ.get("INTERNAL_JWT_SECRET", "")


@dataclass(frozen=True)
class ServiceIdentity:
    service_name: str
    tenant_id: str
    site_id: str
    permitted_site_ids: FrozenSet[str]
    actions: FrozenSet[str]
    request_id: str
    dataset_version: str
    demo_data: bool


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _auth_error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def decode_internal_jwt(token: str) -> Dict[str, Any]:
    if not INTERNAL_JWT_SECRET:
        raise _auth_error(503, "internal_auth_misconfigured", "Internal service authentication is not configured.")
    try:
        encoded_header, encoded_payload, encoded_signature = token.split(".")
        header = json.loads(_b64url_decode(encoded_header))
        claims = json.loads(_b64url_decode(encoded_payload))
    except Exception as exc:
        raise _auth_error(401, "invalid_identity", "Internal identity token is malformed.") from exc
    if header.get("alg") != JWT_ALGORITHM or header.get("typ") != "JWT":
        raise _auth_error(401, "invalid_identity", "Internal identity token algorithm is not allowed.")
    signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
    expected = hmac.new(INTERNAL_JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    try:
        supplied = _b64url_decode(encoded_signature)
    except Exception as exc:
        raise _auth_error(401, "invalid_identity", "Internal identity token signature is malformed.") from exc
    if not hmac.compare_digest(expected, supplied):
        raise _auth_error(401, "invalid_identity", "Internal identity token signature is invalid.")
    now = int(time.time())
    if claims.get("iss") != JWT_ISSUER or claims.get("aud") != JWT_AUDIENCE:
        raise _auth_error(401, "invalid_identity", "Internal identity token issuer or audience is invalid.")
    if not isinstance(claims.get("exp"), (int, float)) or int(claims["exp"]) <= now:
        raise _auth_error(401, "identity_expired", "Internal identity token has expired.")
    return claims


def require_identity(request: Request, action: str) -> ServiceIdentity:
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        raise _auth_error(401, "missing_identity", "A signed internal service identity is required.")
    claims = decode_internal_jwt(authorization[7:].strip())
    required_string_claims = ("service", "tenantId", "siteId", "requestId")
    if any(not isinstance(claims.get(name), str) or not claims[name].strip() for name in required_string_claims):
        raise _auth_error(401, "invalid_identity", "Internal identity token is missing required context.")
    actions = claims.get("actions")
    site_ids = claims.get("siteIds")
    if not isinstance(actions, list) or not all(isinstance(value, str) for value in actions):
        raise _auth_error(401, "invalid_identity", "Internal identity actions are invalid.")
    if not isinstance(site_ids, list) or not all(isinstance(value, str) for value in site_ids):
        raise _auth_error(401, "invalid_identity", "Internal identity site permissions are invalid.")
    permitted_sites = frozenset(value.strip() for value in site_ids if value.strip())
    site_id = claims["siteId"].strip()
    if site_id not in permitted_sites:
        raise _auth_error(403, "site_not_permitted", "The selected site is not permitted for this identity.")
    permitted_actions = frozenset(value.strip() for value in actions if value.strip())
    if action not in permitted_actions:
        raise _auth_error(403, "action_not_permitted", f"Identity is not permitted to perform '{action}'.")
    request_id = claims["requestId"].strip()
    header_request_id = request.headers.get("X-Request-Id", "").strip()
    if header_request_id and header_request_id != request_id:
        raise _auth_error(401, "request_id_mismatch", "Identity request ID does not match the request.")
    return ServiceIdentity(
        service_name=claims["service"].strip(),
        tenant_id=claims["tenantId"].strip(),
        site_id=site_id,
        permitted_site_ids=permitted_sites,
        actions=permitted_actions,
        request_id=request_id,
        dataset_version=str(claims.get("datasetVersion") or "festival-2026"),
        demo_data=bool(claims.get("demoData", True)),
    )


def sign_internal_token(claims: Dict[str, Any], secret: str | None = None) -> str:
    """Create an HS256 token for backend tooling and tests."""
    key = secret if secret is not None else INTERNAL_JWT_SECRET
    if not key:
        raise RuntimeError("INTERNAL_JWT_SECRET is required")
    header = _b64url_encode(json.dumps({"alg": JWT_ALGORITHM, "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64url_encode(json.dumps(claims, separators=(",", ":")).encode())
    signature = hmac.new(key.encode("utf-8"), f"{header}.{payload}".encode("ascii"), hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url_encode(signature)}"
