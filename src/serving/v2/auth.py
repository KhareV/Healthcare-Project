"""Minimal Clerk session-token verification for FastAPI product endpoints.

No official Clerk Python SDK exists, so this verifies tokens the same way
any JWT-based auth works: derive Clerk's public JWKS URL from the
publishable key (a public-key operation -- no secret required), fetch and
cache it, and verify the RS256 signature plus standard claims (exp/nbf/iss)
on each request. This backend never trusts a user id supplied by a request
body or a custom header -- only the `sub` claim of a signature-verified
token, or the fixed local-dev owner when no Clerk key is configured at all.

This module knows nothing about custom records or ownership -- it only
answers "who, if anyone, is this request authenticated as?" Ownership
checks live in api/v2_app.py, next to the endpoints that need them.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Optional

import jwt
from jwt import PyJWKClient

LOCAL_DEV_OWNER_ID = "local-dev-user"


class AuthError(Exception):
    """Raised for any authentication failure (missing/invalid/expired token)."""


def frontend_api_host(publishable_key: str) -> str:
    """Clerk publishable keys are "pk_{test,live}_<base64(host + '$')>"."""

    if not publishable_key.startswith(("pk_test_", "pk_live_")):
        raise AuthError("malformed Clerk publishable key")
    encoded = publishable_key.split("_", 2)[2]
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        decoded = base64.b64decode(padded).decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        raise AuthError("malformed Clerk publishable key") from exc
    return decoded.rstrip("$")


@dataclass
class ClerkAuthenticator:
    """Verifies bearer tokens against one Clerk application's JWKS.

    `publishable_key=None` puts this in local-development mode: every call
    to `verify()` succeeds and returns the fixed LOCAL_DEV_OWNER_ID without
    requiring (or even looking at) a token. This must only ever happen when
    the operator has not configured Clerk for the backend at all -- the
    frontend independently makes its own auth-optional decision the same
    way (see frontend/src/hooks.server.ts) -- so the two can never disagree
    about whether real authentication is active.
    """

    publishable_key: Optional[str]
    _jwk_client: Optional[PyJWKClient] = field(init=False, default=None, repr=False)
    _issuer: Optional[str] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.publishable_key:
            host = frontend_api_host(self.publishable_key)
            self._issuer = f"https://{host}"
            self._jwk_client = PyJWKClient(f"{self._issuer}/.well-known/jwks.json", cache_keys=True, lifespan=3600)

    @property
    def configured(self) -> bool:
        return self._jwk_client is not None

    def verify(self, authorization_header: Optional[str]) -> str:
        """Return the verified Clerk user id, or raise AuthError."""

        if not self.configured:
            return LOCAL_DEV_OWNER_ID
        if not authorization_header or not authorization_header.startswith("Bearer "):
            raise AuthError("missing bearer token")
        token = authorization_header[len("Bearer "):].strip()
        if not token:
            raise AuthError("empty bearer token")
        try:
            signing_key = self._jwk_client.get_signing_key_from_jwt(token)
            claims = jwt.decode(token, signing_key.key, algorithms=["RS256"], issuer=self._issuer, options={"verify_aud": False})
        except Exception as exc:  # noqa: BLE001 - any verification failure is just "not authenticated"
            raise AuthError(f"invalid token: {exc}") from exc
        user_id = claims.get("sub")
        if not user_id or not isinstance(user_id, str):
            raise AuthError("token missing sub claim")
        return user_id
