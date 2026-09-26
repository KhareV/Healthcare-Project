"""Focused tests for serving.v2.auth's Clerk JWT verification.

Network-free: a locally-generated RSA keypair stands in for Clerk's own
signing key, and PyJWKClient's key-fetch is monkeypatched to return it
in-process rather than making a real HTTPS call."""

import sys
import time
from pathlib import Path

import jwt as pyjwt
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

REAL_PUBLISHABLE_KEY = "pk_test_dmVyaWZpZWQta2luZ2Zpc2gtNTQuY2xlcmsuYWNjb3VudHMuZGV2JA"


@pytest.fixture(scope="module")
def rsa_keypair():
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


def _sign(private_key, claims: dict) -> str:
    return pyjwt.encode(claims, private_key, algorithm="RS256")


def _authenticator_with_mocked_key(monkeypatch, public_key):
    from serving.v2.auth import ClerkAuthenticator

    class _FakeSigningKey:
        def __init__(self, key):
            self.key = key

    monkeypatch.setattr(
        "serving.v2.auth.PyJWKClient.get_signing_key_from_jwt",
        lambda self, token: _FakeSigningKey(public_key),
    )
    return ClerkAuthenticator(publishable_key=REAL_PUBLISHABLE_KEY)


def test_frontend_api_host_derived_from_real_publishable_key():
    from serving.v2.auth import frontend_api_host

    assert frontend_api_host(REAL_PUBLISHABLE_KEY) == "verified-kingfish-54.clerk.accounts.dev"


def test_frontend_api_host_rejects_malformed_key():
    from serving.v2.auth import AuthError, frontend_api_host

    with pytest.raises(AuthError):
        frontend_api_host("not-a-clerk-key")


def test_local_dev_mode_returns_fixed_owner_without_a_token():
    from serving.v2.auth import LOCAL_DEV_OWNER_ID, ClerkAuthenticator

    authenticator = ClerkAuthenticator(publishable_key=None)
    assert authenticator.configured is False
    assert authenticator.verify(None) == LOCAL_DEV_OWNER_ID
    assert authenticator.verify("Bearer garbage") == LOCAL_DEV_OWNER_ID


def test_configured_mode_verifies_a_valid_signed_token(monkeypatch, rsa_keypair):
    private_key, public_key = rsa_keypair
    authenticator = _authenticator_with_mocked_key(monkeypatch, public_key)
    assert authenticator.configured is True

    token = _sign(private_key, {
        "sub": "user_abc123", "iss": "https://verified-kingfish-54.clerk.accounts.dev",
        "iat": int(time.time()), "exp": int(time.time()) + 60,
    })
    assert authenticator.verify(f"Bearer {token}") == "user_abc123"


def test_configured_mode_rejects_missing_token(monkeypatch, rsa_keypair):
    from serving.v2.auth import AuthError

    _, public_key = rsa_keypair
    authenticator = _authenticator_with_mocked_key(monkeypatch, public_key)
    with pytest.raises(AuthError, match="missing bearer token"):
        authenticator.verify(None)
    with pytest.raises(AuthError, match="missing bearer token"):
        authenticator.verify("NotBearer abc")


def test_configured_mode_rejects_expired_token(monkeypatch, rsa_keypair):
    from serving.v2.auth import AuthError

    private_key, public_key = rsa_keypair
    authenticator = _authenticator_with_mocked_key(monkeypatch, public_key)
    token = _sign(private_key, {
        "sub": "user_abc123", "iss": "https://verified-kingfish-54.clerk.accounts.dev",
        "iat": int(time.time()) - 120, "exp": int(time.time()) - 60,
    })
    with pytest.raises(AuthError, match="invalid token"):
        authenticator.verify(f"Bearer {token}")


def test_configured_mode_rejects_wrong_issuer(monkeypatch, rsa_keypair):
    from serving.v2.auth import AuthError

    private_key, public_key = rsa_keypair
    authenticator = _authenticator_with_mocked_key(monkeypatch, public_key)
    token = _sign(private_key, {
        "sub": "user_abc123", "iss": "https://not-clerk.example.com",
        "iat": int(time.time()), "exp": int(time.time()) + 60,
    })
    with pytest.raises(AuthError, match="invalid token"):
        authenticator.verify(f"Bearer {token}")


def test_configured_mode_rejects_token_signed_by_a_different_key(monkeypatch, rsa_keypair):
    from cryptography.hazmat.primitives.asymmetric import rsa

    from serving.v2.auth import AuthError

    _, public_key = rsa_keypair
    authenticator = _authenticator_with_mocked_key(monkeypatch, public_key)

    attacker_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    forged = _sign(attacker_key, {
        "sub": "user_attacker", "iss": "https://verified-kingfish-54.clerk.accounts.dev",
        "iat": int(time.time()), "exp": int(time.time()) + 60,
    })
    with pytest.raises(AuthError, match="invalid token"):
        authenticator.verify(f"Bearer {forged}")


def test_configured_mode_rejects_token_missing_sub_claim(monkeypatch, rsa_keypair):
    from serving.v2.auth import AuthError

    private_key, public_key = rsa_keypair
    authenticator = _authenticator_with_mocked_key(monkeypatch, public_key)
    token = _sign(private_key, {
        "iss": "https://verified-kingfish-54.clerk.accounts.dev",
        "iat": int(time.time()), "exp": int(time.time()) + 60,
    })
    with pytest.raises(AuthError, match="missing sub claim"):
        authenticator.verify(f"Bearer {token}")
