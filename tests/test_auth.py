"""Tests for authentication and RBAC permissions."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_login_success(client: TestClient):
    resp = client.post(
        "/api/v1/auth/token",
        data={"username": "test_admin", "password": "admin_pass123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["role"] == "admin"
    assert data["username"] == "test_admin"


def test_login_invalid_password(client: TestClient):
    resp = client.post(
        "/api/v1/auth/token",
        data={"username": "test_admin", "password": "wrong_password"},
    )
    assert resp.status_code == 401


def test_auth_me(client: TestClient, admin_headers: dict[str, str]):
    resp = client.get("/api/v1/auth/me", headers=admin_headers)
    assert resp.status_code == 200
    assert resp.json()["username"] == "test_admin"
    assert resp.json()["role"] == "admin"


def test_admin_route_forbidden_for_engineer(client: TestClient, engineer_headers: dict[str, str]):
    resp = client.get("/api/v1/auth/users", headers=engineer_headers)
    assert resp.status_code == 403


def test_admin_route_allowed_for_admin(client: TestClient, admin_headers: dict[str, str]):
    resp = client.get("/api/v1/auth/users", headers=admin_headers)
    assert resp.status_code == 200
    users = resp.json()
    assert len(users) >= 2
