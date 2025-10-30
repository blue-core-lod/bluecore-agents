import httpx


def get_token(bc_base_url: str, user: str, password: str):
    keycloak_token = httpx.post(
        f"{bc_base_url}/keycloak/realms/bluecore/protocol/openid-connect/token",
        data={
            "client_id": "bluecore_api",
            "username": user,
            "password": password,
            "grant_type": "password",
        },
    )

    keycloak_token.raise_for_status()
    return keycloak_token.json().get("access_token")
