from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request, Security

from cekdulu.api.container import Container
from cekdulu.api.security import api_key_header, client_identity, verify_api_key


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]


def require_api_key(
    request: Request,
    container: ContainerDep,
    api_key: Annotated[str | None, Security(api_key_header)] = None,
) -> str:
    settings = container.settings
    principal = verify_api_key(api_key, settings.api_key_list, settings.env)
    request.app.state.rate_limiter.check(client_identity(request, principal))
    return principal


AuthDep = Annotated[str, Depends(require_api_key)]
