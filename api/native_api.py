"""Compatibility imports for the retired local admission module.

New code should import :mod:`foundry.admission.request_admission`. This module
remains so existing local wrappers can migrate without changing their request
contract. It does not provide an HTTP API.
"""

from foundry.admission.request_admission import (  # noqa: F401
    ApiConfig,
    ROOT,
    RequestAdmissionService,
    RequestFailure,
    _MODEL_KEYS,
    _prepare_request,
    build_parser,
    main,
)

# The old class name is retained for one compatibility release.
NativeApiService = RequestAdmissionService

__all__ = [
    "ApiConfig",
    "NativeApiService",
    "RequestFailure",
    "build_parser",
    "main",
]
