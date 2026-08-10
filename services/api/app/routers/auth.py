"""Registration, login and the current-user endpoint.

Two details that are security-relevant rather than cosmetic:

* **Registration and login return the same generic failure text for a duplicate email
  and a bad password respectively.** Login never distinguishes "no such user" from
  "wrong password", because doing so turns the endpoint into an account-enumeration
  oracle.
* **A failed login still runs a bcrypt verification** against a dummy hash. Returning
  early for an unknown email makes the response measurably faster than for a known one,
  which is the same oracle by a side channel.
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from cutoutml.core.logging import get_logger
from cutoutml.db.models import User
from services.api.app.deps import CurrentUser, RateLimited, SessionDep, SettingsDep
from services.api.app.errors import ApiError
from services.api.app.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from services.api.app.security import create_access_token, hash_password, verify_password

log = get_logger(__name__)

router = APIRouter(prefix="/v1/auth", tags=["auth"], dependencies=[RateLimited])

_DUMMY_HASH = hash_password("timing-equalisation-placeholder")
"""Verified against when the email is unknown, so both paths cost one bcrypt round."""


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account",
)
def register(
    payload: RegisterRequest, session: SessionDep, settings: SettingsDep
) -> TokenResponse:
    user = User(
        email=payload.email.lower(),
        password_hash=hash_password(payload.password),
        display_name=payload.display_name,
    )
    session.add(user)
    try:
        session.commit()
    except IntegrityError:
        # The unique index is the authority, not a prior SELECT: checking first and
        # inserting second is a race that two concurrent signups will lose.
        session.rollback()
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "email_taken",
            "an account with that email address already exists",
        ) from None

    log.info("user_registered", user_id=str(user.id))
    token = create_access_token(user.id, settings=settings)
    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_access_ttl_seconds,
        user_id=user.id,
    )


@router.post("/login", response_model=TokenResponse, summary="Exchange credentials for a token")
def login(payload: LoginRequest, session: SessionDep, settings: SettingsDep) -> TokenResponse:
    stmt = select(User).where(User.email == payload.email.lower())
    user = session.execute(stmt).scalar_one_or_none()

    stored = user.password_hash if user else _DUMMY_HASH
    valid = verify_password(payload.password, stored)

    if user is None or not valid or not user.is_active:
        log.info("login_failed", email_domain=payload.email.rsplit("@", 1)[-1])
        raise ApiError(
            status.HTTP_401_UNAUTHORIZED,
            "invalid_credentials",
            "email or password is incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user.updated_at = dt.datetime.now(dt.UTC)
    session.commit()
    log.info("login_ok", user_id=str(user.id))
    return TokenResponse(
        access_token=create_access_token(user.id, settings=settings),
        expires_in=settings.jwt_access_ttl_seconds,
        user_id=user.id,
    )


@router.get("/me", response_model=UserResponse, summary="The authenticated principal")
def me(user: CurrentUser) -> UserResponse:
    return UserResponse.model_validate(user)
