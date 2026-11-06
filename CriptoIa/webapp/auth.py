"""
webapp/auth.py — Módulo de autenticação segura e controle de sessão.

Utiliza werkzeug.security para hashing seguro (pbkdf2:sha256) e
decoradores para proteção de rotas restritas.

Credenciais:
    - WEB_ADMIN_USER (padrão: admin)
    - WEB_ADMIN_PASS (obrigatória em produção; em desenvolvimento,
      usa uma senha padrão apenas para facilitar a execução local)

Em produção (CRIPTOIA_ENV=production), a ausência de WEB_ADMIN_PASS
interrompe a inicialização em vez de usar credenciais conhecidas
"""

import os
from functools import wraps
from flask import session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv()

DEFAULT_ADMIN_USER = os.getenv("WEB_ADMIN_USER", "admin")
_SENHA_ENV = (os.getenv("WEB_ADMIN_PASS") or "").strip()

_SENHA_PLACEHOLDER = _SENHA_ENV.lower() in {
    "troque-por-uma-senha-forte",
    "sua_senha",
    "changeme",
    "password",
} or (0 < len(_SENHA_ENV) < 8)

if not _SENHA_ENV or _SENHA_PLACEHOLDER:
    if os.getenv("CRIPTOIA_ENV", "").lower() == "production":
        raise RuntimeError(
            "WEB_ADMIN_PASS ausente, placeholder ou com menos de 8 caracteres. "
            "Defina uma senha forte antes de iniciar a aplicação em produção."
        )
    if _SENHA_ENV:
        print(
            "[AVISO] WEB_ADMIN_PASS é placeholder/curta; usando senha padrão de "
            "desenvolvimento. Defina uma senha forte para uso real."
        )
    else:
        print(
            "[AVISO] WEB_ADMIN_PASS não configurada; usando senha padrão de "
            "desenvolvimento. Defina WEB_ADMIN_PASS e FLASK_SECRET_KEY para uso real."
        )
    _SENHA_ENV = "admin"
    USANDO_SENHA_PADRAO = True
else:
    USANDO_SENHA_PADRAO = False

DEFAULT_ADMIN_PASS = _SENHA_ENV

# Armazena hash da senha (nunca em texto plano)
USUARIOS_DB = {
    DEFAULT_ADMIN_USER: generate_password_hash(DEFAULT_ADMIN_PASS)
}


def verificar_credenciais(usuario: str, senha_plana: str) -> bool:
    """Verifica se o usuário e senha correspondem ao hash armazenado."""
    if usuario not in USUARIOS_DB:
        return False
    return check_password_hash(USUARIOS_DB[usuario], senha_plana)


def login_required(f):
    """Decorador para proteger rotas que exigem usuário autenticado."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("usuario_autenticado"):
            flash("Por favor, faça login para acessar esta página.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function
