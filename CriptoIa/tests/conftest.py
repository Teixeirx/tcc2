# conftest.py — Configurações globais do pytest para o CriptoIA
#
# Este arquivo é carregado automaticamente pelo pytest antes de qualquer teste.
# Define fixtures compartilhadas entre todos os módulos de teste.

import sys
import os

# Adiciona a raiz do projeto ao path para importações corretas
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Credenciais e segredo determinísticos para os testes da interface web.
# Nunca use estes valores em produção.
os.environ["WEB_ADMIN_USER"] = "admin"
os.environ["WEB_ADMIN_PASS"] = "admin"
os.environ["FLASK_SECRET_KEY"] = "segredo-de-teste-nao-usar-em-producao"
os.environ["CRIPTOIA_ENV"] = "test"
