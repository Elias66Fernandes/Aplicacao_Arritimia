import os
import sys
from flask import Flask
from src.routes.arrhythmia_simple import arrhythmia_bp

# Corrige o path base do projeto
BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, 'src')

# Criar o app com caminhos corretos
app = Flask(
    __name__,
    static_folder=os.path.join(SRC_DIR, 'static'),
    template_folder=os.path.join(SRC_DIR, 'templates')
)

# Registrar blueprint
app.register_blueprint(arrhythmia_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
