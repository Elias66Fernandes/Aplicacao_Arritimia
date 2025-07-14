from flask import Blueprint, render_template, request, jsonify
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pasta src
model_path = os.path.join(BASE_DIR, 'models', 'CatBoost_best_model.joblib')
model = joblib.load(model_path)


scaler = StandardScaler()


arrhythmia_bp = Blueprint('arrhythmia', __name__)

# Mapeamento das classes
CLASS_MAPPING = {
    0: "Normal",
    1: "Ischemic changes (Coronary Artery Disease)",
    2: "Old Anterior Myocardial Infarction",
    3: "Old Inferior Myocardial Infarction",
    4: "Sinus tachycardy",
    5: "Sinus bradycardy",
    6: "Right bundle branch block",
    7: "Outras Arritmias Raras",  # Agrupamento de [7, 8, 9, 14, 15]
    8: "Others"
}


# Lista das 50 features esperadas pelo modelo
EXPECTED_FEATURES = [
    'QRS_duration', 'QT_interval', 'T_interval', 'heart_rate', 'DI_R_width', 
    'DII_Num_Intrinsic_Deflections', 'AVF_Q_width', 'V1_Q_width', 'V1_R_prime_width', 
    'V1_Num_Intrinsic_Deflections', 'V2_Q_width', 'V2_S_width', 'V2_R_prime_width', 
    'V3_Q_width', 'V3_R_width', 'V3_S_width', 'V3_Num_Intrinsic_Deflections', 
    'V3_Diphasic_T_Derivation', 'V4_Q_width', 'V4_R_width', 'V5_R_width', 
    'V5_Num_Intrinsic_Deflections', 'V6_R_width', 'V6_Num_Intrinsic_Deflections', 
    'DI_JJ_amp', 'DI_S_amp', 'DI_T_amp', 'DII_Q_amp', 'DII_T_amp', 'DIII_Q_amp', 
    'AVR_JJ_amp', 'AVR_R_amp', 'AVR_T_amp', 'AVF_Q_amp', 'V1_R_prime_amp', 
    'V1_QRSA', 'V2_JJ_amp', 'V2_Q_amp', 'V2_QRSA', 'V3_JJ_amp', 'V3_Q_amp', 
    'V3_QRSA', 'V3_QRSTA', 'V4_Q_amp', 'V5_T_amp', 'V5_QRSTA', 'V6_JJ_amp', 
    'V6_T_amp', 'V6_QRSA', 'V6_QRSTA'
]

@arrhythmia_bp.route('/')
def home():
    """Página inicial"""
    return render_template('home.html')

@arrhythmia_bp.route('/evaluate')
def evaluate():
    """Página de avaliação"""
    return render_template('evaluate.html')

@arrhythmia_bp.route('/api/predict', methods=['POST'])
@arrhythmia_bp.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Nenhum dado fornecido'}), 400

        # Verifica se todas as features estão presentes
        missing = [f for f in EXPECTED_FEATURES if f not in data]
        if missing:
            return jsonify({'error': f'Features faltando: {missing}'}), 400

        # Converte os dados para float e ordena pela ordem esperada
        try:
            input_vector = np.array([float(data[feat]) for feat in EXPECTED_FEATURES]).reshape(1, -1)
        except Exception as e:
            return jsonify({'error': f'Erro ao converter dados: {str(e)}'}), 400

        # Normaliza os dados
        input_scaled = scaler.transform(input_vector)

        # Faz a predição
        prediction = model.predict(input_scaled)[0]
        prediction = int(prediction)

        # Confiança (se houver predict_proba no pipeline)
        if hasattr(model, 'predict_proba'):
            confidence = float(np.max(model.predict_proba(input_scaled)))
        else:
            confidence = None

        # Mapeamento da classe final
        class_name = CLASS_MAPPING.get(prediction, f"Classe {prediction}")

        return jsonify({
            'prediction': prediction,
            'class_name': class_name,
            'confidence': confidence,
            'note': 'Predição realizada com modelo real (CatBoost).'
        })

    except Exception as e:
        return jsonify({'error': f'Erro interno do servidor: {str(e)}'}), 500


