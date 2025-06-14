import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

# Inicializando o MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuração do MediaPipe
hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7)

# Lista para armazenar os dados
data = []

# Inicializando a câmera
cap = cv2.VideoCapture(0)

# Perguntar qual letra será capturada
label = input("Digite a letra que será capturada: ").upper()

print("\n>>> Captura iniciada <<<")
print("Pressione 'C' para capturar uma amostra.")
print("Pressione 'ESC' para finalizar e salvar.\n")

# ---- Loop principal ----
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a câmera.")
        break

    # Flip da imagem e conversão para RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processamento da imagem
    result = hands.process(rgb_frame)

    # Desenha landmarks se houver detecção
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extrair landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # ---- Pré-processamento ----
            points = np.array(landmarks).reshape((21, 3))

            # Centralizar em relação ao pulso
            base_point = points[0]
            points -= base_point

            # Normalizar pela maior distância entre pontos
            max_value = np.max(np.linalg.norm(points, axis=1))
            if max_value != 0:
                points /= max_value

            # Achatar de volta para lista
            processed_landmarks = points.flatten().tolist()

            # Mostrar informação na tela
            cv2.putText(frame, f'Letra: {label}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    else:
        processed_landmarks = None  # Sem mão detectada

    # Mostrar a janela
    cv2.imshow('Captura de Dados - Libras', frame)

    # ---- Controle de teclado ----
    key = cv2.waitKey(10) & 0xFF

    if key == ord('c'):
        if processed_landmarks is not None:
            data.append(processed_landmarks + [label])
            print(f'[+] Amostra capturada para {label}. Total de amostras: {len(data)}')
        else:
            print('[!] Nenhuma mão detectada. Tente novamente.')

    elif key == 27:  # ESC
        print("\n>>> Finalizando e salvando o dataset...")
        break

# ---- Salvamento dos dados ----
if data:
    columns = []
    for i in range(21):
        columns += [f'x{i}', f'y{i}', f'z{i}']
    columns.append('label')

    df = pd.DataFrame(data, columns=columns)

    nome_arquivo = f'dataset_{label}.csv'
    df.to_csv(nome_arquivo, index=False)

    print(f'Dataset salvo como {nome_arquivo}')
    print(f'Total de amostras: {len(df)}')
else:
    print('Nenhuma amostra foi capturada.')

# ---- Encerrando ----
cap.release()
cv2.destroyAllWindows()
