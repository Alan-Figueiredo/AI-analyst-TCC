import cv2
import numpy as np
from ultralytics import YOLO

# Funções e Constantes
INDICE_OMBRO_ESQ = 5; INDICE_COTOVELO_ESQ = 7; INDICE_PULSO_ESQ = 9
INDICE_OMBRO_DIR = 6; INDICE_COTOVELO_DIR = 8; INDICE_PULSO_DIR = 10

def calcular_angulo(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# CONFIGURAÇÃO
model = YOLO('yolov8m-pose.pt')
NOME_DO_VIDEO = "videoteste.mp4"
video = cv2.VideoCapture(NOME_DO_VIDEO)

# Parâmetros da lógica de contagem
LIMIAR_CONFIANCA_PONTO = 0.5 
ANGULO_ESTICADO = 115 
ANGULO_RECOLHIDO = 90

# Variáveis de estado e contagem
contador_esq = 0
estado_esq = "RECOLHIDO"
contador_dir = 0
estado_dir = "RECOLHIDO"

if not video.isOpened():
    print(f"Erro ao abrir o arquivo de vídeo: {NOME_DO_VIDEO}")
    exit()

# LOOP PRINCIPAL
while True:
    success, img = video.read()
    if not success:
        print("Fim do vídeo.")
        break

    results = model(img, stream=True, verbose=False, conf=0.6)

    for r in results:
        annotated_frame = r.plot()

        if r.keypoints and len(r.keypoints.xy) > 0:
            keypoints = r.keypoints.xy[0].cpu().numpy()
            confiancas = r.keypoints.conf[0].cpu().numpy()

            try:
                # Lógica do Braço Esquerdo
                conf_ombro_esq = confiancas[INDICE_OMBRO_ESQ]
                conf_cotovelo_esq = confiancas[INDICE_COTOVELO_ESQ]
                conf_pulso_esq = confiancas[INDICE_PULSO_ESQ]

                if conf_ombro_esq > LIMIAR_CONFIANCA_PONTO and conf_cotovelo_esq > LIMIAR_CONFIANCA_PONTO and conf_pulso_esq > LIMIAR_CONFIANCA_PONTO:
                    ombro_esq = keypoints[INDICE_OMBRO_ESQ]
                    cotovelo_esq = keypoints[INDICE_COTOVELO_ESQ]
                    pulso_esq = keypoints[INDICE_PULSO_ESQ]
                    angulo_esq = calcular_angulo(ombro_esq, cotovelo_esq, pulso_esq)
                    
                    # LINHA READICIONADA: Desenha o ângulo do braço esquerdo
                    cv2.putText(annotated_frame, f"{int(angulo_esq)}", tuple(cotovelo_esq.astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    
                    if angulo_esq > ANGULO_ESTICADO and estado_esq == 'RECOLHIDO':
                        estado_esq = 'ESTICADO'
                    if angulo_esq < ANGULO_RECOLHIDO and estado_esq == 'ESTICADO':
                        estado_esq = 'RECOLHIDO'
                        contador_esq += 1

                # Lógica do Braço Direito
                conf_ombro_dir = confiancas[INDICE_OMBRO_DIR]
                conf_cotovelo_dir = confiancas[INDICE_COTOVELO_DIR]
                conf_pulso_dir = confiancas[INDICE_PULSO_DIR]

                if conf_ombro_dir > LIMIAR_CONFIANCA_PONTO and conf_cotovelo_dir > LIMIAR_CONFIANCA_PONTO and conf_pulso_dir > LIMIAR_CONFIANCA_PONTO:
                    ombro_dir = keypoints[INDICE_OMBRO_DIR]
                    cotovelo_dir = keypoints[INDICE_COTOVELO_DIR]
                    pulso_dir = keypoints[INDICE_PULSO_DIR]
                    angulo_dir = calcular_angulo(ombro_dir, cotovelo_dir, pulso_dir)
                    
                    # LINHA READICIONADA: Desenha o ângulo do braço direito
                    cv2.putText(annotated_frame, f"{int(angulo_dir)}", tuple(cotovelo_dir.astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    
                    if angulo_dir > ANGULO_ESTICADO and estado_dir == 'RECOLHIDO':
                        estado_dir = 'ESTICADO'
                    if angulo_dir < ANGULO_RECOLHIDO and estado_dir == 'ESTICADO':
                        estado_dir = 'RECOLHIDO'
                        contador_dir += 1
            except Exception as e:
                pass
        
        # PLACAR E VISUALIZAÇÃO
        cv2.rectangle(annotated_frame, (0, 0), (600, 80), (20, 20, 20), -1)
        cv2.putText(annotated_frame, "ESQUERDO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, str(contador_esq), (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(annotated_frame, "DIREITO", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, str(contador_dir), (450, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        cv2.imshow("Contador de Socos Final - YOLOv8", annotated_frame)
        
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

video.release()
cv2.destroyAllWindows()