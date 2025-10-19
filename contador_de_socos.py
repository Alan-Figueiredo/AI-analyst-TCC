import cv2
import numpy as np
import time
from ultralytics import YOLO

# === MODELOS ===
model_pose = YOLO("yolov8m-pose.pt")
model_obj = YOLO("best.pt") # Usando seu modelo treinado!

# === CONFIGURAÇÕES ===
NOME_DO_VIDEO = "videoteste.mp4"
video = cv2.VideoCapture(NOME_DO_VIDEO)
fps = video.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

# --- PARÂMETRO DE OTIMIZAÇÃO ---
# Processa a IA a cada N quadros. Um valor entre 2 e 4 é ideal.
FREQUENCIA_IA = 3
# --------------------------------

# === ÍNDICES DE KEYPOINTS (YOLOv8 Pose) ===
INDICE_QUEIXO = 0
LIMIAR_CONFIANCA = 0.55

# === PARÂMETROS DE DETECÇÃO ===
DIST_QUEIXO_INICIO = 90
DIST_QUEIXO_RESET = 75
IMPACT_PADDING = 15
DISTANCIA_PROJECAO_MAO = 40

# === VARIÁVEIS DE ESTADO E TRACKING ===
dados_socos = []
estado_braços = {
    "direito": {"movendo": False, "inicio": 0, "p_inicial": None, "impactado": False},
    "esquerdo": {"movendo": False, "inicio": 0, "p_inicial": None, "impactado": False}
}
ultima_posicao_saco = None
frames_sem_deteccao = 0
LIMITE_FRAMES_PERDIDOS = 10
# --- Novas variáveis para guardar as últimas posições conhecidas ---
keypoints_recentes = None
saco_box_recente = None
# ------------------------------------------------------------------

def detectar_saco(frame, modelo):
    result_obj = modelo(frame, verbose=False, conf=0.50)
    for det in result_obj[0].boxes:
        nome = modelo.names[int(det.cls)]
        if "saco_pancada" in nome.lower():
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            return (x1, y1, x2, y2)
    return None

# === LOOP PRINCIPAL ===
print("▶️ Iniciando análise de vídeo...")
frame_id = 0
while True:
    try:
        ret, frame = video.read()
        if not ret:
            print("\nFim do vídeo.")
            break
        
        frame_id += 1
        
        # --- LÓGICA DE OTIMIZAÇÃO ---
        # Só roda a detecção da IA nos quadros múltiplos da frequência definida
        if frame_id % FREQUENCIA_IA == 0:
            result_pose = model_pose(frame, verbose=False)
            saco_box_detectado = detectar_saco(frame, model_obj)

            # Guarda os resultados mais recentes
            if result_pose[0].keypoints and len(result_pose[0].keypoints.xy) > 0:
                keypoints_recentes = result_pose[0].keypoints
            
            if saco_box_detectado is not None:
                saco_box_recente = saco_box_detectado
        # ----------------------------

        # A lógica de tracking (memória) continua rodando em todos os frames
        if saco_box_recente is not None:
             if saco_box_detectado is not None:
                ultima_posicao_saco = saco_box_detectado
                frames_sem_deteccao = 0
             else:
                frames_sem_deteccao += 1
        
        if ultima_posicao_saco is not None and frames_sem_deteccao < LIMITE_FRAMES_PERDIDOS:
            saco_box = ultima_posicao_saco
        else:
            saco_box = None
            ultima_posicao_saco = None


        if saco_box:
            cor_caixa = (0, 255, 0) if frames_sem_deteccao == 0 else (0, 255, 255)
            cv2.rectangle(frame, saco_box[:2], saco_box[2:], cor_caixa, 2)
            cv2.putText(frame, "ALVO", (saco_box[0], saco_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_caixa, 2)

        # Se não temos nenhum keypoint recente, pulamos o resto
        if keypoints_recentes is None:
            cv2.imshow("Análise de Socos", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            continue

        keypoints = keypoints_recentes.xy[0].cpu().numpy()
        confs = keypoints_recentes.conf[0].cpu().numpy()

        # --- O RESTO DO CÓDIGO (PROCESSAMENTO E DESENHO) RODA EM TODOS OS FRAMES ---
        # Isso garante que a parte visual e a lógica de impacto sejam fluidas
        for lado, (indice_pulso, indice_cotovelo, indice_ombro, cor) in [
            ("direito", (10, 8, 6, (0, 0, 255))),
            ("esquerdo", (9, 7, 5, (255, 100, 0)))
        ]:
            if max(indice_pulso, indice_cotovelo, indice_ombro) >= len(keypoints):
                continue
            
            # ... (todo o resto da sua lógica de detecção de soco, projeção da mão, etc.)
            ombro_pos = keypoints[indice_ombro]
            pulso_conf = confs[indice_pulso]
            cotovelo_conf = confs[indice_cotovelo]

            if pulso_conf < LIMIAR_CONFIANCA or cotovelo_conf < LIMIAR_CONFIANCA:
                cv2.putText(frame, f"{lado.upper()}: N/D", (int(ombro_pos[0]-80), int(ombro_pos[1]-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 2)
                continue

            pulso_pt = keypoints[indice_pulso]
            cotovelo_pt = keypoints[indice_cotovelo]

            vetor_antebraco = pulso_pt - cotovelo_pt
            norma_vetor = np.linalg.norm(vetor_antebraco)
            if norma_vetor > 0:
                vetor_unitario = vetor_antebraco / norma_vetor
                ponto_impacto = tuple((pulso_pt + vetor_unitario * DISTANCIA_PROJECAO_MAO).astype(int))
            else:
                ponto_impacto = tuple(pulso_pt.astype(int))

            queixo_pt = tuple(keypoints[INDICE_QUEIXO].astype(int))
            dist_queixo_impacto = np.linalg.norm(np.array(ponto_impacto) - np.array(queixo_pt))

            cv2.circle(frame, ponto_impacto, 9, cor, -1)
            cv2.line(frame, tuple(pulso_pt.astype(int)), ponto_impacto, cor, 2)

            estado = estado_braços[lado]
            status_text = "Guarda"

            impacto_detectado = False
            if saco_box:
                x1, y1, x2, y2 = saco_box
                if (x1 - IMPACT_PADDING) < ponto_impacto[0] < (x2 + IMPACT_PADDING) and \
                   (y1 - IMPACT_PADDING) < ponto_impacto[1] < (y2 + IMPACT_PADDING):
                    impacto_detectado = True

            if not estado["movendo"] and dist_queixo_impacto > DIST_QUEIXO_INICIO:
                estado["movendo"] = True
                estado["inicio"] = time.time()
                estado["p_inicial"] = ponto_impacto
                estado["impactado"] = False

            if estado["movendo"] and not estado["impactado"] and impacto_detectado:
                tempo = time.time() - estado["inicio"]
                dist_total = np.linalg.norm(np.array(ponto_impacto) - np.array(estado["p_inicial"]))
                if tempo > 0:
                    vel = dist_total / tempo
                    dados_socos.append({"braço": lado, "tempo": tempo, "distancia": dist_total, "velocidade": vel})
                    print(f"💥 SOCO {lado.upper()} | Tempo: {tempo:.2f}s | Velocidade: {vel:.1f}px/s")
                estado["impactado"] = True

            if estado["movendo"] and dist_queixo_impacto < DIST_QUEIXO_RESET:
                estado["movendo"] = False

            if estado["movendo"]:
                status_text = "Socando"
                if estado["impactado"]:
                    status_text = "IMPACTO!"

            text_color = (0, 255, 0) if status_text == "IMPACTO!" else (255, 255, 255)
            cv2.putText(frame, f"{lado.upper()}: {status_text}", (int(ombro_pos[0]-90), int(ombro_pos[1]-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        # --- EXIBIÇÃO ---
        cv2.imshow("Análise de Socos", frame)
        # O waitKey controla a velocidade de exibição. 1 é o mais rápido possível.
        key = cv2.waitKey(1) & 0xFF 
        if key == 27 or key == ord('q'): break

    except KeyboardInterrupt:
        print("\nAnálise interrompida pelo usuário.")
        break
    except Exception as e:
        if 'index' in str(e) and 'out of bounds' in str(e): pass
        else: print(f"Ocorreu um erro inesperado: {e}"); break

# --- FINALIZAÇÃO ---
video.release()
cv2.destroyAllWindows()
# ... (código para salvar o arquivo)
# (O código para salvar o arquivo permanece o mesmo)
with open("dados_socos.txt", "w", encoding="utf-8") as f:
    f.write("=== RESULTADOS DA ANÁLISE DE SOCOS ===\n\n")
    if not dados_socos:
        f.write("Nenhum soco foi detectado.\n")
    else:
        for i, d in enumerate(dados_socos, 1):
            f.write(f"Soco {i} ({d['braço'].upper()}):\n")
            f.write(f"  - Tempo: {d['tempo']:.2f}s\n")
            f.write(f"  - Distância: {d['distancia']:.1f}px\n")
            f.write(f"  - Velocidade: {d['velocidade']:.1f}px/s\n\n")
    f.write("=======================================\n")
print(f"✅ Análise concluída. {len(dados_socos)} socos registrados em 'dados_socos.txt'")