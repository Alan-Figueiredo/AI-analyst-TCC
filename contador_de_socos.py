import cv2
import numpy as np
import time
import json
import logging
from ultralytics import YOLO

# === CONFIGURACAO DE LOGS ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("analise_socos.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# === CONFIGURACOES GLOBAIS ===
NOME_DO_VIDEO = "video3.mp4"
MODELO_POSE = "yolov8m-pose.pt"
MODELO_OBJETO = "best_NewData.pt"

# Parametros de otimizacao
FREQUENCIA_IA = 3

# Indices de keypoints (YOLOv8 Pose)
INDICE_QUEIXO = 0
LIMIAR_CONFIANCA = 0.55

# Parametros de deteccao
DIST_QUEIXO_INICIO = 90
DIST_QUEIXO_RESET = 75
IMPACT_PADDING = 15
DISTANCIA_PROJECAO_MAO = 40

# Tracking
LIMITE_FRAMES_PERDIDOS = 10

# === CALIBRACAO (IMPORTANTE: ajuste para seu vídeo) ===
# Exemplo: se 1 px equivale a 0.002 metros (2 mm), defina METROS_POR_PIXEL = 0.002
METROS_POR_PIXEL = 0.002  # <-- ajuste esse valor conforme medição do seu vídeo

# === FUNCOES DE DETECCAO ===
def carregar_modelos(caminho_pose, caminho_objeto):
    """Carrega os modelos YOLO para pose e deteccao de objetos."""
    try:
        model_pose = YOLO(caminho_pose)
        model_obj = YOLO(caminho_objeto)
        logger.info("Modelos carregados com sucesso")
        return model_pose, model_obj
    except Exception as e:
        logger.error(f"Erro ao carregar modelos: {e}")
        raise


def detectar_saco(frame, modelo):
    """Detecta o saco de pancadas no frame."""
    result_obj = modelo(frame, verbose=False, conf=0.70)
    for det in result_obj[0].boxes:
        nome = modelo.names[int(det.cls)]
        if "saco_pancada" in nome.lower():
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            return (x1, y1, x2, y2)
    return None


def processar_frame_ia(frame, frame_id, model_pose, model_obj, frequencia_ia):
    """Processa o frame com IA (a cada N frames conforme frequencia)."""
    keypoints = None
    saco_box = None

    if frame_id % frequencia_ia == 0:
        result_pose = model_pose(frame, verbose=False)
        saco_box = detectar_saco(frame, model_obj)

        if result_pose[0].keypoints and len(result_pose[0].keypoints.xy) > 0:
            keypoints = result_pose[0].keypoints

    return keypoints, saco_box


# === FUNCOES DE TRACKING ===
def atualizar_tracking_saco(
    saco_box_detectado, ultima_posicao_saco, frames_sem_deteccao
):
    """Atualiza o tracking do saco de pancadas."""
    if saco_box_detectado is not None:
        ultima_posicao_saco = saco_box_detectado
        frames_sem_deteccao = 0
    else:
        frames_sem_deteccao += 1

    if ultima_posicao_saco is not None and frames_sem_deteccao < LIMITE_FRAMES_PERDIDOS:
        saco_box_atual = ultima_posicao_saco
    else:
        saco_box_atual = None
        ultima_posicao_saco = None

    return saco_box_atual, ultima_posicao_saco, frames_sem_deteccao


# === FUNCOES DE CALCULO ===
def calcular_ponto_impacto(pulso_pt, cotovelo_pt, distancia_projecao):
    """Calcula o ponto de impacto projetado do soco."""
    vetor_antebraco = pulso_pt - cotovelo_pt
    norma_vetor = np.linalg.norm(vetor_antebraco)

    if norma_vetor > 0:
        vetor_unitario = vetor_antebraco / norma_vetor
        ponto_impacto = tuple(
            (pulso_pt + vetor_unitario * distancia_projecao).astype(int)
        )
    else:
        ponto_impacto = tuple(pulso_pt.astype(int))

    return ponto_impacto


def verificar_impacto(ponto_impacto, saco_box, padding):
    """Verifica se o ponto de impacto esta dentro da area do saco."""
    if saco_box is None:
        return False

    x1, y1, x2, y2 = saco_box
    return (x1 - padding) < ponto_impacto[0] < (x2 + padding) and (
        y1 - padding
    ) < ponto_impacto[1] < (y2 + padding)


def calcular_metricas_soco(tempo_inicio, ponto_inicial, ponto_final, fps):
    """
    Calcula as metricas do soco:
    - tempo (s)
    - distancia em px
    - distancia em metros
    - velocidade em px/s
    - velocidade normalizada em px/s (corrigida para 30 fps)
    - velocidade em km/h
    """
    tempo = time.time() - tempo_inicio
    dist_px = np.linalg.norm(np.array(ponto_final) - np.array(ponto_inicial))

    if tempo > 0:
        vel_px_s = dist_px / tempo
        # Normalizacao por FPS (mantendo compatibilidade com sua lógica anterior)
        vel_normalizada = vel_px_s * (fps / 30.0)

        # Conversao px -> metros -> km/h
        dist_m = dist_px * METROS_POR_PIXEL
        vel_m_s = vel_px_s * METROS_POR_PIXEL
        vel_kmh = vel_m_s * 3.6
    else:
        vel_px_s = vel_normalizada = vel_kmh = 0.0
        dist_m = 0.0

    return tempo, dist_px, dist_m, vel_px_s, vel_normalizada, vel_kmh


# === FUNCOES DE PROCESSAMENTO DE SOCO ===
def processar_soco_braco(
    keypoints,
    confs,
    lado,
    indice_pulso,
    indice_cotovelo,
    indice_ombro,
    indice_queixo,
    cor,
    estado_bracos,
    saco_box,
    dados_socos,
    fps,
):
    """Processa a deteccao e analise de soco para um braco especifico."""

    if max(indice_pulso, indice_cotovelo, indice_ombro) >= len(keypoints):
        return None, "N/D", None

    ombro_pos = keypoints[indice_ombro]
    pulso_conf = confs[indice_pulso]
    cotovelo_conf = confs[indice_cotovelo]

    if pulso_conf < LIMIAR_CONFIANCA or cotovelo_conf < LIMIAR_CONFIANCA:
        return None, "N/D", ombro_pos

    pulso_pt = keypoints[indice_pulso]
    cotovelo_pt = keypoints[indice_cotovelo]
    queixo_pt = keypoints[indice_queixo]

    # Calcular ponto de impacto
    ponto_impacto = calcular_ponto_impacto(
        pulso_pt, cotovelo_pt, DISTANCIA_PROJECAO_MAO
    )

    # Calcular distancia do queixo (em px)
    dist_queixo_impacto = np.linalg.norm(np.array(ponto_impacto) - queixo_pt)

    estado = estado_bracos[lado]
    status_text = "Guarda"

    # Verificar impacto
    impacto_detectado = verificar_impacto(ponto_impacto, saco_box, IMPACT_PADDING)

    # Iniciar movimento
    if not estado["movendo"] and dist_queixo_impacto > DIST_QUEIXO_INICIO:
        estado["movendo"] = True
        estado["inicio"] = time.time()
        estado["p_inicial"] = ponto_impacto
        estado["impactado"] = False

    # Detectar impacto
    if estado["movendo"] and not estado["impactado"] and impacto_detectado:
        (
            tempo,
            dist_px,
            dist_m,
            vel_px_s,
            vel_norm,
            vel_kmh,
        ) = calcular_metricas_soco(estado["inicio"], estado["p_inicial"], ponto_impacto, fps)

        dados_socos.append(
            {
                "braco": lado,
                "tempo": tempo,
                "distancia_px": dist_px,
                "distancia_m": dist_m,
                "velocidade_px_s": vel_px_s,
                "velocidade_normalizada_px_s": vel_norm,
                "velocidade_kmh": vel_kmh,
            }
        )

        logger.info(
            f"SOCO {lado.upper()} | Tempo: {tempo:.2f}s | Velocidade: {vel_kmh:.1f} km/h ({vel_px_s:.1f}px/s)"
        )
        estado["impactado"] = True

    # Reset do movimento
    if estado["movendo"] and dist_queixo_impacto < DIST_QUEIXO_RESET:
        estado["movendo"] = False

    # Atualizar status
    if estado["movendo"]:
        status_text = "Socando"
        if estado["impactado"]:
            status_text = "IMPACTO!"

    info_desenho = {
        "ombro_pos": ombro_pos,
        "pulso_pt": pulso_pt,
        "ponto_impacto": ponto_impacto,
        "cor": cor,
    }

    return info_desenho, status_text, None


# === FUNCOES DE DESENHO ===
def desenhar_caixa_saco(frame, saco_box, frames_sem_deteccao):
    """Desenha a caixa do saco de pancadas no frame."""
    if saco_box:
        cor_caixa = (0, 255, 0) if frames_sem_deteccao == 0 else (0, 255, 255)
        cv2.rectangle(frame, saco_box[:2], saco_box[2:], cor_caixa, 2)
        cv2.putText(
            frame,
            "ALVO",
            (saco_box[0], saco_box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            cor_caixa,
            2,
        )
    return frame


def desenhar_soco(frame, info_soco, status, ombro_pos):
    """Desenha os elementos visuais do soco no frame."""

    # Caso nao tenha informacao valida (N/D)
    if info_soco is None:
        if ombro_pos is not None:
            cv2.putText(
                frame,
                f"{status}",
                (int(ombro_pos[0] - 80), int(ombro_pos[1] - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (100, 100, 100),
                2,
            )
        return frame

    # Desenhar circulo do ponto de impacto
    cv2.circle(frame, info_soco["ponto_impacto"], 9, info_soco["cor"], -1)

    # Desenhar linha de projecao
    pulso_pt_tuple = tuple(info_soco["pulso_pt"].astype(int))
    cv2.line(frame, pulso_pt_tuple, info_soco["ponto_impacto"], info_soco["cor"], 2)

    # Desenhar status
    text_color = (0, 255, 0) if status == "IMPACTO!" else (255, 255, 255)
    cv2.putText(
        frame,
        f"{status}",
        (int(info_soco["ombro_pos"][0] - 90), int(info_soco["ombro_pos"][1] - 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        text_color,
        2,
    )

    return frame


# === FUNCOES DE SALVAMENTO ===
def salvar_dados_txt(dados_socos, arquivo="dados_socos.txt"):
    """Salva os dados em formato TXT."""
    with open(arquivo, "w", encoding="utf-8") as f:
        f.write("=== RESULTADOS DA ANALISE DE SOCOS ===\n\n")
        if not dados_socos:
            f.write("Nenhum soco foi detectado.\n")
        else:
            for i, d in enumerate(dados_socos, 1):
                f.write(f"Soco {i} ({d['braco'].upper()}):\n")
                f.write(f"  - Tempo: {d['tempo']:.2f}s\n")
                f.write(f"  - Distancia: {d['distancia_px']:.1f}px ({d['distancia_m']:.3f} m)\n")
                f.write(f"  - Velocidade: {d['velocidade_kmh']:.2f} km/h ({d['velocidade_px_s']:.1f}px/s)\n")
                f.write(f"  - Velocidade Normalizada: {d['velocidade_normalizada_px_s']:.1f}px/s\n\n")
        f.write("=======================================\n")
    logger.info(f"Dados salvos em {arquivo}")


def salvar_dados_json(dados_socos, arquivo="dados_socos.json"):
    """Salva os dados em formato JSON."""
    with open(arquivo, "w", encoding="utf-8") as f:
        json.dump(dados_socos, f, indent=4, ensure_ascii=False)
    logger.info(f"Dados salvos em {arquivo}")


def salvar_dados_csv(dados_socos, arquivo="dados_socos.csv"):
    """Salva os dados em formato CSV."""
    import csv

    with open(arquivo, "w", newline="", encoding="utf-8") as f:
        if dados_socos:
            writer = csv.DictWriter(f, fieldnames=dados_socos[0].keys())
            writer.writeheader()
            writer.writerows(dados_socos)
    logger.info(f"Dados salvos em {arquivo}")


# === FUNCAO PRINCIPAL ===
def main():
    """Funcao principal de execucao."""
    logger.info("Iniciando analise de video...")

    # Carregar modelos
    model_pose, model_obj = carregar_modelos(MODELO_POSE, MODELO_OBJETO)

    # Abrir video
    video = cv2.VideoCapture(NOME_DO_VIDEO)
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    logger.info(f"Video: {NOME_DO_VIDEO} | FPS: {fps}")

    # Variaveis de estado
    dados_socos = []
    estado_bracos = {
        "direito": {
            "movendo": False,
            "inicio": 0,
            "p_inicial": None,
            "impactado": False,
        },
        "esquerdo": {
            "movendo": False,
            "inicio": 0,
            "p_inicial": None,
            "impactado": False,
        },
    }

    ultima_posicao_saco = None
    frames_sem_deteccao = 0
    keypoints_recentes = None
    saco_box_recente = None

    frame_id = 0

    # Loop principal
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                logger.info("\nFim do video.")
                break

            frame_id += 1

            # Processar IA
            keypoints_novos, saco_box_detectado = processar_frame_ia(
                frame, frame_id, model_pose, model_obj, FREQUENCIA_IA
            )

            # Atualizar dados recentes
            if keypoints_novos is not None:
                keypoints_recentes = keypoints_novos
            if saco_box_detectado is not None:
                saco_box_recente = saco_box_detectado

            # Atualizar tracking do saco
            saco_box, ultima_posicao_saco, frames_sem_deteccao = (
                atualizar_tracking_saco(
                    saco_box_detectado, ultima_posicao_saco, frames_sem_deteccao
                )
            )

            # Desenhar caixa do saco
            frame = desenhar_caixa_saco(frame, saco_box, frames_sem_deteccao)

            # Processar keypoints
            if keypoints_recentes is not None:
                keypoints = keypoints_recentes.xy[0].cpu().numpy()
                confs = keypoints_recentes.conf[0].cpu().numpy()

                # Processar ambos os bracos
                for lado, (indice_pulso, indice_cotovelo, indice_ombro, cor) in [
                    ("direito", (10, 8, 6, (0, 0, 255))),
                    ("esquerdo", (9, 7, 5, (255, 100, 0))),
                ]:
                    info_soco, status, ombro_pos = processar_soco_braco(
                        keypoints,
                        confs,
                        lado,
                        indice_pulso,
                        indice_cotovelo,
                        indice_ombro,
                        INDICE_QUEIXO,
                        cor,
                        estado_bracos,
                        saco_box,
                        dados_socos,
                        fps,
                    )

                    frame = desenhar_soco(frame, info_soco, status, ombro_pos)

            # Exibir frame
            # Criar janela ajustável (evita corte e zoom automático)
            cv2.namedWindow("Analise de Socos", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Analise de Socos", frame.shape[1], frame.shape[0])

            cv2.imshow("Analise de Socos", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                logger.info("\nAnalise interrompida pelo usuario.")
                break

    except KeyboardInterrupt:
        logger.info("\nAnalise interrompida pelo usuario.")
    except Exception as e:
        logger.error(f"Erro durante analise: {e}", exc_info=True)
    finally:
        # Finalizar
        video.release()
        cv2.destroyAllWindows()

        # Salvar resultados
        salvar_dados_txt(dados_socos)
        salvar_dados_json(dados_socos)
        salvar_dados_csv(dados_socos)

        logger.info(f"Analise concluida. {len(dados_socos)} socos registrados.")


if __name__ == "__main__":
    main()
