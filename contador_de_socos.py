import cv2
import numpy as np
import time
import json
import logging
from dataclasses import dataclass
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
NOME_DO_VIDEO = "boxe.mp4"
MODELO_POSE = "yolov8m-pose.pt"
MODELO_OBJETO = "best_v4.pt"


@dataclass
class Config:
    # Frequência de inferência (IA a cada N frames)
    FREQUENCIA_IA_INICIAL: int = 3

    # Índices de keypoints (YOLOv8 Pose) - CONFIRME PARA SEU MODELO
    INDICE_QUEIXO: int = 0

    # Limiar de confiança para keypoints
    LIMIAR_CONFIANCA: float = 0.55

    # Parâmetros de detecção de início/reset do soco (em px)
    DIST_QUEIXO_INICIO: float = 90.0
    DIST_QUEIXO_RESET: float = 75.0

    # Janela de suavização da distância (frames)
    JANELA_MEDIA_DIST: int = 5

    # Impacto / alvo
    IMPACT_PADDING: int = 15
    DISTANCIA_PROJECAO_MAO: int = 40
    FRAMES_IMPACTO_MINIMO: int = 2  # impacto deve durar pelo menos N frames

    # Tracking
    LIMITE_FRAMES_PERDIDOS: int = 10

    # Calibração (px -> metros)
    METROS_POR_PIXEL: float = 0.002  # ajuste esse valor conforme seu vídeo

    # FPS de referência para normalização
    FPS_REFERENCIA: float = 30.0


CFG = Config()


# === FUNCOES DE DETECCAO ===
def carregar_modelos(caminho_pose, caminho_objeto):
    """Carrega os modelos YOLO para pose e detecção de objetos."""
    try:
        model_pose = YOLO(caminho_pose)
        model_obj = YOLO(caminho_objeto)

        # Se tiver GPU, você pode forçar:
        # model_pose.to("cuda")
        # model_obj.to("cuda")

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
            return (x1, y1, x2, y2), float(det.conf[0])
    return None, None


def processar_frame_ia(frame, frame_id, model_pose, model_obj, frequencia_ia):
    """Processa o frame com IA (a cada N frames conforme frequência)."""
    keypoints = None
    saco_box = None
    saco_conf = None

    if frame_id % frequencia_ia == 0:
        result_pose = model_pose(frame, verbose=False)
        saco_box, saco_conf = detectar_saco(frame, model_obj)

        if result_pose[0].keypoints and len(result_pose[0].keypoints.xy) > 0:
            keypoints = result_pose[0].keypoints

    return keypoints, saco_box, saco_conf


# === FUNCOES DE TRACKING ===
def atualizar_tracking_saco(saco_box_detectado, ultima_posicao_saco, frames_sem_deteccao):
    """Atualiza o tracking do saco de pancadas."""
    if saco_box_detectado is not None:
        ultima_posicao_saco = saco_box_detectado
        frames_sem_deteccao = 0
    else:
        frames_sem_deteccao += 1

    if ultima_posicao_saco is not None and frames_sem_deteccao < CFG.LIMITE_FRAMES_PERDIDOS:
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
    """Verifica se o ponto de impacto está dentro da área do saco."""
    if saco_box is None:
        return False

    x1, y1, x2, y2 = saco_box
    return (x1 - padding) < ponto_impacto[0] < (x2 + padding) and (
        y1 - padding
    ) < ponto_impacto[1] < (y2 + padding)


def calcular_metricas_soco(tempo_inicio, ponto_inicial, ponto_final, fps):
    """
    Calcula as métricas do soco:
    - tempo (s)
    - distância em px
    - distância em metros
    - velocidade em px/s
    - velocidade normalizada em px/s (corrigida para FPS de referência)
    - velocidade em km/h
    """
    tempo = time.time() - tempo_inicio
    dist_px = np.linalg.norm(np.array(ponto_final) - np.array(ponto_inicial))

    if tempo > 0:
        vel_px_s = dist_px / tempo
        vel_normalizada = vel_px_s * (fps / CFG.FPS_REFERENCIA)

        dist_m = dist_px * CFG.METROS_POR_PIXEL
        vel_m_s = vel_px_s * CFG.METROS_POR_PIXEL
        vel_kmh = vel_m_s * 3.6
    else:
        vel_px_s = vel_normalizada = vel_kmh = 0.0
        dist_m = 0.0

    return tempo, dist_px, dist_m, vel_px_s, vel_normalizada, vel_kmh


# === FUNCOES DE PROCESSAMENTO DE SOCO ===
def atualizar_media_movel(lista_dist, novo_valor, max_len):
    lista_dist.append(float(novo_valor))
    if len(lista_dist) > max_len:
        lista_dist.pop(0)
    return float(sum(lista_dist) / len(lista_dist))


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
    frame_id,
):
    """Processa a detecção e análise de soco para um braço específico."""
    if max(indice_pulso, indice_cotovelo, indice_ombro) >= len(keypoints):
        return None, "N/D", None

    ombro_pos = keypoints[indice_ombro]
    pulso_conf = confs[indice_pulso]
    cotovelo_conf = confs[indice_cotovelo]
    ombro_conf = confs[indice_ombro]

    if (
        pulso_conf < CFG.LIMIAR_CONFIANCA
        or cotovelo_conf < CFG.LIMIAR_CONFIANCA
        or ombro_conf < CFG.LIMIAR_CONFIANCA
    ):
        return None, "N/D", ombro_pos

    pulso_pt = keypoints[indice_pulso]
    cotovelo_pt = keypoints[indice_cotovelo]
    queixo_pt = keypoints[indice_queixo]

    # Calcular ponto de impacto
    ponto_impacto = calcular_ponto_impacto(
        pulso_pt, cotovelo_pt, CFG.DISTANCIA_PROJECAO_MAO
    )

    # Calcular distância do queixo (em px) e suavizar com média móvel
    dist_queixo_impacto = np.linalg.norm(np.array(ponto_impacto) - queixo_pt)

    estado = estado_bracos[lado]
    status_text = "Guarda"

    # inicializar lista de distâncias se não existir
    if "distancias" not in estado:
        estado["distancias"] = []
    dist_media = atualizar_media_movel(
        estado["distancias"], dist_queixo_impacto, CFG.JANELA_MEDIA_DIST
    )

    # Verificar impacto neste frame
    impacto_frame = verificar_impacto(
        ponto_impacto, saco_box, CFG.IMPACT_PADDING
    )

    # Contador de frames em que impacto está ativo
    if "frames_impacto" not in estado:
        estado["frames_impacto"] = 0
    if impacto_frame:
        estado["frames_impacto"] += 1
    else:
        estado["frames_impacto"] = 0

    # Iniciar movimento
    if not estado["movendo"] and dist_media > CFG.DIST_QUEIXO_INICIO:
        estado["movendo"] = True
        estado["inicio"] = time.time()
        estado["p_inicial"] = ponto_impacto
        estado["impactado"] = False
        estado["frame_inicio"] = frame_id

    # Detectar impacto (somente se impacto durar alguns frames)
    if (
        estado["movendo"]
        and not estado["impactado"]
        and estado["frames_impacto"] >= CFG.FRAMES_IMPACTO_MINIMO
    ):
        (
            tempo,
            dist_px,
            dist_m,
            vel_px_s,
            vel_norm,
            vel_kmh,
        ) = calcular_metricas_soco(estado["inicio"], estado["p_inicial"], ponto_impacto, fps)

        # Centro do saco para métricas adicionais
        centro_saco = None
        dist_centro_saco = None
        if saco_box is not None:
            x1, y1, x2, y2 = saco_box
            centro_saco = ((x1 + x2) // 2, (y1 + y2) // 2)
            dist_centro_saco = float(
                np.linalg.norm(np.array(ponto_impacto) - np.array(centro_saco))
            )

        dados_socos.append(
            {
                "braco": lado,
                "tempo": tempo,
                "distancia_px": dist_px,
                "distancia_m": dist_m,
                "velocidade_px_s": vel_px_s,
                "velocidade_normalizada_px_s": vel_norm,
                "velocidade_kmh": vel_kmh,
                "frame_impacto": frame_id,
                "ponto_impacto_x": int(ponto_impacto[0]),
                "ponto_impacto_y": int(ponto_impacto[1]),
                "distancia_centro_saco_px": dist_centro_saco,
            }
        )

        logger.info(
            f"SOCO {lado.upper()} | Frame: {frame_id} | Tempo: {tempo:.2f}s | "
            f"Velocidade: {vel_kmh:.1f} km/h ({vel_px_s:.1f}px/s)"
        )
        estado["impactado"] = True

    # Reset do movimento
    if estado["movendo"] and dist_media < CFG.DIST_QUEIXO_RESET:
        estado["movendo"] = False
        estado["frames_impacto"] = 0

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
        "status": status_text,
    }

    return info_desenho, status_text, None


def processar_bracos_frame(
    frame,
    keypoints_recentes,
    saco_box,
    estado_bracos,
    dados_socos,
    fps,
    frame_id,
):
    """Processa os dois braços para um frame."""
    if keypoints_recentes is None:
        return frame

    keypoints = keypoints_recentes.xy[0].cpu().numpy()
    confs = keypoints_recentes.conf[0].cpu().numpy()

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
            CFG.INDICE_QUEIXO,
            cor,
            estado_bracos,
            saco_box,
            dados_socos,
            fps,
            frame_id,
        )

        frame = desenhar_soco(frame, info_soco, status, ombro_pos)

    return frame


# === FUNCOES DE DESENHO ===
def desenhar_caixa_saco(frame, saco_box, frames_sem_deteccao, saco_conf):
    """Desenha a caixa do saco de pancadas no frame."""
    if saco_box:
        cor_caixa = (0, 255, 0) if frames_sem_deteccao == 0 else (0, 255, 255)
        cv2.rectangle(frame, saco_box[:2], saco_box[2:], cor_caixa, 2)

        texto = "ALVO"
        if saco_conf is not None:
            texto += f" ({saco_conf:.2f})"

        cv2.putText(
            frame,
            texto,
            (saco_box[0], saco_box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            cor_caixa,
            2,
        )
    return frame


def desenhar_soco(frame, info_soco, status, ombro_pos):
    """Desenha os elementos visuais do soco no frame."""
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

    # Desenhar círculo do ponto de impacto
    cv2.circle(frame, info_soco["ponto_impacto"], 9, info_soco["cor"], -1)

    # Desenhar linha de projeção
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
                f.write(
                    f"  - Distancia: {d['distancia_px']:.1f}px ({d['distancia_m']:.3f} m)\n"
                )
                f.write(
                    f"  - Velocidade: {d['velocidade_kmh']:.2f} km/h ({d['velocidade_px_s']:.1f}px/s)\n"
                )
                f.write(
                    f"  - Velocidade Normalizada: {d['velocidade_normalizada_px_s']:.1f}px/s\n"
                )
                if d.get("distancia_centro_saco_px") is not None:
                    f.write(
                        f"  - Distancia ao centro do saco: {d['distancia_centro_saco_px']:.1f}px\n"
                    )
                f.write(f"  - Frame de impacto: {d['frame_impacto']}\n\n")
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
    """Função principal de execução."""
    logger.info("Iniciando análise de vídeo...")

    # Carregar modelos
    model_pose, model_obj = carregar_modelos(MODELO_POSE, MODELO_OBJETO)

    # Abrir vídeo
    video = cv2.VideoCapture(NOME_DO_VIDEO)
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = CFG.FPS_REFERENCIA

    logger.info(f"Video: {NOME_DO_VIDEO} | FPS: {fps}")

    # Ajuste simples da frequência de IA com base em FPS
    if fps > 45:
        frequencia_ia = 4
    elif fps < 25:
        frequencia_ia = 2
    else:
        frequencia_ia = CFG.FREQUENCIA_IA_INICIAL

    logger.info(f"Frequência de inferência da IA: a cada {frequencia_ia} frames")

    # Variáveis de estado
    dados_socos = []
    estado_bracos = {
        "direito": {
            "movendo": False,
            "inicio": 0,
            "p_inicial": None,
            "impactado": False,
            "distancias": [],
            "frames_impacto": 0,
            "frame_inicio": None,
        },
        "esquerdo": {
            "movendo": False,
            "inicio": 0,
            "p_inicial": None,
            "impactado": False,
            "distancias": [],
            "frames_impacto": 0,
            "frame_inicio": None,
        },
    }

    ultima_posicao_saco = None
    frames_sem_deteccao = 0
    keypoints_recentes = None
    saco_box_recente = None
    saco_conf_recente = None

    frame_id = 0

    # Loop principal
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                logger.info("\nFim do vídeo.")
                break

            frame_id += 1

            # Processar IA
            keypoints_novos, saco_box_detectado, saco_conf = processar_frame_ia(
                frame, frame_id, model_pose, model_obj, frequencia_ia
            )

            # Atualizar dados recentes
            if keypoints_novos is not None:
                keypoints_recentes = keypoints_novos
            if saco_box_detectado is not None:
                saco_box_recente = saco_box_detectado
                saco_conf_recente = saco_conf

            # Atualizar tracking do saco
            saco_box, ultima_posicao_saco, frames_sem_deteccao = atualizar_tracking_saco(
                saco_box_detectado, ultima_posicao_saco, frames_sem_deteccao
            )
            if saco_box is None:
                saco_box = saco_box_recente

            # Desenhar caixa do saco
            frame = desenhar_caixa_saco(
                frame, saco_box, frames_sem_deteccao, saco_conf_recente
            )

            # Processar braços e desenhar
            frame = processar_bracos_frame(
                frame,
                keypoints_recentes,
                saco_box,
                estado_bracos,
                dados_socos,
                fps,
                frame_id,
            )

            # Exibir frame
            cv2.namedWindow("Analise de Socos", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Analise de Socos", frame.shape[1], frame.shape[0])
            cv2.imshow("Analise de Socos", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                logger.info("\nAnálise interrompida pelo usuário.")
                break

    except KeyboardInterrupt:
        logger.info("\nAnálise interrompida pelo usuário (KeyboardInterrupt).")
    except Exception as e:
        logger.error(f"Erro durante análise: {e}", exc_info=True)
    finally:
        # Finalizar
        video.release()
        cv2.destroyAllWindows()

        # Salvar resultados
        salvar_dados_txt(dados_socos)
        salvar_dados_json(dados_socos)
        salvar_dados_csv(dados_socos)

        logger.info(f"Análise concluída. {len(dados_socos)} socos registrados.")


if __name__ == "__main__":
    main()
