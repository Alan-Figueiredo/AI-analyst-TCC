import cv2
import os

NOME_DO_VIDEO = "videoteste.mp4"
PASTA_DE_SAIDA = "fotos_para_treino"
INTERVALO_DE_FRAMES = 15 # Salva 1 frame a cada 15 (aprox. 2 fotos por segundo)

# Cria a pasta de saída se ela não existir
if not os.path.exists(PASTA_DE_SAIDA):
    os.makedirs(PASTA_DE_SAIDA)

# Abre o vídeo
video = cv2.VideoCapture(NOME_DO_VIDEO)
contador_frames = 0
fotos_salvas = 0

print(f"Iniciando extração de frames do vídeo '{NOME_DO_VIDEO}'...")

while True:
    ret, frame = video.read()
    # Se ret for False, o vídeo acabou
    if not ret:
        break

    # Verifica se é hora de salvar um frame
    if contador_frames % INTERVALO_DE_FRAMES == 0:
        nome_arquivo = os.path.join(PASTA_DE_SAIDA, f"frame_{fotos_salvas:04d}.jpg")
        cv2.imwrite(nome_arquivo, frame)
        fotos_salvas += 1
        print(f"Salvo: {nome_arquivo}")

    contador_frames += 1

# Libera o vídeo e informa a conclusão
video.release()
print(f"\n✅ Extração concluída! {fotos_salvas} fotos salvas na pasta '{PASTA_DE_SAIDA}'.")
