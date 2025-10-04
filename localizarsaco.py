import cv2

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Saco clicado em: x={x}, y={y}")

video = cv2.VideoCapture("videoteste.mp4")

while True:
    success, img = video.read()
    if not success:
        break
    cv2.imshow("Clique no saco", img)
    cv2.setMouseCallback("Clique no saco", click_event)
    if cv2.waitKey(50) & 0xFF == 27:  # ESC para sair
        break

video.release()
cv2.destroyAllWindows()
