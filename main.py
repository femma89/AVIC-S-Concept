import cv2
import numpy as np
import datetime
import argparse

# Define una variable global para almacenar la región de interés (ROI)
roi = None

# Variable global para rastrear el estado de restablecimiento de ROI
reset_roi = False

# Variable global para rastrear el estado de arrastre del mouse
dragging = False

# Función para dibujar la ROI en un fotograma
def draw_roi(frame):
    if roi is not None and len(roi) == 2:
        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, roi[0], roi[1], (0, 255, 0), 2)
        return frame_copy
    else:
        return frame

# Función para manejar el evento del mouse
def select_roi(event, x, y, flags, param):
    global roi, reset_roi, dragging

    if event == cv2.EVENT_LBUTTONDOWN:
        roi = [(x, y)]
        dragging = True
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        frame_copy = frame.copy()
        roi_temp = roi.copy()
        roi_temp.append((x, y))
        cv2.rectangle(frame_copy, roi_temp[0], roi_temp[1], (0, 255, 0), 2)
        cv2.imshow('Objetos de color', frame_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        roi.append((x, y))
        dragging = False
        if len(roi) == 2:  # Verifica que haya al menos dos puntos en la ROI
            cv2.rectangle(frame, roi[0], roi[1], (0, 255, 0), 2)
    elif event == cv2.EVENT_RBUTTONDOWN:  # Detecta clic derecho para restablecer ROI
        reset_roi = True

parser = argparse.ArgumentParser(description='Detector de objetos de color.')
parser.add_argument('--color', default='rojo', help='Color a detectar (por ejemplo, "rojo", "verde", "azul").')
args = parser.parse_args()

# Abre el vídeo
video_capture = cv2.VideoCapture('VideoConcept3.mp4')  # O cambia a 0 para usar una cámara en tiempo real

# Inicializa el fotograma anterior
prev_frame = None

while True:
    # Captura un fotograma del vídeo
    ret, frame = video_capture.read()

    if not ret:
        break

    color = args.color.lower()  # Convierte el color a minúsculas para que sea insensible a mayúsculas

    # Restablece ROI si se ha solicitado
    if reset_roi:
        roi = None
        reset_roi = False

    # Si no se ha definido una ROI o la ROI no tiene dos puntos, permite seleccionar una
    if roi is None or len(roi) != 2:
        cv2.imshow('Objetos de color', draw_roi(frame))  # Muestra la ROI en el fotograma
        cv2.setMouseCallback('Objetos de color', select_roi)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        continue

    # Restablece prev_frame si la ROI se ha restablecido
    if prev_frame is not None and reset_roi:
        prev_frame = None

    if color == 'rojo':
        lower_color = np.array([0, 0, 150])  # Umbral inferior para el color rojo en el espacio BGR
        upper_color = np.array([80, 80, 255])  # Umbral superior para el color rojo en el espacio BGR
    elif color == 'blanco':
        lower_color = np.array([200, 200, 200])  # Umbral inferior para el color blanco en el espacio BGR
        upper_color = np.array([255, 255, 255])  # Umbral superior para el color blanco en el espacio BGR
    # ... (otros colores)

    # Aplica la ROI al fotograma
    if len(roi) == 2:
        x1, y1 = roi[0]
        x2, y2 = roi[1]
        frame_roi = frame[y1:y2, x1:x2]

        # Redimensiona prev_frame para que coincida con las dimensiones de frame_roi
        if prev_frame is not None:
            prev_frame = cv2.resize(prev_frame, (frame_roi.shape[1], frame_roi.shape[0]))

        color_mask = cv2.inRange(frame_roi, lower_color, upper_color)

        # Detección de movimiento
        if prev_frame is not None:
            diff_frame = cv2.absdiff(frame_roi, prev_frame)
            gray_diff = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
            _, motion_mask = cv2.threshold(gray_diff, 5, 255, cv2.THRESH_BINARY)

            # Encuentra contornos de objetos en movimiento del color especificado
            motion_color_mask = cv2.bitwise_and(color_mask, motion_mask)
            contours, _ = cv2.findContours(motion_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours = []  # Define contours como una lista vacía si no hay fotograma anterior

        # Restaura prev_frame para el siguiente ciclo
        prev_frame = frame_roi.copy()

        for contour in contours:
            # Filtra contornos pequeños
            if cv2.contourArea(contour) > 1000:
                # Dibuja un rectángulo alrededor del objeto en movimiento del color especificado
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Guarda la captura de pantalla en la carpeta "eventos" con fecha y hora
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                file_path = f"Eventos/evento_{timestamp}.jpg"
                cv2.imwrite(file_path, frame_roi)

    # Muestra el vídeo con objetos detectados y la ROI
    cv2.imshow('Objetos de color', draw_roi(frame))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la captura de vídeo y cierra las ventanas
video_capture.release()
cv2.destroyAllWindows()
