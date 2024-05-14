from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2


# Constantes para la detección de carriles y control
CONTROL_COEFFICIENT = 0.010 # Coeficiente utilizado para calcular el ángulo de dirección.
SHOW_IMAGE_WINDOW = True # Bandera para mostrar u ocultar la ventana de procesamiento de imágenes.
MAX_SPEED = 40  # Velocidad máxima
MIN_SPEED = 10  # Velocidad mínima
MAX_SLOPE = 5  # Pendiente máxima de la línea para detección
MIN_LINE_LENGTH = 30  # Longitud mínima de línea detectada
REDUCED_SPEED_FACTOR = 0.5  # Factor de reducción de velocidad en intersecciones
CURVE_TRANSITION_THRESHOLD = 1.0  # Umbral para detectar transiciones de curva

# Configuraciones iniciales
ANGLE = 0.0  # Ángulo actual (en grados).
SPEED = 50  # Velocidad actual de crucero del vehículo en km/h.
STEERING_ANGLE = 0.0 # Ángulo actual del volante.

# Variable global para rastrear la trayectoria del vehículo
previous_lines = None  # Ángulo actual del volante.


def get_image(camera: Camera) -> np.ndarray:
    """
    Captura una imagen de la cámara especificada, la procesa y devuelve una imagen RGB.

    Parámetros:
    - cámara (Camera): El dispositivo de cámara del que se captura la imagen.

    Devuelve:
    - np.ndarray o None: La imagen RGB procesada como un arreglo de NumPy si es exitoso, None de lo contrario.
    """
    # Recuperar una imagen de la cámara como un arreglo de bytes crudos.
    raw_image = camera.getImage()

    # Verificar si la cámara no logró capturar una imagen; si es así, registrar un error y devolver None.
    if raw_image is None:
        print("Imagen de cámara no disponible.")
        return None

    # Convertir el arreglo de bytes crudos en un arreglo de NumPy y remodelarlo a las dimensiones adecuadas.
    # Las dimensiones se derivan de las propiedades de la cámara (altura y anchura),
    # y el arreglo se remodela para tener 4 canales por píxel (presumiblemente RGBA).
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4))

    # Si la imagen tiene 4 canales, convertirla de formato RGBA a RGB para eliminar el canal alfa
    # ya que la mayoría de las operaciones de procesamiento de imágenes en OpenCV esperan imágenes en formato RGB.
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Devolver la imagen RGB lista para más procesamiento.
    return image


def display_image(display: Display, image: np.ndarray) -> None:
    """
    Muestra una imagen dada en el dispositivo de visualización especificado.

    Parámetros:
    - display (Display): El dispositivo de visualización para mostrar la imagen.
    - image (np.ndarray): Los datos de la imagen como un arreglo de NumPy.

    Devoluciones:
    - None
    """
    # Apilar el arreglo de imágenes a lo largo de la tercera dimensión para convertir una imagen en escala de grises a RGB
    # duplicando los datos en escala de grises en cada canal de color.
    image_rgb = np.dstack((image, image, image))

    # Crear una nueva referencia de imagen en la memoria del display para sostener la imagen RGB.
    image_ref = display.imageNew(
        image_rgb.tobytes(), Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0])

    # Pegar la nueva imagen en el display en la posición (0, 0) y refrescar el display.
    display.imagePaste(image_ref, 0, 0, False)


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen RGB a escala de grises, aplica desenfoque gaussiano y realiza un umbral binario para extraer bordes.

    Parámetros:
    - img (np.ndarray): La imagen de entrada en formato RGB.

    Devuelve:
    - np.ndarray: La imagen preprocesada que muestra los bordes.
    """
    # Convertir la imagen RGB de entrada a escala de grises.
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Aplicar desenfoque gaussiano a la imagen en escala de grises para reducir ruido y detalle.
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Realizar umbralización binaria en la imagen desenfocada
    # para crear una imagen binaria donde se resalten los bordes.
    _, edges = cv2.threshold(blur_img, 100, 255, cv2.THRESH_BINARY)

    # Devolver la imagen con detección de bordes.
    return edges


def detect_lines(edges: np.ndarray) -> np.ndarray:
    """
    Detecta segmentos de línea en una imagen utilizando la Transformada de Hough probabilística.

    Parámetros:
    - edges (np.ndarray): La imagen con detección de bordes donde se detectarán las líneas.

    Devuelve:
    - np.ndarray: Un arreglo de líneas detectadas, donde cada línea está representada por los puntos finales (x1, y1, x2, y2).
    """
    # Aplicar la Transformada de Hough Lineal Probabilística para detectar líneas en la imagen de bordes.
    # La función requiere:
    # - Imagen a procesar: bordes
    # - Resolución de distancia del acumulador en píxeles: 1
    # - Resolución angular del acumulador en radianes: np.pi / 180 (1 grado)
    # - Umbral: 50 (número mínimo de intersecciones para detectar una línea)
    # - minLineLength: 50 (la longitud mínima de una línea en píxeles)
    # - maxLineGap: 10 (la brecha máxima permitida entre segmentos de línea para tratarlos como una sola línea)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
    return lines


def draw_lines(img: np.ndarray, lines: np.ndarray) -> None:
    """
    Dibuja líneas en una imagen basada en las coordenadas de línea proporcionadas.

    Parámetros:
    - img (np.ndarray): La imagen en la que se dibujarán las líneas.
    - lines (np.ndarray): Un arreglo de segmentos de línea, donde cada línea está representada por sus puntos finales (x1, y1, x2, y2).

    Devoluciones:
    - None: Esta función modifica la imagen en el lugar y no devuelve ningún valor.
    """
    # Verificar si hay alguna línea para dibujar
    if lines is not None:
        # Recorrer cada línea en el arreglo
        for line in lines:
            x1, y1, x2, y2 = line[0]  # Desempacar las coordenadas de la línea
            # Dibujar cada línea en la imagen dada usando la función line de OpenCV
            # Color: verde (0, 255, 0), Grosor: 2
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


def calculate_steering(lines: np.ndarray, width: int) -> float:
    """
    Calcula el ángulo óptimo de dirección basado en las líneas detectadas y el ancho de la imagen.

    Parámetros:
    - lines (np.ndarray): Un arreglo de líneas detectadas, donde cada línea está representada por sus puntos finales (x1, y1, x2, y2).
    - width (int): El ancho de la imagen en píxeles.

    Devuelve:
    - float: El ángulo de corrección de dirección calculado para centrar el vehículo en el carril.
    """
    # Verificar si hay alguna línea detectada
    if lines:
        # Encontrar la línea que esté más cerca de estar centrada en la imagen y tenga la menor orientación horizontal
        # La clave para encontrar el mínimo es una combinación de la distancia al centro horizontal y un ajuste ponderado
        # para la horizontalidad de la línea (para priorizar líneas más verticales)
        best_line = min(lines, key=lambda line: (
                abs((line[0][0] + line[0][2]) / 2 - width / 2) +  # Distancia al centro
                10 * abs(line[0][2] - line[0][0]) / (abs(line[0][3] - line[0][1]) + 1)  # Desplazamiento
                # horizontal ponderado
        ))

        # Calcular la corrección de dirección necesaria para esta línea
        return calculate_steering_correction(best_line, width)

    # Si no se detectan líneas, devolver 0 indicando que no se necesita corrección de dirección
    return 0


def calculate_steering_correction(line: np.ndarray, width: int) -> float:
    """
    Calcula la corrección de dirección necesaria basada en una línea detectada y el ancho de la imagen.

    Parámetros:
    - line (np.ndarray): Un arreglo que representa los puntos finales de la línea detectada (x1, y1, x2, y2).
    - width (int): El ancho de la imagen en píxeles.

    Devuelve:
    - float: El ángulo de corrección de dirección, donde un valor negativo sugiere un ajuste a la izquierda,
      y un valor positivo sugiere un ajuste a la derecha.
    """
    # Desempacar los puntos finales de la línea
    x1, y1, x2, y2 = line[0]

    # Asegurar que la línea no sea vertical para prevenir la división por cero
    if (x2 - x1) != 0:
        # Calcular la pendiente de la línea
        slope = (y2 - y1) / (x2 - x1)
        # Calcular el ángulo desde la vertical, en grados
        angle_from_vertical = np.arctan(slope) * 180 / np.pi
        # Determinar la distancia horizontal del punto medio de la línea desde el centro de la imagen
        center_distance = abs((x1 + x2) / 2 - width / 2)
        # Calcular el factor de ajuste basado en la distancia al centro
        adjustment_factor = max(1, center_distance / (width / 4))
        # Calcular el ángulo de corrección de dirección
        return -angle_from_vertical * CONTROL_COEFFICIENT * adjustment_factor * 4.5

    # Si la línea es vertical, devolver 0 ya que no se necesita corrección de dirección
    return 0


def filter_lines(lines: np.ndarray) -> list:
    """
    Filtra líneas basadas en su pendiente y longitud para cumplir con criterios específicos.

    Parámetros:
    - lines (np.ndarray): Un arreglo de líneas detectadas, donde cada línea está representada por sus puntos finales (x1, y1, x2, y2).

    Devuelve:
    - np.ndarray: Un arreglo que contiene solo las líneas que cumplen con los criterios de pendiente y longitud especificados.
    """
    filtered_lines = []
    # Verificar explícitamente si 'lines' no es None y tiene elementos
    if lines is not None and len(lines) > 0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calcular la pendiente de la línea, evitando la división por
            # cero al asignar una pendiente infinita si x1 == x2
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
            # Filtrar líneas basadas en la magnitud de la pendiente y la longitud física de la línea
            if abs(slope) < MAX_SLOPE and np.linalg.norm((x1 - x2, y1 - y2)) > MIN_LINE_LENGTH:
                filtered_lines.append(line)
    return filtered_lines


def set_speed(driver: Driver, kmh: float) -> None:
    """
    Establece la velocidad de crucero del vehículo dentro de los límites predefinidos.

    Parámetros:
    - driver (Driver): La interfaz del conductor para controlar el vehículo.
    - kmh (float): La velocidad deseada en kilómetros por hora.

    Devoluciones:
    - None: Esta función modifica la variable de velocidad global y ajusta la velocidad del vehículo.
    """
    global speed
    # Ajustar la velocidad deseada para que esté dentro de los límites de velocidad mínimos y máximos.
    speed = np.clip(kmh, MIN_SPEED, MAX_SPEED)
    # Establecer la velocidad de crucero del vehículo al valor ajustado.
    driver.setCruisingSpeed(speed)


def main():
    # Inicializar componentes y dispositivos del robot
    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    display_img = robot.getDevice("display_image")
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # Establecer la velocidad inicial utilizando la variable global SPEED
    set_speed(driver, SPEED)

    # Bucle principal de simulación
    while robot.step() != -1:
        # Capturar y procesar la imagen de la cámara
        image = get_image(camera)
        if image is not None:
            grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            display_image(display_img, grey_image)

            # Cortar la parte inferior de la imagen para la detección de carriles
            height, width, _ = image.shape
            img_slice = image[int(height * 0.6):, :]
            edges = preprocess_image(img_slice)
            lines = detect_lines(edges)
            lines = filter_lines(lines)
            draw_lines(img_slice, lines)

            # Calcular y aplicar correcciones de dirección basadas en líneas detectadas
            steering_correction = calculate_steering(lines, width)
            driver.setSteeringAngle(steering_correction)

            # Mostrar resultados de detección de bordes y líneas
            cv2.imshow('Edges', edges)
            cv2.imshow('ROI with Lines', img_slice)
            cv2.waitKey(1)

    # Manejar entradas de teclado para ajustes manuales de velocidad y dirección
    key = keyboard.getKey()
    if key == Keyboard.UP:
        set_speed(driver, SPEED + 5)
    elif key == Keyboard.DOWN:
        set_speed(driver, SPEED - 5)
    elif key == Keyboard.RIGHT:
        driver.setSteeringAngle(ANGLE + 0.1)
    elif key == Keyboard.LEFT:
         driver.setSteeringAngle(ANGLE - 0.1)

    # Cerrar todas las ventanas de OpenCV si está habilitado
    if SHOW_IMAGE_WINDOW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
