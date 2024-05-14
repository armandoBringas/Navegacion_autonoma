from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2


# Constantes para la detección de carriles y control
CONTROL_COEFFICIENT = 0.010  # Coeficiente de control para el cálculo de la dirección
SHOW_IMAGE_WINDOW = True  # Bandera para mostrar ventanas de imagen

# Configuraciones iniciales de ángulo y velocidad
manual_steering = 0  # Variable para dirección manual (sin uso aparente)
steering_angle = 0  # Ángulo inicial de dirección
angle = 0.0  # Ángulo de dirección como valor flotante
speed = 45.0  # Establece la velocidad de crucero inicial a 45 km/h

def empty(): pass  # Función vacía para uso como callback en trackbars

# Valores predefinidos para los trackbars [2,1,4,6,10] ajustados a [13, 21,4,11,100]
p1=13
p2=21
p3=4
p4=11
p5=100

# Creación y configuración de la ventana de parámetros
cv2.namedWindow("Parameters")  # Crea una ventana llamada "Parameters"
cv2.resizeWindow("Parameters", 640, 240)  # Ajusta el tamaño de la ventana a 640x240

# Creación de trackbars para ajuste de parámetros de detección de líneas
cv2.createTrackbar("Rho", "Parameters", 1, 100, empty)  # Trackbar para el parámetro Rho
cv2.createTrackbar("Theta", "Parameters", 1, 100, empty)  # Trackbar para el parámetro Theta
cv2.createTrackbar("Threshold", "Parameters", 1, 100, empty)  # Trackbar para el umbral de detección
cv2.createTrackbar("minLength", "Parameters", 1, 100, empty)  # Trackbar para la longitud mínima de línea
cv2.createTrackbar("maxGap", "Parameters", 1, 100, empty)  # Trackbar para el máximo espacio entre segmentos de línea

# Establecimiento de las posiciones iniciales de los trackbars
cv2.setTrackbarPos('Rho', "Parameters", p1)  # Configura el valor inicial de Rho
cv2.setTrackbarPos('Theta', "Parameters", p2)  # Configura el valor inicial de Theta
cv2.setTrackbarPos('Threshold', "Parameters", p3)  # Configura el valor inicial del umbral
cv2.setTrackbarPos('minLength', "Parameters", p4)  # Configura el valor inicial de la longitud mínima
cv2.setTrackbarPos('maxGap', "Parameters", p5)  # Configura el valor inicial del máximo espacio


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



def process_image(img):
    """
    Procesa una imagen para extraer bordes usando técnicas de procesamiento de imágenes.

    Parámetros:
    - img (np.ndarray): La imagen RGB a procesar.

    Devoluciones:
    - edges (np.ndarray): Imagen de bordes detectados.

    La función convierte la imagen a escala de grises, aplica un desenfoque gaussiano para reducir el ruido y luego usa el detector de bordes de Canny para identificar los bordes en la imagen.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convierte la imagen de RGB a escala de grises
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), cv2.BORDER_DEFAULT)  # Aplica desenfoque gaussiano para suavizar la imagen
    edges = cv2.Canny(gray_img, 200, 250, apertureSize=3)  # Detecta los bordes usando el detector de Canny
    return edges  # Devuelve la imagen de bordes


def detect_lines(edges):
    """
    Detecta líneas en una imagen usando la transformada de Hough.

    Parámetros:
    - edges (np.ndarray): La imagen de bordes donde se buscarán líneas.

    Devoluciones:
    - lines (np.ndarray): Arreglo de líneas detectadas.

    La función obtiene valores dinámicos para los parámetros de la transformada de Hough desde interfaces gráficas de usuario (trackbars) y luego aplica la transformada de Hough para líneas probables para detectar y retornar estas.
    """
    p1 = cv2.getTrackbarPos('Rho', "Parameters") / 10  # Obtiene el valor de Rho desde el trackbar y lo ajusta
    p2 = cv2.getTrackbarPos('Theta', "Parameters") / 10  # Obtiene el valor de Theta desde el trackbar y lo ajusta
    p3 = cv2.getTrackbarPos('Threshold', "Parameters")  # Obtiene el umbral para la detección de líneas
    p4 = cv2.getTrackbarPos('minLength', "Parameters") / 10  # Obtiene la longitud mínima de una línea desde el trackbar y lo ajusta
    p5 = cv2.getTrackbarPos('maxGap', "Parameters") / 10  # Obtiene el máximo espacio entre segmentos conectados de línea desde el trackbar y lo ajusta
    return cv2.HoughLinesP(edges,
                           rho=p1,
                           theta=(np.pi * p2) / 180,  # Convierte Theta a radianes
                           threshold=p3,
                           minLineLength=p4,
                           maxLineGap=p5)  # Aplica la transformada de Hough para líneas


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
    if lines is not None:
        # Encontrar la línea que esté más cerca de estar centrada en la imagen y tenga la menor orientación horizontal
        # La clave para encontrar el mínimo es una combinación de la distancia al centro horizontal y un ajuste ponderado
        # para la horizontalidad de la línea (para priorizar líneas más verticales)
        best_line = min(lines, key=lambda line: (
                abs((line[0][0] + line[0][2]) / 2 - width / 2) +  # Distancia al centro
                10 * abs(line[0][2] - line[0][0]) / (abs(line[0][3] - line[0][1]) + 1)
        # Desplazamiento horizontal ponderado
        ))
        # Calcular la corrección de dirección necesaria para esta línea
        return calculate_steering_correction(best_line, width)
    else:
        return -1


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
        return -angle_from_vertical * CONTROL_COEFFICIENT * adjustment_factor

    # Si la línea es vertical, devolver 0 ya que no se necesita corrección de dirección
    return 0


def set_speed(driver: Driver, kmh: float) -> None:
    """
    Establece la velocidad de crucero del vehículo utilizando la interfaz del conductor.

    Esta función actualiza la variable global de velocidad y ordena a la interfaz del conductor del vehículo ajustar la velocidad de crucero al valor especificado.

    Parámetros:
    - driver (Driver): Una instancia de la clase Driver responsable de controlar la velocidad del vehículo.
    - kmh (float): La velocidad deseada en kilómetros por hora para configurar el vehículo.

    Devoluciones:
    - None
    """
    global speed
    speed = kmh  # Actualiza la variable global a la nueva velocidad
    driver.setCruisingSpeed(speed)  # Ordena al conductor establecer la velocidad de crucero


def set_steering_angle(driver: Driver, wheel_angle: float) -> None:
    """
    Establece el ángulo de dirección del vehículo utilizando la interfaz del conductor.

    Esta función actualiza las variables globales de ángulo y ordena a la interfaz del conductor del vehículo ajustar el ángulo de dirección al valor especificado. Controla efectivamente la dirección en la que el vehículo está dirigiendo.

    Parámetros:
    - driver (Driver): Una instancia de la clase Driver responsable de controlar la dirección del vehículo.
    - wheel_angle (float): El ángulo de dirección deseado en radianes para configurar el vehículo.

    Devoluciones:
    - None
    """
    global angle, steering_angle  # Declara que estamos usando las variables globales
    angle = wheel_angle  # Actualiza la variable global al nuevo ángulo de dirección
    steering_angle = wheel_angle  # Opcionalmente, mantener un seguimiento del ángulo de dirección por separado si es necesario
    driver.setSteeringAngle(angle)  # Ordena al conductor establecer el ángulo de dirección


def regulate(driver: Driver, camera: Camera) -> float:
    """
    Procesa una imagen capturada de la cámara, detecta carriles de conducción y ajusta el ángulo de dirección en consecuencia.

    Esta función recupera una imagen de la cámara, la procesa para detectar bordes, identifica carriles utilizando estos bordes, y luego calcula la corrección de dirección necesaria. El ángulo de dirección del conductor se ajusta basado en este cálculo. Además, redimensiona y muestra las imágenes procesadas.

    Parámetros:
    - driver (Driver): La interfaz del conductor para controlar la dirección del vehículo.
    - camera (Camera): El dispositivo de cámara utilizado para capturar imágenes de la carretera.

    Devoluciones:
    - Optional[float]: La corrección del ángulo de dirección en grados si el procesamiento es exitoso; de lo contrario, None.
    """
    img = get_image(camera)  # Captura imagen de la cámara
    if img is None or img.size == 0:  # Verifica si la imagen es válida
        print("No hay datos de imagen para procesar.")
        return None

    height, width, _ = img.shape  # Obtiene las dimensiones de la imagen
    img_slice = img[int(height * 0.75):, :]  # Corta la parte inferior de la imagen para la detección de carriles
    edges = process_image(img_slice)  # Procesa la imagen cortada para obtener bordes

    lines = detect_lines(edges)  # Detecta líneas de la imagen de bordes
    draw_lines(img_slice, lines)  # Dibuja líneas detectadas en la rebanada de imagen
    steering_correction = calculate_steering(lines, width)  # Calcula la corrección de dirección necesaria
    driver.setSteeringAngle(steering_correction)  # Establece el ángulo de dirección calculado
    print(f"Ángulo de dirección ajustado por: {steering_correction} grados")  # Registra el ajuste de dirección

    # Redimensiona las imágenes para mostrar
    edges = cv2.resize(edges, (200, 100))  # Redimensiona la imagen detectada de bordes
    img_slice = cv2.resize(img_slice, (200, 100))  # Redimensiona ROI con líneas

    # Muestra los resultados
    cv2.imshow('Edges', edges)  # Muestra imagen de bordes detectados
    cv2.imshow('ROI with Lines', img_slice)  # Muestra la rebanada de imagen con líneas
    cv2.waitKey(1)  # Refresca las ventanas de visualización

    return steering_correction  # Devuelve la corrección del ángulo de dirección


def main() -> None:
    """
    Función principal para iniciar y controlar la simulación del robot. Configura dispositivos, controla el vehículo basado en
    líneas detectadas, y responde a entradas del teclado.

    Esta función inicializa los dispositivos del robot, entra en un bucle que procesa imágenes de la cámara para controlar la dirección y la velocidad
    basadas en la detección de carriles, y maneja entradas de usuario para anular el control manual.
    """
    robot = Car()  # Crea una instancia del Car, que representa al robot
    driver = Driver()  # Crea una instancia del Driver, que controla el vehículo

    timestep = int(robot.getBasicTimeStep())  # Obtén el intervalo de tiempo de simulación, convirtiéndolo a entero

    camera = robot.getDevice("camera")  # Accede al dispositivo de cámara
    camera.enable(timestep)  # Activa la cámara con el intervalo de tiempo de simulación

    display_img = robot.getDevice("display_image")  # Accede al dispositivo de pantalla (no utilizado en este fragmento)

    keyboard = Keyboard()  # Crea una instancia de Keyboard para manejar la entrada
    keyboard.enable(timestep)  # Activa el teclado con el intervalo de tiempo de simulación

    set_speed(driver, speed)  # Establece la velocidad inicial
    set_steering_angle(driver, angle)  # Establece el ángulo de dirección inicial

    no_line = 0  # Contador para rastrear cuadros sin líneas detectadas
    flag = False  # Bandera para gestionar el estado de giro correctivo
    while robot.step() != -1:  # Bucle principal de simulación
        image = get_image(camera)  # Captura imagen de la cámara

        if image is not None:
            grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convierte la imagen a escala de grises (no utilizada en este fragmento)
            display_image(display_img, grey_image)


        if not flag:
            line_value = regulate(driver, camera)  # Procesa la imagen y ajusta la dirección basada en líneas detectadas
            if line_value == -1:
                print("No se detectaron líneas, incrementando contador.")
                no_line += 1

        if no_line > 10:  # Verifica si no se han detectado líneas en más de 10 cuadros consecutivos
            if not flag:
                start_time = robot.getTime()  # Tiempo de inicio para la acción correctiva
                flag = True
            print("Iniciando giro correctivo.")
            elapsed_time = robot.getTime() - start_time
            if elapsed_time <= 2:
                driver.setSteeringAngle(0.30)  # Gira bruscamente hacia la derecha
                driver.setCruisingSpeed(18)  # Reduce la velocidad durante el giro
            if elapsed_time > 2:
                print("Reiniciando después de la acción correctiva.")
                no_line = 0
                flag = False
                driver.setSteeringAngle(0.0)  # Restablece la dirección
                driver.setCruisingSpeed(45)  # Restaura la velocidad

        key = keyboard.getKey()  # Obtiene la entrada del teclado
        if key == Keyboard.UP:
            set_speed(driver, speed + 5)  # Aumenta la velocidad
        elif key == Keyboard.DOWN:
            set_speed(driver, speed - 5)  # Disminuye la velocidad
        elif key == Keyboard.RIGHT:
            set_steering_angle(driver, angle + 0.1)  # Gira ligeramente hacia la derecha
        elif key == Keyboard.LEFT:
            set_steering_angle(driver, angle - 0.1)  # Gira ligeramente hacia la izquierda

    if SHOW_IMAGE_WINDOW:
        cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV


if __name__ == "__main__":
    main()
