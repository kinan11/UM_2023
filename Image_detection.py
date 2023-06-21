import cv2
import numpy as np

def detect_objects(image, object_color):
    # Konwertowanie obrazu do przestrzeni kolorów HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Określenie zakresu koloru obiektów
    lower_color = np.array(object_color, dtype=np.uint8)
    upper_color = np.array(object_color, dtype=np.uint8)

    # Przygotowanie maski dla obiektów o podanym kolorze
    mask = cv2.inRange(hsv_image, lower_color, upper_color)

    # Wykrywanie konturów obiektów
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Zapisywanie wykrytych obiektów jako elementów listy
    detected_objects = []
    for contour in contours:
        # Ignorowanie małych konturów
        if cv2.contourArea(contour) > 100:  # Możesz dostosować ten próg do Twoich potrzeb
            # Obliczanie współrzędnych prostokąta otaczającego kontur
            x, y, w, h = cv2.boundingRect(contour)

            # Wycięcie obiektu z obrazu
            object_image = image[y:y+h, x:x+w]

            # Dodawanie obiektu do listy
            detected_objects.append(object_image)

    return detected_objects

# Wczytanie obrazu
image = cv2.imread('II.jpg')

# Określenie koloru obiektów do wykrycia (w formacie BGR)
object_color = [0, 0, 255]  # Tutaj przykład dla koloru czerwonego

# Wykrywanie obiektów na obrazie
detected_objects = detect_objects(image, object_color)

# Wyświetlanie wyników
print(f"Liczba wykrytych obiektów: {len(detected_objects)}")

# Zapisywanie wykrytych obiektów jako pliki
for i, obj in enumerate(detected_objects):
    cv2.imwrite(f'object_{i+1}.png', obj)
