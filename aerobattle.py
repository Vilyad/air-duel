from inavmspapi import MultirotorControl  
from inavmspapi.transmitter import TCPTransmitter  
from agrotechsimapi import SimClient
import cv2
import numpy as np
import time
from collections import deque
from pynput.keyboard import Controller as KeyboardController, Key, Listener

# Глобальная переменная для отслеживания нажатия R
r_pressed = False

def on_press(key):
    global r_pressed
    try:
        if key.char == 'r' or key.char == 'R':
            r_pressed = True
            print("R key pressed - preparing to exit...")
    except AttributeError:
        pass

# Запускаем слушатель клавиш в фоне
listener = Listener(on_press=on_press)
listener.start()

# Добавляем контроллер клавиатуры
keyboard = KeyboardController()

HOST = '127.0.0.1'
PORT = 5762
ADDRESS = (HOST, PORT)

tcp_transmitter = TCPTransmitter(ADDRESS)
tcp_transmitter.connect()
control = MultirotorControl(tcp_transmitter)
client = SimClient(address="127.0.0.1", port=8080)

# Константы
IMG_WIDTH = 640
IMG_HEIGHT = 480
DODGE_THRESHOLD = 50
ENEMY_COLOR_RANGE = ((0, 50, 50), (10, 255, 255))
ENEMY_MIN_AREA = 500
PROJECTILE_HISTORY = 5  # количество кадров для отслеживания траектории

# Глобальные переменные для отслеживания снарядов
projectile_history = {}  # словарь для хранения истории позиций снарядов
projectile_id_counter = 0  # счетчик для идентификации снарядов

time.sleep(2)

# Функция для активации квеста
def activate_quest():
    print("Activating quest with SPACE key...")
    keyboard.press(Key.space)
    time.sleep(0.1)
    keyboard.release(Key.space)
    time.sleep(1)  # Даем время на активацию квеста

def stable_fly(steps, timer):
    for i in range(steps):
        msg = control.send_RAW_RC([1481 + (29 if i % 2 == 0 else 0) - (1 if i % 4 == 0 else 0), 
                                 1518 + (1 if i % 2 == 0 else 0), 
                                 1321 + (9 if i % 2 != 0 else 0) - (0 if i % 4 != 0 else 1), 
                                 1500, 2000, 1000, 1000])
        data_handler = control.receive_msg()    
        time.sleep(timer)
    return

def track_projectiles(current_projectiles):
    """Отслеживание траектории снарядов и определение их направления"""
    global projectile_id_counter, projectile_history
    
    updated_projectiles = []
    danger_projectiles = []
    
    # Обновляем историю для существующих снарядов
    for proj_id, history in list(projectile_history.items()):
        matched = False
        for i, (x, y) in enumerate(current_projectiles):
            # Проверяем расстояние до предыдущей позиции
            if len(history) > 0:
                last_x, last_y = history[-1]
                distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)
                if distance < 30:  # порог для сопоставления
                    history.append((x, y))
                    if len(history) > PROJECTILE_HISTORY:
                        history.popleft()
                    matched = True
                    updated_projectiles.append(i)
                    break
        
        if not matched:
            # Снаряд исчез - удаляем из истории
            del projectile_history[proj_id]
    
    # Добавляем новые снаряды
    for i, (x, y) in enumerate(current_projectiles):
        if i not in updated_projectiles:
            projectile_id_counter += 1
            projectile_history[projectile_id_counter] = deque([(x, y)], maxlen=PROJECTILE_HISTORY)
    
    # Определяем опасные снаряды (входящие)
    for proj_id, history in projectile_history.items():
        if len(history) >= 2:
            # Анализируем траекторию
            first_x, first_y = history[0]
            last_x, last_y = history[-1]
            
            # Определяем направление движения (по вертикали)
            if last_y > first_y:  # снаряд движется вниз (входящий)
                danger_projectiles.append((last_x, last_y, True))
            else:  # снаряд движется вверх (исходящий)
                danger_projectiles.append((last_x, last_y, False))
    
    return danger_projectiles

def detect_objects():
    """Обнаружение объектов с отслеживанием траектории снарядов"""
    color_img = client.get_camera_capture(camera_id=0, is_clear=True)
    if color_img is None or len(color_img) == 0:
        return None, []
    
    # Обнаружение врага через камеру глубины
    depth_img = client.get_camera_capture(camera_id=0, is_clear=True, is_depth=True)
    enemy_pos = None
    if depth_img is not None:
        gray = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            enemy_pos = (x + w//2, y + h//2)
    
    # Обнаружение красных снарядов
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ENEMY_COLOR_RANGE[0], ENEMY_COLOR_RANGE[1])
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    projectiles = []
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 30:
            x, y, w, h = cv2.boundingRect(cnt)
            projectiles.append((x + w//2, y + h//2))
    
    # Отслеживание траектории снарядов
    danger_projectiles = track_projectiles(projectiles)
    
    return enemy_pos, danger_projectiles

def dodge_projectile(projectile_info):
    """Уклонение от входящего снаряда"""
    x, y, is_incoming = projectile_info
    
    if not is_incoming:
        return False  # игнорируем исходящие снаряды
    
    dodge_right = x < IMG_WIDTH//2
    print(f"Dodging {'right' if dodge_right else 'left'} from incoming projectile!")
    
    # Резкое уклонение
    control.send_RAW_RC([1600 if dodge_right else 1400, 1500, 1500, 1500, 2000, 1000, 1500])
    time.sleep(0.3)
    
    # Плавный возврат
    control.send_RAW_RC([1500, 1500, 1500, 1500, 2000, 1000, 1500])
    time.sleep(0.2)
    
    return True

def aim_and_shoot():
    """Наведение и стрельба с проверкой входящих снарядов"""
    enemy_pos, projectiles = detect_objects()
    
    if not enemy_pos:
        print("Target lost!")
        return False
    
    # Проверка на опасные снаряды
    for proj in projectiles:
        x, y, is_incoming = proj
        if is_incoming and abs(x - IMG_WIDTH//2) < DODGE_THRESHOLD and y > IMG_HEIGHT//2:
            if dodge_projectile(proj):
                return False
    
    # Наведение и стрельба
    x, y = enemy_pos
    if x < IMG_WIDTH//2 - 50:
        control.send_RAW_RC([1490, 1589, 1488, 1500, 2000, 1000, 1500])
    elif x > IMG_WIDTH//2 + 50:
        control.send_RAW_RC([1500, 1589, 1488, 1500, 2000, 1000, 1500])
    else:
        control.send_RAW_RC([1491, 1589, 1488, 1500, 2000, 1000, 1500])
        client.call_event_action()
        print("Firing!")
        time.sleep(0.3)
        return True
    
    time.sleep(0.1)
    return False

#============== АКТИВАЦИЯ КВЕСТА ==============
activate_quest()

#============== ВЗЛЕТ И ПОДГОТОВКА ==============
msg = control.send_RAW_RC([1492, 1518, 1000, 1500, 1000, 1000, 1000])
data_handler = control.receive_msg()
time.sleep(0.5)

msg = control.send_RAW_RC([1492, 1518, 1000, 1500, 2000, 1000, 1000])
data_handler = control.receive_msg()
time.sleep(3)

msg = control.send_RAW_RC([1491, 1518, 1330, 1500, 2000, 1000, 1000])
data_handler = control.receive_msg()
time.sleep(1)

msg = control.send_RAW_RC([1491, 1518, 1310, 1500, 2000, 1000, 1000])
time.sleep(0.3)

msg = control.send_RAW_RC([1491, 1519, 1325, 1500, 2000, 1000, 1000])
time.sleep(2)

#============== ОСНОВНОЙ ЦИКЛ ==============
shoots = 0
while not r_pressed and shoots < 12:
    enemy_pos, projectiles = detect_objects()
    
    if enemy_pos:
        print("Enemy detected! Engaging...")
        if aim_and_shoot():
            shoots += 1
    else:
        # Поисковое движение с проверкой входящих снарядов
        for proj in projectiles:
            x, y, is_incoming = proj
            if is_incoming and abs(x - IMG_WIDTH//2) < DODGE_THRESHOLD:
                print("Dodging incoming projectile while searching...")
                dodge_projectile(proj)
                break
        else:
            control.send_RAW_RC([1500, 1589, 1488, 1500, 2000, 1000, 1500])
            time.sleep(0.1)

# Завершение
print("R key pressed - shutting down...")
listener.stop()  # Останавливаем слушатель клавиш
control.send_RAW_RC([1491, 1518, 1324, 1500, 1000, 1000, 1000])