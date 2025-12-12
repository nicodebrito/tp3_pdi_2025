import cv2
import numpy as np
import matplotlib.pyplot as plt

# Constantes
AREA_MIN_DADO = 1500
AREA_MAX_DADO = 10000
AREA_MIN_PUNTO = 40
AREA_MAX_PUNTO = 300
CIRCULARIDAD_MIN = 0.6  # para considerar un contorno como punto
UMBRAL_ESTABILIDAD = 4000000  # si entre dos frames cambia esta cantidad de píxeles, consideramos que el video no está estable
FRAMES_ESTABLES = 15  # Si el video tiene 15 frames estables seguidos, asumimos que la tirada terminó
MOSTRAR_VIDEO = True

def detectar_dados_rojos(hsv):
    rlow = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([5, 255, 255])) # Máscara para rojo bajo
    #plt.imshow(rlow, cmap='gray')
    #plt.title("resultado máscara rojo bajo")
    #plt.show()
    rhigh = cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255])) # Máscara para rojo alto
    #plt.imshow(rhigh, cmap='gray')
    #plt.title("resultado máscara rojo alto")
    #plt.show()
    mask = cv2.bitwise_or(rlow, rhigh)
    #plt.imshow(mask, cmap='gray')
    #plt.title("resultado máscara roja combinada")
    #plt.show()
    dados = []
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if AREA_MIN_DADO <= area <= AREA_MAX_DADO:
            x, y, w, h = cv2.boundingRect(cnt)
            if 0.6 <= w/float(h) <= 1.4:
                dados.append((x, y, w, h))
    return dados

def contar_puntos(hsv_roi):
    pts_mask = cv2.inRange(hsv_roi, np.array([0, 0, 180]), np.array([180, 60, 255])) 
    #plt.imshow(pts_mask, cmap='gray')
    #plt.title("máscara puntos blancos")
    #plt.show()
    puntos = []
    contours = cv2.findContours(pts_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)
        if not (AREA_MIN_PUNTO <= area <= AREA_MAX_PUNTO and perimetro > 0):
            continue
        circularidad = 4 * np.pi * area / (perimetro ** 2)
        if circularidad <= CIRCULARIDAD_MIN:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        puntos.append((cx, cy))

    return puntos

def dados_en_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    salida = frame.copy()
    valores = []
    for indice, (x, y, w, h) in enumerate(detectar_dados_rojos(hsv)):
        puntos = contar_puntos(hsv[y:y+h, x:x+w])
        num = len(puntos)
        if 1 <= num <= 6:
            cv2.rectangle(salida, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(salida, f"Dado {indice+1}: {num}", (x-50, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
            for cx, cy in puntos:
                cv2.circle(salida, (cx+x, cy+y), 8, (0, 255, 255), -1)
            valores.append(num)
    return salida, valores

videos = ["tirada_1.mp4", "tirada_2.mp4", "tirada_3.mp4", "tirada_4.mp4"] 
for video in videos:
    print(f"Procesando {video}")
    cap = cv2.VideoCapture(video)  # Abre el archivo de video especificado ('tirada_#.mp4') para su lectura.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtiene el ancho del video en píxeles usando la propiedad CAP_PROP_FRAME_WIDTH.
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtiene la altura del video en píxeles usando la propiedad CAP_PROP_FRAME_HEIGHT.
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Obtiene los cuadros por segundo (FPS) del video usando CAP_PROP_FPS.
    out = cv2.VideoWriter(f"resultado_{video}", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_number = 0
    cont_estables = 0
    frame_gris_ant = None
    dados_detectados = None
    estatico = False
    
    while (cap.isOpened()):  # Verifica si el video se abrió correctamente.
        ret, frame = cap.read()  # 'ret' indica si la lectura fue exitosa (True/False) y 'frame' contiene el contenido del frame si la lectura fue exitosa.
        if ret == True:
            frame_number += 1
            
            if not estatico:
                frame_gris= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
                if frame_gris_ant is not None:
                    diff = cv2.absdiff(frame_gris, frame_gris_ant)
                    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    cambio_pixeles = np.sum(thresh)
                    if cambio_pixeles < UMBRAL_ESTABILIDAD: cont_estables += 1
                    else: cont_estables = 0
                    
                    if cont_estables >= FRAMES_ESTABLES:
                        estatico = True
                        dados_detectados, valores_detectados = dados_en_frame(frame)
                        num_dados = len(valores_detectados)
                        print(f"Frame estable: {frame_number}")
                        frame_objetivo = frame_number
                        print(f"{num_dados} dados detectados:")
                        for indice, valor in enumerate(valores_detectados):
                            print(f"  Dado {indice+1}: {valor} puntos")
                        print(f"\nVideo generado: resultado_{video}")
                        print("-" * 40)
                frame_gris_ant = frame_gris
                frame_dados = frame
            else: frame_dados = dados_detectados
            
            if estatico and dados_detectados is not None:
                cv2.putText(frame_dados, f"Video estable - Frame estudiado: {frame_objetivo}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            out.write(frame_dados) 
            if MOSTRAR_VIDEO:
                frame_show = cv2.resize(frame_dados, dsize=(int(width/3), int(height/3)))  
                cv2.imshow(f"Procesando {video}", frame_show)  
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break
        else:
            break
    cap.release()  # Libera el objeto 'cap', cerrando el archivo.
    out.release()  # Libera el objeto 'out', cerrando el archivo.
    if MOSTRAR_VIDEO:
        cv2.destroyAllWindows()  # Cierra todas las ventanas abiertas.