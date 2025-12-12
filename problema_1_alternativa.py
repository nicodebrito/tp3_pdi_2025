import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def grafica_imagen_contorno(recorte, contorno):
    
    recorte_color = cv2.cvtColor(recorte, cv2.COLOR_GRAY2BGR)

    # Dibujamos el contorno
    cv2.drawContours(
        recorte_color,
        [contorno],  # lista con un solo contorno
        -1,
        (0, 255, 0), # verde
        2
    )

   
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(recorte, cmap="gray")
    plt.title("Recorte binario")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(recorte_color)
    plt.title("Recorte + Contorno")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def grafica_imagen_contorno_2(recorte, contorno):
    
    recorte_color = cv2.cvtColor(recorte, cv2.COLOR_GRAY2BGR)

    # Dibujamos el contorno
    cv2.drawContours(
        recorte_color,
        [contorno],  # lista con un solo contorno
        -1,
        (0, 255, 0), # verde
        2
    )

   
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(recorte, cmap="gray")
    plt.title("Recorte binario")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(recorte_color)
    plt.title("Recorte + Contorno")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


os.makedirs("frames", exist_ok = True)  # Si no existe, crea la carpeta 'frames' en el directorio actual.

# --- Leer un video ------------------------------------------------
cap = cv2.VideoCapture('videos/tirada_2.mp4')  # Abre el archivo de video especificado ('tirada_1.mp4') para su lectura.
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtiene el ancho del video en píxeles usando la propiedad CAP_PROP_FRAME_WIDTH.
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtiene la altura del video en píxeles usando la propiedad CAP_PROP_FRAME_HEIGHT.
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Obtiene los cuadros por segundo (FPS) del video usando CAP_PROP_FPS.
# n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Obtiene el número total de frames en el video usando CAP_PROP_FRAME_COUNT.

area_total = width * height
frame_number = 0

while (cap.isOpened()): # Verifica si el video se abrió correctamente.
    cant_dados = 0
    ret, frame = cap.read() # 'ret' indica si la lectura fue exitosa (True/False) y 'frame' contiene el contenido del frame si la lectura fue exitosa.

    if ret == True:  

        print(frame_number)
        print('*****************')
        frame = cv2.resize(frame, dsize=(int(width/3), int(height/3))) # Redimensiona el frame capturado.

        frame_hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, _ = cv2.split(frame_hsv)
        mask_R1 = (h < 10)
        mask_R2 = (h > 170)
        mask_R = mask_R1 | mask_R2

        mask_S = (s > 100)

        mask = (mask_R & mask_S).astype('uint8') * 255

        elemento_estructural = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    
        mask_modif = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, elemento_estructural)

        #plt.imshow(mask_modif, cmap='gray'), plt.show()
        #mask_modif_rect = mask_modif.copy()
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_modif)

        dados = []

        for i in range(1, n):  # 0 es fondo
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            

            #Factor de forma
            recorte_bool = (labels == i).astype('uint8')
            #plt.imshow(recorte_bool, cmap='gray'), plt.show()
            #ext_cont, hierarchy  = cv2.findContours(recorte_bool, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contours, hierarchy = cv2.findContours(recorte_bool, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if hierarchy is None:
                continue

            idx_dado = None
            #cnt_dado = None

            for j, cnt in enumerate(contours):
                parent = hierarchy[0][j][3]

                if parent == -1: #Contorno padre, el dado
                    area = cv2.contourArea(cnt)
                    perimetro = cv2.arcLength(cnt, True)

                    if perimetro == 0:
                        continue
                    
                    rho = 4 * np.pi * area /(perimetro ** 2)

                    if rho > 0.5 and area/area_total > 0.0001 and area/area_total < 0.0005:
                        idx_dado = j
                        dados.append(stats[i])
                        #grafica_imagen_contorno(recorte_bool,cnt) #Solo para generar imagen para reporte
                        #cnt_dado = cnt
                        break
    
            if idx_dado is None:
                continue
            
            #puntos = []
            puntos = 0
            for j, cnt in enumerate(contours):
                if hierarchy[0][j][3] == idx_dado:
                    #puntos.append(cnt)
                    puntos += 1
                    
            #numero = len(puntos)        
            numero = puntos
            numero_txt = str(numero) if numero>=1 and numero<=6 else ''
            color = (0,255,0)
            margen = 2
            x_new = x - margen
            y_new = y - margen
            w_new = w + 2*margen
            h_new = h + 2*margen
            a=cv2.rectangle(frame, (x_new, y_new), (x_new + w_new, y_new + h_new), color, 1)
            b=cv2.putText(frame, numero_txt, (x_new, y_new-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            #a=cv2.rectangle(mask_modif_rect, (x, y), (x+w, y+h), (255,255,255), 1)
            print(f'Dado {i}: {numero_txt}. Area Dado: {area}. RHO: {rho}. Perimetro Dado: {perimetro}')


        cv2.imshow('Frame', frame) # Muestra el frame redimensionado.
        print(f'Cantidad de datos detectados: {len(dados)}')
        frame_number += 1
        if cv2.waitKey(25) & 0xFF == ord('q'): # Espera 25 milisegundos a que se presione una tecla. Si se presiona 'q' se rompe el bucle y se cierra la ventana.
            break
    else:  
        break  

cap.release() # Libera el objeto 'cap', cerrando el archivo.
cv2.destroyAllWindows() # Cierra todas las ventanas abiertas.
