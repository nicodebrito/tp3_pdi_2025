import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

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
        mask_modif_rect = mask_modif.copy()
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_modif)

        for i in range(1, n):  # 0 es fondo
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            

            #Factor de forma
            recorte_bool = (labels == i).astype('uint8')
            #plt.imshow(recorte_bool, cmap='gray'), plt.show()
            ext_cont, hierarchy  = cv2.findContours(recorte_bool, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area = cv2.contourArea(ext_cont[0])
            if area > 0:
                perimeter = cv2.arcLength(ext_cont[0], True)
                rho = 4 * np.pi * area /(perimeter ** 2)
            ##
                if rho > 0.5 and area/area_total > 0.0001 and area/area_total < 0.0005:
                    
                    contours_int, _ = cv2.findContours(recorte_bool,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                    numero = len(contours_int)-1
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
            print(f'Dado {i}: {numero_txt}')

        #plt.imshow(frame, cmap='gray'), plt.show()

        #print(n)

        cv2.imshow('Frame', frame) # Muestra el frame redimensionado.

        frame_number += 1
        if cv2.waitKey(25) & 0xFF == ord('q'): # Espera 25 milisegundos a que se presione una tecla. Si se presiona 'q' se rompe el bucle y se cierra la ventana.
            break
    else:  
        break  

cap.release() # Libera el objeto 'cap', cerrando el archivo.
cv2.destroyAllWindows() # Cierra todas las ventanas abiertas.
