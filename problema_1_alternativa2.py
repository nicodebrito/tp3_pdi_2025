import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


os.makedirs("frames", exist_ok = True)  # Si no existe, crea la carpeta 'frames' en el directorio actual.

# --- Leer un video ------------------------------------------------
cap = cv2.VideoCapture('videos/tirada_4.mp4')  # Abre el archivo de video especificado ('tirada_1.mp4') para su lectura.
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtiene el ancho del video en píxeles usando la propiedad CAP_PROP_FRAME_WIDTH.
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtiene la altura del video en píxeles usando la propiedad CAP_PROP_FRAME_HEIGHT.
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Obtiene los cuadros por segundo (FPS) del video usando CAP_PROP_FPS.
# n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Obtiene el número total de frames en el video usando CAP_PROP_FRAME_COUNT.

area_total = width * height
frame_number = 0
dados_anterior_frame = []
suma_frame_a_frame = 0
estado_dados_frame = []
distancias_hist = []
distancias = []

while (cap.isOpened()): # Verifica si el video se abrió correctamente.
    cant_dados = 0
    ret, frame = cap.read() # 'ret' indica si la lectura fue exitosa (True/False) y 'frame' contiene el contenido del frame si la lectura fue exitosa.

    if ret == True:  

        print(frame_number)
        print('*****************')
        frame = cv2.resize(frame, dsize=(int(width/3), int(height/3))) # Redimensiona el frame capturado.

        frame_hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, _ = cv2.split(frame_hsv)
        mask_R = (h < 10) | (h > 170)
        mask_S = (s > 100)
        mask = (mask_R & mask_S).astype('uint8') * 255
        elemento_estructural = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
        mask_modif = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, elemento_estructural)

        #plt.imshow(mask_modif, cmap='gray'), plt.show()
        mask_modif_rect = mask_modif.copy()
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

            for j, cnt in enumerate(contours):
                parent = hierarchy[0][j][3]

                if parent == -1: #Contorno padre, el dado
                    
                    area = cv2.contourArea(cnt)
                    perimetro = cv2.arcLength(cnt, True)
                    if perimetro == 0:
                        continue
                    rho = 4 * np.pi * area /(perimetro ** 2)

                    if rho > 0.65 and area/area_total > 0.0001 and area/area_total < 0.0005:
                        
                        idx_dado = j
                        dados.append(stats[i])

                        puntos = 0
                        for j, cnt in enumerate(contours):
                            if hierarchy[0][j][3] == idx_dado:
                                #puntos.append(cnt)
                                puntos += 1


                        print(f'Dado {i}. Area Dado: {area}. Relacion area/areatotal: {area/area_total}. RHO: {rho}. Perimetro Dado: {perimetro}')
                        break
                    else:
                        print(f'No Dado {i}. Area Dado: {area}. Relacion area/areatotal: {area/area_total}. RHO: {rho}. Perimetro Dado: {perimetro}')
    
            if idx_dado is None:
                continue
        
        if (len(dados) == len(dados_anterior_frame)) and len(dados)==5:
            dados = sorted(dados, key=lambda s: (s[1], s[0]))
            dados_anterior_frame = sorted(dados_anterior_frame, key=lambda s: (s[1], s[0]))
            distancias = []
            for s1, s2 in zip(dados, dados_anterior_frame):
                
                # centroides
                cx1 = s1[0] + s1[2] / 2
                cy1 = s1[1] + s1[3] / 2
                cx2 = s2[0] + s2[2] / 2
                cy2 = s2[1] + s2[3] / 2

                # distancia
                dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2) ** 0.5
                distancias.append(dist)
            print('Distancias: ',distancias)
            print('Suma: ', sum(distancias))
            
            if sum(distancias) <6 :
                suma_frame_a_frame = suma_frame_a_frame + 1 
            else: 
                suma_frame_a_frame = 0

            if suma_frame_a_frame >= 7:
                for i,dado in enumerate(dados):
                    x=dado[0]
                    y=dado[1]
                    w=dado[2]
                    h=dado[3]
                    color = (0,255,0)
                    margen = 2
                    x_new = x - margen
                    y_new = y - margen
                    w_new = w + 2*margen
                    h_new = h + 2*margen
                    a=cv2.rectangle(frame, (x_new, y_new), (x_new + w_new, y_new + h_new), color, 1)
                    texto = f'Dado {i+1}. Valor: '
                    b=cv2.putText(frame, texto, (x_new, y_new-10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
                
                estado_dados_frame.append([frame_number,'ESTATICOS'])
                distancias_hist.append([frame_number,'ESTATICOS',distancias,sum(distancias)])
            
            else:
                estado_dados_frame.append([frame_number,'EN MOVIMIENTO - ESPERANDO 5 FRAMES'])
                distancias_hist.append([frame_number,'EN MOVIMIENTO - ESPERANDO 5 FRAMES',distancias,sum(distancias)])
        else:
            estado_dados_frame.append([frame_number,'EN MOVIMIENTO - NO TENGO 5 DADOS'])    
            distancias_hist.append([frame_number,'EN MOVIMIENTO - NO TENGO 5 DADOS',distancias,sum(distancias)])

        dados_anterior_frame = dados

        cv2.imshow('Frame', frame) # Muestra el frame redimensionado.
        #cv2.imshow('Frame', mask_modif) # Muestra el frame redimensionado.
        frame_number += 1
        print('*****************')
        if cv2.waitKey(25) & 0xFF == ord('q'): # Espera 25 milisegundos a que se presione una tecla. Si se presiona 'q' se rompe el bucle y se cierra la ventana.
            break
    else:  
        break  

cap.release() # Libera el objeto 'cap', cerrando el archivo.
cv2.destroyAllWindows() # Cierra todas las ventanas abiertas.


#for e in estado_dados_frame:
#    print(f'Frame: {e[0]}. Estado: {e[1]}')
#
#for e in distancias_hist:
#    print(f'Frame: {e[0]}')
#    print(f'Estado: {e[1]}')
#    print(f'Distancias: {e[2]}')
#    print(f'Suma: {e[3]}')
#    print('*******************')