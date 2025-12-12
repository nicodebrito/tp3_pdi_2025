import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

folder_output = 'videos/output'
os.makedirs(folder_output, exist_ok = True)  # Si no existe, crea la carpeta 'output' en el directorio videos.

mask_modif = None
frame_hsv = None


def generar_mascara(frame):
    frame_hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(frame_hsv)
    mask_R = (h < 10) | (h > 170)
    mask_S = (s > 100)
    mask = (mask_R & mask_S).astype('uint8') * 255
    elemento_estructural = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    mask_modif = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, elemento_estructural)
    return mask_modif

def es_dado(stats_dado, recorte_bool_dado, area_total):
    x = stats_dado[0]
    y = stats_dado[1]
    w = stats_dado[2]
    h = stats_dado[3]
    
    #Factor de forma
    #recorte_bool = (labels == i).astype('uint8')
    #plt.imshow(recorte_bool, cmap='gray'), plt.show()

    contours, hierarchy = cv2.findContours(recorte_bool_dado, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is None:
        return -1
    
    idx_dado = None

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
                puntos = 0
                
                for j, cnt in enumerate(contours):
                    if hierarchy[0][j][3] == idx_dado: #Contornos hijos del padel detectado
                        puntos += 1
                #print(f'Dado {i}. Area Dado: {area}. Relacion area/areatotal: {area/area_total}. RHO: {rho}. Perimetro Dado: {perimetro}')
                break
            #else:
                #print(f'No Dado {i}. Area Dado: {area}. Relacion area/areatotal: {area/area_total}. RHO: {rho}. Perimetro Dado: {perimetro}')
    
    if idx_dado is None:
        return -1
    else:
        return puntos

def calcular_distancias(dados_frame_actual, dados_frame_anterior):
    dados_frame_actual = sorted(dados_frame_actual, key=lambda s: (s[1], s[0]))
    dados_frame_anterior = sorted(dados_frame_anterior, key=lambda s: (s[1], s[0]))
    distancias = []
    for s1, s2 in zip(dados_frame_actual, dados_frame_anterior):
        
        # centroides
        cx1 = s1[0] + s1[2] / 2
        cy1 = s1[1] + s1[3] / 2
        cx2 = s2[0] + s2[2] / 2
        cy2 = s2[1] + s2[3] / 2

        # distancia
        dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2) ** 0.5
        distancias.append(dist)
    #print('Distancias: ',distancias)
    #print('Suma: ', sum(distancias))
    return distancias

def procesar_video(file):
    
    print(f'Iniciando procesamiento {file}...')
    path_input = f'videos/{file}'
    path_output = f'{folder_output}/{file}'
    # --- Leer un video ------------------------------------------------
    cap = cv2.VideoCapture(path_input)  # Abre el archivo de video especificado ('tirada_1.mp4') para su lectura.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtiene el ancho del video en píxeles usando la propiedad CAP_PROP_FRAME_WIDTH.
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtiene la altura del video en píxeles usando la propiedad CAP_PROP_FRAME_HEIGHT.
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Obtiene los cuadros por segundo (FPS) del video usando CAP_PROP_FPS.
    # n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Obtiene el número total de frames en el video usando CAP_PROP_FRAME_COUNT.
    dsize=(int(width/3), int(height/3))

    out = cv2.VideoWriter(path_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, dsize)

    area_total = width * height
    frame_number = 0
    dados_frame_anterior = []
    suma_frame_a_frame = 0
    estado_dados_frame = []
    distancias_hist = []
    distancias = []
    flag_info_consola = 1

    while (cap.isOpened()): # Verifica si el video se abrió correctamente.
        cant_dados = 0
        ret, frame = cap.read() # 'ret' indica si la lectura fue exitosa (True/False) y 'frame' contiene el contenido del frame si la lectura fue exitosa.

        if ret == True:  

            #print(frame_number)
            #print('*****************')
            frame = cv2.resize(frame, dsize=(int(width/3), int(height/3))) # Redimensiona el frame capturado.

            #mask = generar_mascara(frame)

            frame_hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, _ = cv2.split(frame_hsv)
            mask_R = (h < 10) | (h > 170)
            mask_S = (s > 100)
            mask = (mask_R & mask_S).astype('uint8') * 255
            elemento_estructural = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
            mask_modif = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, elemento_estructural)
            mask= mask_modif

            if frame_number == 80:

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img1 = frame_rgb
                img2 = mask

                plt.figure(figsize=(10, 4))

                # Imagen 1
                plt.subplot(1, 2, 1)
                plt.imshow(img1, cmap='gray')   # sacá cmap si son RGB
                plt.title("Frame")
                plt.axis("off")

                # Imagen 2
                plt.subplot(1, 2, 2)
                plt.imshow(img2, cmap='gray')
                plt.title("Mascara")
                plt.axis("off")

                plt.tight_layout()
                plt.show()

            dados_frame_actual = []

            n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
            
            mask_u8 = mask.astype(np.uint8) * 255
            frame_contornos = mask_u8.copy()
            frame_contornos_bgr = cv2.cvtColor(frame_contornos, cv2.COLOR_GRAY2BGR)
            
            for i in range(1, n):  # 0 es fondo
                stats_dado = stats[i]
                recorte_bool = (labels == i).astype('uint8')
                #valor = es_dado(stats[i], recorte_bool, area_total) #Si es dado, devuelvo el valor del dado. Si no logra detectar el valor, devuelve 0. Si no es dado, devuelve -1.
                
                x = stats_dado[0]
                y = stats_dado[1]
                w = stats_dado[2]
                h = stats_dado[3]

                #Factor de forma
                #recorte_bool = (labels == i).astype('uint8')
                #plt.imshow(recorte_bool, cmap='gray'), plt.show()
                recorte_bool_dado = recorte_bool
                contours, hierarchy = cv2.findContours(recorte_bool_dado, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                

                if hierarchy is None:
                    return -1

                idx_dado = None

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
                            puntos = 0
                            cv2.drawContours(frame_contornos_bgr, cnt, -1, (0, 0, 255), 2)

                            for j, cnt in enumerate(contours):
                                if hierarchy[0][j][3] == idx_dado: #Contornos hijos del padel detectado
                                    puntos += 1
                                    cv2.drawContours(frame_contornos_bgr, cnt, -1, (0, 255, 0), 2)
                            #print(f'Dado {i}. Area Dado: {area}. Relacion area/areatotal: {area/area_total}. RHO: {rho}. Perimetro Dado: {perimetro}')
                            break
                        #else:
                            #print(f'No Dado {i}. Area Dado: {area}. Relacion area/areatotal: {area/area_total}. RHO: {rho}. Perimetro Dado: {perimetro}')

                if idx_dado is None:
                    valor = -1
                else:
                    valor = puntos

                if frame_number == 80:

                    frame_contornos_rgb = cv2.cvtColor(frame_contornos_bgr, cv2.COLOR_BGR2RGB)
                    img1 = recorte_bool_dado
                    img2 = frame_contornos_rgb

                    plt.figure(figsize=(10, 4))

                    # Imagen 1
                    plt.subplot(1, 2, 1)
                    plt.imshow(img1,cmap='gray' )   # sacá cmap si son RGB
                    plt.title("Recorte de un dado")
                    plt.axis("off")

                    # Imagen 2
                    plt.subplot(1, 2, 2)
                    plt.imshow(img2)
                    plt.title("Contornos detectados")
                    plt.axis("off")

                    plt.tight_layout()
                    plt.show()


                if valor >= 0:
                    stats_completas = np.append(stats[i], valor)
                    dados_frame_actual.append(stats_completas)

            if (len(dados_frame_actual) == len(dados_frame_anterior) == 5):

                distancias = calcular_distancias(dados_frame_actual, dados_frame_anterior)

                if sum(distancias) <= 5:
                    suma_frame_a_frame = suma_frame_a_frame + 1 
                else: 
                    suma_frame_a_frame = 0

                if suma_frame_a_frame >= 7:
                    for i,dado in enumerate(dados_frame_actual):
                        x=dado[0]
                        y=dado[1]
                        w=dado[2]
                        h=dado[3]
                        valor = dado[5]
                        color = (0,0,0)
                        margen = 2
                        x_new = x - margen
                        y_new = y - margen
                        w_new = w + 2*margen
                        h_new = h + 2*margen
                        a=cv2.rectangle(frame, (x_new, y_new), (x_new + w_new, y_new + h_new), color, 1)
                        texto_dado = f'Dado {i+1}' 
                        texto_valor = f'Valor: {valor}'
                        b=cv2.putText(frame, texto_dado, (x_new, y_new - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
                        b=cv2.putText(frame, texto_valor, (x_new, y_new + h_new + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
                    
                    if flag_info_consola:
                        dados_info_consola = dados_frame_actual
                        flag_info_consola = 0

                    estado_dados_frame.append([frame_number,'ESTATICOS'])
                    distancias_hist.append([frame_number,'ESTATICOS',distancias,sum(distancias)])

                else:
                    estado_dados_frame.append([frame_number,'EN MOVIMIENTO - ESPERANDO 5 FRAMES'])
                    distancias_hist.append([frame_number,'EN MOVIMIENTO - ESPERANDO 5 FRAMES',distancias,sum(distancias)])
            else:
                estado_dados_frame.append([frame_number,'EN MOVIMIENTO - NO TENGO 5 DADOS'])    
                distancias_hist.append([frame_number,'EN MOVIMIENTO - NO TENGO 5 DADOS',distancias,sum(distancias)])

            dados_frame_anterior = dados_frame_actual

            #cv2.imshow('Frame', frame) # Muestra el frame redimensionado.
            #cv2.imshow('Frame', mask_modif) # Muestra el frame redimensionado.

            out.write(frame)
            frame_number += 1
            #print('*****************')
            if cv2.waitKey(25) & 0xFF == ord('q'): # Espera 25 milisegundos a que se presione una tecla. Si se presiona 'q' se rompe el bucle y se cierra la ventana.
                break
        else:  
            break  

    cap.release() # Libera el objeto 'cap', cerrando el archivo.
    cv2.destroyAllWindows() # Cierra todas las ventanas abiertas.
    print('----------------------------------------')
    print('Los dados detectados son los siguientes:')
    print('----------------------------------------')
    
    for i,dado_consola in enumerate(dados_info_consola):
        valor = dado_consola[5]
        texto_dado_consola = f'Dado {i+1}. Valor: {valor}' 
        print(texto_dado_consola)

    print('----------------------------------------')
    print(f'Fin procesamiento {file}')
    print('****************************************')


videos = ['1']#,'2','3','4']
for video in videos:
    file = f'tirada_{video}.mp4'
    procesar_video(file)
