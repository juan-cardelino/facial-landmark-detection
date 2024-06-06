# Este codigo toma un video de camara y lo guarda
# Luego extrae el video guardado y lo preproduce en bucle


import cv2


iter_max = 30 # Largo del video a grabar

# Captura de camara
webcam_cap = cv2.VideoCapture(0)

# Cargar primer frame
ret, frame = webcam_cap.read()

# Dimensiones de frame
h, w = frame.shape[:2]

# Formato de video a guardar
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output/video_prueba.avi',fourcc, 60, (w,h))

# Guardar primer frame
if ret == True:
    cv2.imshow('captura', frame)
    out.write(frame)

iter = 0
while webcam_cap.isOpened():
    # Cargar frame
    ret, frame = webcam_cap.read()
    
    # Si frame no esta vacio, mostrar y guardar
    if ret == True:
        # Mostrar frame
        cv2.imshow('captura', frame)
        # Guardar frame
        out.write(frame)
    
        # Condicion de parada por teclado
        if cv2.waitKey(20) & 0xFF  == ord('q'):
            break
    else:
        # Parada por frame vacio 
        break
    
    # Condicion de parada por largo de video
    if iter > iter_max:
        break
    
    iter = iter+1

# Soltar camara y video
webcam_cap.release()
out.release()
# Cerrar ventana
cv2.destroyAllWindows()

# Abrir video gaurdado
cap = cv2.VideoCapture("output/video_prueba.avi")

# Lista de frame de videos
video =[]

while cap.isOpened():
    # Cargar frame
    ret, frame = cap.read()
    # Si frame no esta vacio, mostrar y guardar en lista de frame
    if ret:
        cv2.imshow("video", frame)
        video.append(frame)
    # Cortar por frame vacio
    else: break
    # Condicion de parada por teclado
    if cv2.waitKey(30) == ord('q'):
        break

# Mostrar en consola condicion de parada
print()
print("Press Q to release")
print()
l = len(video)
iter = 0
# Video en bucle
while True:
    aux = iter%l
    cv2.imshow("video", video[aux])
    iter = aux + 1
    # Condicion de parada por teclado
    if cv2.waitKey(30) == ord('q'):
        break
# Cerrar ventana
cv2.destroyAllWindows()