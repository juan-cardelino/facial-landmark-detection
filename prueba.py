import cv2
import numpy as np
import os
import elipse
import json

def extraer_x_e_y(a):
    aux_x = aux_y = []
    for x, y in a:
        aux_x = aux_x+[x]
        aux_y = aux_y+[y]
    return np.array(aux_x), np.array(aux_y)

def norma(a):
    return np.sqrt(sum(a*a))

def coreccion(points):
    puntos = []
    for i in points[0]:
        puntos.append([points[0][i], points[1][i]])
    aux1 = np.mean(puntos, axis=0)
    aux2 = puntos[1:3]+puntos[4:6]
    normas = []
    for j in aux2:
        normas.append(norma(aux1-j))
    norma_m = np.mean(normas)
    aux3 = []
    for k in aux2:
        aux3.append(list(aux1+(aux1-k)*(1.5*norma_m/norma(aux1-k))))
    aux4 = puntos[0:1]+aux3+puntos[3:4]
    aux_x = aux_y = []
    for x, y in aux4:
        aux_x = aux_x+[x]
        aux_y = aux_y+[y]
    return elipse.fit_ellipse(np.array(aux_x), np.array(aux_y))

def get_best_ellipse_alt_alt(puntos):
    aux1 = np.mean(puntos, axis=0)
    
    aux5 = puntos - aux1
    aux6 = []
    for i in aux5:
        aux6.append(norma(i))
    
    aux7 = np.concatenate((aux6[:1], aux6[3:4]))
    aux7 = np.mean(aux7)
    print(puntos[1:3])
    aux81 = np.mean(puntos[1:3], axis=0)
    print(aux81-aux1)
    aux82 = np.mean(puntos[4:6], axis=0)
    print(aux82)
    aux8 = np.mean([norma(aux81-aux1), norma(aux82-aux1)])
    print(aux8)
    print("aux7",aux7)
    return aux7, aux7/aux8, aux1

def get_best_ellipse_alt_alter(puntos):
    aux1 = np.mean(puntos, axis=0)
    aux2 = np.concatenate((puntos[0:1], puntos[3:4], [np.mean(puntos[1:3], axis=0)], [np.mean(puntos[4:6], axis=0)]))
    aux5 = aux2 - aux1
    aux6 = []
    for i in aux5:
        aux6.append(norma(i))
    
    aux7 = np.mean(aux6[0:2])
    aux8 = np.mean(aux6[2:4])
    return aux7, aux8/aux7, aux1

if 0:
    imagen = 2
    os.listdir("input")[imagen]

    img = cv2.imread("input/"+os.listdir("input")[imagen], 1)

    puntos = [[936.9950561523438,637.3349609375],
                [942.6434326171875,632.6384887695312],
                [948.8971557617188,632.423828125],
                [954.689453125,634.623046875],
                [949.6332397460938,636.6954956054688],
                [943.4136352539062,637.2012329101562]]

    puntos = [[394.60977173, 623.94610596],
                [400.59967041, 620.89489746],
                [406.58483887, 621.26568604],
                [412.18478394, 624.07305908],
                [406.59036255, 624.96112061],
                [400.53363037, 624.74304199]]

    puntos = np.array(puntos)

    aux1 = np.mean(puntos, axis=0)
    aux2 = np.concatenate((puntos[1:3],puntos[4:6]))
    aux3 = (aux2-aux1)*1.5+aux1
    aux4 = np.concatenate((puntos[0:1],aux3,puntos[3:4]))
    
    aux5 = puntos - aux1
    aux6 = []
    for i in aux5:
        aux6.append(norma(i))
    
    aux7 = np.concatenate((aux6[:1], aux6[3:4]))
    aux7 = np.mean(aux7)
    aux8 = np.concatenate((aux6[1:3], aux6[4:6]))
    print(aux8)
    aux8 = np.mean(aux8)
    
    print(aux1)
    print(aux7)
    print(aux8)
    print(aux8/aux7)


    if 0:
        for x, y in puntos:
            cv2.circle(img, (int(x), int(y)), 0, (0, 255, 0), 1)

    #valores_elipse_ojoizq = elipse.get_best_ellipse_alt(puntos)
    valores_elipse_ojoizq = elipse.get_best_ellipse_alt(aux4)
    print(valores_elipse_ojoizq['center'])
    print(valores_elipse_ojoizq['major'])
    print(valores_elipse_ojoizq['major']*valores_elipse_ojoizq["ratio"])
    print(valores_elipse_ojoizq["ratio"])
    

    #Aparentemente el orden de los puntos genera diferencia en el codigo, porque? ni idea

    if 1:
        elipse_ojoizq = elipse.get_ellipse(valores_elipse_ojoizq['center'], valores_elipse_ojoizq['major'], valores_elipse_ojoizq["ratio"], valores_elipse_ojoizq['rotation'], 100)
        elipse_ojoizq = elipse.get_ellipse(aux1, aux7, aux8/aux7, 0, 100)
        
        
        for x, y in elipse_ojoizq:
            cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), 1)
    if 1:    
        for x, y in aux4:
            cv2.circle(img, (int(x), int(y)), 1, (255, 0, 0), 1)



    if 0:
        cv2.imshow("image", img)
        cv2.waitKey(0)


    print(img.shape[:2])

    image_cropped = img[610:640, 380:420]
    #cv2.imwrite("face-detect-greyscale.jpg", img)

    if 1:
        cv2.imshow("image", cv2.resize(image_cropped, (800,600)))

        cv2.waitKey(0)

    cv2.destroyAllWindows()
if 0:
    algo = [1, 2, 3]
    for i in algo:
        print(algo)
    
    algo = (-0.024424451081098857+0.053190070790805184j)
    print(np.abs(algo))

if 0:
    while True:
        try:
            x = int(input("Please enter a number: "))
            break
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")
if 0:        
    class B(Exception):
        pass

    class C(B):
        pass

    class D(C):
        pass

    for cls in [B, C, D]:
        try:
            raise cls()
        except B:
            print("B")
        except D:
            print("D")
        except C:
            print("C")

if 0:
    try:
        raise Exception('spam', 'eggs')
        print("holas")
    except Exception as inst:
        print(type(inst))    # the exception type
        print(inst.args)     # arguments stored in .args
        print(inst)          # __str__ allows args to be printed directly,
                             # but may be overridden in exception subclasses
        x, y = inst.args     # unpack args
        print('x =', x)
        print('y =', y)
if 0:
    puntos = [[936.9950561523438,637.3349609375],
                [942.6434326171875,632.6384887695312],
                [948.8971557617188,632.423828125],
                [954.689453125,634.623046875],
                [949.6332397460938,636.6954956054688],
                [943.4136352539062,637.2012329101562]]
    puntos = np.array(puntos).T
    print(type(puntos[0]))
    print(puntos[0])
    
if 0:
    file = 'FFHQ Json'
    datos = os.listdir(file)
    datos2 = []
    ojos = []
    for i in datos:
        if i.find("deteccion") != -1:
            with open(file+'/'+i) as archivo:
                deteccion = json.load(archivo)
            if deteccion["Error"] == "No se encontraron errores":
                print(i)
                datos2.append(i)
                ojos.append(np.array(deteccion["caras"][0]["ojo derecho"]))
                ojos.append(np.array(deteccion["caras"][0]["ojo izquierdo"]))
    
    #print(datos2)
    
    errores = []
    
    for i in ojos:
        eje_m, ratio, centro = get_best_ellipse_alt_alter(i)
        valores_elipse_ojo = elipse.get_best_ellipse_conical(i)
        if 1:
            print("")
            print("Centro")
            print((valores_elipse_ojo['center']-centro)/valores_elipse_ojo['center'])
            print("Eje  mayor")
            print("teo",valores_elipse_ojo['major'])
            print("exp",eje_m)
            print((valores_elipse_ojo['major']-eje_m)/valores_elipse_ojo['major'])
            print("Eje  menor")
            print(((valores_elipse_ojo['major']*valores_elipse_ojo["ratio"])-(eje_m*ratio))/(valores_elipse_ojo['major']*valores_elipse_ojo["ratio"]))
            print("Ratio")
            print((valores_elipse_ojo["ratio"])/ratio)
            print(valores_elipse_ojo["ratio"])
            print(ratio)
            print("Ratio error relativo")
            print((valores_elipse_ojo["ratio"]-ratio)/valores_elipse_ojo["ratio"])
            print("ratios")
            print("teo", valores_elipse_ojo["ratio"])
            print("exp", ratio)
            print("")
        
        errores.append([np.abs((valores_elipse_ojo['center']-centro)/valores_elipse_ojo['center']), np.abs((valores_elipse_ojo['major']-eje_m)/valores_elipse_ojo['major']), np.abs(((valores_elipse_ojo['major']*valores_elipse_ojo["ratio"])-(eje_m*ratio))/(valores_elipse_ojo['major']*valores_elipse_ojo["ratio"])), np.abs((valores_elipse_ojo["ratio"]-ratio)/valores_elipse_ojo["ratio"])])
    if 1:
        centro = []
        eje_mayor = []
        eje_menor = []
        ratio = []
        for i in errores:
            centro.append(norma(i[0])*100)
            eje_mayor.append(i[1]*100)
            eje_menor.append(i[2]*100)
            ratio.append(i[3]*100)
        print("centro")
        print("Promedio: ",np.mean(centro))
        print("Maximo: ",max(centro))
        print("Eje mayor")
        print("Promedio: ",np.mean(eje_mayor))
        print("Maximo: ",max(eje_mayor))
        print("Eje menor")
        print("Promedio: ",np.mean(eje_menor))
        print("Maximo: ",max(eje_menor))
        print("Ratio")
        print("Promedio: ",np.mean(ratio))
        print("Maximo: ",max(ratio))
        
        print(eje_mayor)
        


if 0:
    aux = []
    aux = [[674.23028564, 179.21231079],[680.81091309, 176.44174194],[687.1696167,  176.87538147],[691.63867188, 180.10926819],[686.73260498, 181.9070282 ],[680.35028076, 181.04837036]]
    aux = np.array(aux)
    print(aux[1:3])
    print(np.mean(aux[1:3], axis=0))
    #print(np.mean(aux, axis=1))

if 0:
    # input
    print("inserte primer numero a sumar:")
    num1 = int(input())
    
    print("inserte segundo numero a sumar:")
    num2 = int(input())
 
    # printing the sum in intager
    print("el resultado de la suma es:")
    print(num1 + num2)