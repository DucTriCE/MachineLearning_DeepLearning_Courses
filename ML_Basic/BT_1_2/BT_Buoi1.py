# myList = list(tuple(map(int , input().split())) for i in range(n))

class_1 = [(1,2),(3,1),(2,3)]
class_2 = [(5,7),(8,5),(7,6)]

a, b = map(int,input("Nhap toa do diem H: ").split())
min_of_distance = 999999
class_H = 0
distance = 0

case = int(input("Chon TH 1 hoac 2: "))
if case==1:
    #Tính khoảng cách nhỏ nhất của class 1
    for x,y in class_1:
        distance = (x-a)*(x-a)+(y-b)*(y-b)
        #Tìm khoảng cách nhỏ nhất và gán vào min
        if distance<min_of_distance:
            min_of_distance = distance
            #Gán class của H là class 1
            class_H = 1

    for x,y in class_2:
        distance = (x-a)*(x-a)+(y-b)*(y-b)
        if distance<min_of_distance:
            min_of_distance = distance
            class_H = 2
    print("Class cua H la {}".format(class_H))
else:
    #Tính tổng các khoảng cách trong class_1
    for x,y in class_1:
        distance += (x-a)*(x-a)+(y-b)*(y-b)
    #Gán min là tổng các khoảng cách
    min_of_distance = distance
    #Giả sử class_H là 1
    class_H = 1
    #Gán khoảng cách trở về 0 để thực hiện tính toán cho class_2
    distance = 0
    for x,y in class_2:
        distance += (x-a)*(x-a)+(y-b)*(y-b)
    #So sánh tổng khoảng cách của class_2 với min đã được gán ở trên
    if distance<min_of_distance:
        class_H = 2
    print("Class cua H la {}".format(class_H))


