import numpy as np
with open(file="Iris.csv", mode="r") as my_file:
    x = []
    y = []
    for line in my_file.readlines()[1:]:
        x.append([float(line.split(",")[1]), float(line.split(",")[2])])
        y.append(int(line.split(",")[-1][0]))

x = np.array(x)
y = np.array(y)

#1a
'''
#Khởi tạo các biến chiều dài, chiều rộng trung bình của đài hoa từng loại
length_average_1 = length_average_2 = length_average_3 = 0
width_average_1 = width_average_2 = width_average_3 = 0

for i in range(150):
    #Nếu loại hoa là loại 1
    if y[i]==0:
        #Tính tổng chiều dài và rộng của đài hoa của loại 1
        length_average_1 += x[i][0]
        width_average_1 += x[i][1]
    #Nếu loại hoa là loại 2
    elif y[i]==1:
        #Tính tổng chiều dài và rộng của đài hoa của loại 1
        length_average_2 += x[i][0]
        width_average_2 += x[i][1]
    #Nếu loại hoa là loại 3
    else:
        #Tính tổng chiều dài và rộng của đài hoa của loại 1
        length_average_3 += x[i][0]
        width_average_3 += x[i][1]

#Chia trung bình của chiều dài và chiều rộng đài hoa từng loại
length_average_1/=50
width_average_1/=50
length_average_2/=50
width_average_2/=50
length_average_3/=50
width_average_3/=50
'''

#2a
'''
#Gán biến chiều dài, chiều rộng của từng loại hoa lớn nhất cho -1
length_max_1, length_max_2, length_max_3 = -1 , -1 , -1
width_max_1, width_max_2, width_max_3 = -1, -1 , -1

for i in range(150):
    #Nếu đài hoa thuộc loại 1
    if y[i]==0:
        #So sánh với chiều dài max
        if x[i][0]>length_max_1:
            #Gán chiều dài max là đài hoa đang xét
            length_max_1=x[i][0]
        #So sánh với chiều rộng max
        if x[i][1]>width_max_1:
            #Gán chiều rộng max là đài hoa đang xét
            width_max_1=x[i][1]
    #Nếu đài hoa thuộc loại 2
    elif y[i]==1:
        if x[i][0]>length_max_2:
            length_max_2=x[i][0]
        if x[i][1]>width_max_2:
            width_max_2=x[i][1]
    #Nếu đài hoa thuộc loại 3
    else:
        if x[i][0]>length_max_3:
            length_max_3=x[i][0]
        if x[i][1]>width_max_3:
            width_max_3=x[i][1]
'''

#3a.1 ( 2 cach )
'''
#Cach1
a, b = map(float,input("Nhap chieu dai va chieu rong cua mot bong hoa: ").split())
min_of_distance = 999999
distance = 0

for x1,y1 in x[0:50]:
    distance = (x1-a)*(x1-a)+(y1-b)*(y1-b)
    if distance<min_of_distance:
        min_of_distance = distance
        class_of_flower = 'Iris setosa'

for x1,y1 in x[50:100]:
    distance = (x1-a)*(x1-a)+(y1-b)*(y1-b)
    if distance<min_of_distance:
        min_of_distance = distance
        class_of_flower = 'Iris virginica'

for x1, y1 in x[100:150]:
    distance = (x1-a)*(x1-a)+(y1-b)*(y1-b)
    if distance < min_of_distance:
        min_of_distance = distance
        class_of_flower = 'Iris versicolor'

print("Bong hoa thuoc loai {}".format(class_of_flower))

#Cach 2 (tối ưu) 
#Tạo mảng lưu các khoảng cách
distance_list = []
#Nhập chiều dài và chiều rộng
a, b = map(float,input("Nhap chieu dai va chieu rong cua mot bong hoa: ").split())

for i in range(150):
    #Lần lượt lưu các khoảng cách từ tọa độ bông hoa input với các bông hoa trong df và loại của nó
    distance_list.append([y[i],(x[i][0]-a)*(x[i][0]-a)+(x[i][1]-b)*(x[i][1]-b)])
    #distance_list = [[1,5.5],[2,6]]
#Sắp xếp danh sách trên với thứ tự sắp xếp tăng dần và sắp xếp theo cột khoảng cách
distance_list.sort(key= lambda x: x[1])
#In ra loại của bông hoa đầu tiên của danh sách sau khi sắp xếp
print("Bong hoa thuoc loai {}".format(distance_list[0][0]+1))
'''

#3a.2 ( 2 cach )
'''
#Cach1 
a, b = map(float,input("Nhap chieu dai va chieu rong cua mot bong hoa: ").split())
distance = 0
for x1, y1 in x[0:50]:
    distance += (x1 - a) * (x1 - a) + (y1 - b) * (y1 - b)
min_of_distance = distance
class_of_flower = 'Iris setosa'

for x1, y1 in x[50:100]:
    distance += (x1 - a) * (x1 - a) + (y1 - b) * (y1 - b)
if distance < min_of_distance:
    class_of_flower = 'Iris virginica'

for x1, y1 in x[50:100]:
    distance += (x1 - a) * (x1 - a) + (y1 - b) * (y1 - b)
if distance < min_of_distance:
    class_of_flower = 'Iris versicolor'

print("Bong hoa thuoc loai {}".format(class_of_flower))

#Cach2 (tối ưu)
#Nhập chiều dài và chiều rộng
a, b = map(float,input("Nhap chieu dai va chieu rong cua mot bong hoa: ").split())
#Tạo danh sách có 3 phần tử tương ứng với tổng khoảng cách của từng loại hoa đến bông hoa cần tìm
distance_list = [0,0,0]

#Tính tổng các khoảng cách của riêng mỗi bông hoa từng loại đến với bông hoa nhập vào
for i in range(150):
    if y[i]==0:
        distance_list[0]+=(x[i][0]-a)*(x[i][0]-a)+(x[i][1]-b)*(x[i][1]-b)
    elif y[i]==1:
        distance_list[1]+=(x[i][0]-a)*(x[i][0]-a)+(x[i][1]-b)*(x[i][1]-b)
    else:
        distance_list[2]+=(x[i][0]-a)*(x[i][0]-a)+(x[i][1]-b)*(x[i][1]-b)
        
#Tìm min của danh sách và in ra index của phần tử min đó 
print("Bong hoa thuoc loai {}".format(distance_list.index(min(distance_list))+1))
'''

#1b
'''
#Tạo ra 3 danh sách của mỗi loại và lần lượt sắp xếp thứ tự giảm dần theo chiều dài của đài hoa
list_sorted_1 = sorted(x[0:50],key = lambda x: x[0], reverse=True)
list_sorted_2 = sorted(x[50:100],key = lambda x: x[0], reverse=True)
list_sorted_3 = sorted(x[100:150],key = lambda x: x[0], reverse=True)

#In ra 10 phần tử đầu tiên của mỗi danh sách tương ứng với chiều dài max của mỗi loại hoa
for i in range(10):
    print("Chieu dai lon nhat cua bong hoa {} thuoc loai 1: {}".format(i+1,list_sorted_1[i][0]))
    print("Chieu dai lon nhat cua bong hoa {} thuoc loai 2: {}".format(i+1,list_sorted_2[i][0]))
    print("Chieu dai lon nhat cua bong hoa {} thuoc loai 3: {}".format(i+1,list_sorted_3[i][0]))
'''

#2b ( 2 cach )
'''
#Cach1
sum_list = []
for i in range(150):
    sum_list.append([y[i],sum(x[i])])
sum_list.sort(key=lambda x: x[1],reverse=True)
flowers_1 = flowers_2 = flowers_3 = 0
for i in range(50):
    if sum_list[i][0]==0:
        flowers_1+=1
    elif sum_list[i][0]==1:
        flowers_2+=1
    else:
        flowers_3+=1
print(sum_list)
print("So bong hoa thuoc loai 1: {}".format(flowers_1))
print("So bong hoa thuoc loai 2: {}".format(flowers_2))
print("So bong hoa thuoc loai 3: {}".format(flowers_3))

#Cach2 (tối ưu)
#Tạo các biến để lưu số lượng bông hoa mỗi loại
flowers_1 = flowers_2 = flowers_3 = 0
#Tạo danh sách tổng chiều dài và chiều rộng từng bông hoa
sum_list = []

for i in range(150):
    #Thêm vào danh sách tổng chiều dài và chiều rộng phần tử hoa thứ i
    sum_list.append(sum(x[i]))
    
#Sort danh sách theo thứ tự giảm dần của tổng chiều dài và chiều rộng
sum_list.sort(reverse=True)

for i in range(150):
    #So sánh tổng CD và CR từng bông hoa với phần tử thứ 50 của danh sách vừa được sắp xếp
    if sum(x[i])>=sum_list[49]:
        if y[i]==0:
            flowers_1+=1
        elif y[i]==1:
            flowers_2+=1
        else:
            flowers_3+=1
print("So bong hoa thuoc loai 1: {}".format(flowers_1))
print("So bong hoa thuoc loai 2: {}".format(flowers_2))
print("So bong hoa thuoc loai 3: {}".format(flowers_3))
'''

#3b.1
'''
#5.8 2.7
#Tạo danh sách khoảng cách
distance_list = []
a, b = map(float,input("Nhap chieu dai va chieu rong cua mot bong hoa: ").split())
for i in range(150):
    #Lần lượt lưu các khoảng cách từ tọa độ bông hoa đang xét với các bông hoa trong df và loại của nó
    distance_list.append([y[i],(x[i][0]-a)*(x[i][0]-a)+(x[i][1]-b)*(x[i][1]-b)])

#Sắp xếp danh sách trên với thứ tự sắp xếp tăng dần và sắp xếp theo cột khoảng cách
distance_list.sort(key= lambda x: x[1])

#Tạo một danh sách khoảng cách mới
new_distance_list = []

#Duyệt các phần tử trong danh sách cũ
for i in distance_list:
    #Nếu phần tử đó không trong có trong danh sách mới thì thêm vào danh sách mới đó
    if i not in new_distance_list:
        new_distance_list.append(i)
#[ [1,6.], [1,6.], [2,6.], [2,5.], [3,4.] ]
new = [ [1,6.], [2,6.], [2,5.], [3,4.] ]
#Duyệt từ phần tử thứ 2 của danh sách mới
for i in range(1,len(new_distance_list)):
    #Nếu khoảng cách của phần tử đang xét bằng với khoảng cách phần tử trước đó và loại hoa của 2 phần tử này khác nhau
    if new_distance_list[i][1]==new_distance_list[i-1][1] and new_distance_list[i][0]!=new_distance_list[i-1][0]:
        continue
    #Ngược lại, in ra loại hoa của phần tử trước đó
    else:
        print("Bong hoa thuoc loai {}".format(new_distance_list[i][0]+1))
        break
'''

#3b.2
'''
distance_list = []
a, b = map(float,input("Nhap chieu dai va chieu rong cua mot bong hoa: ").split())

for i in range(150):
    distance_list.append([y[i],(x[i][0]-a)*(x[i][0]-a)+(x[i][1]-b)*(x[i][1]-b)])
distance_list.sort(key= lambda x: x[1])
print(distance_list)

#Tạo mảng chứa các biến đếm các loại hoa
count_list = [0,0,0]
for i in range(150):
    #Duyệt danh sách đã sắp xếp và lần lượt cộng các biến đếm
    count_list[distance_list[i][0]] +=1
    #Nếu 1 trong 3 loại hoa đạt 7 điểm thì in ra loại đó và kết thúc
    if 7 in count_list:
        print("Bong hoa thuoc loai {}".format(count_list.index(7)+1))
        break
'''

#3b.3 ( 2 cach )
'''
#Cach1
distance_list = []
a, b = map(float,input("Nhap chieu dai va chieu rong cua mot bong hoa: ").split())

for i in range(150):
    if (x[i][0]-a)*(x[i][0]-a)+(x[i][1]-b)*(x[i][1]-b)<=4:
        distance_list.append([y[i],(x[i][0]-a)*(x[i][0]-a)+(x[i][1]-b)*(x[i][1]-b)])
print(distance_list)
temp = [x[0] for x in distance_list]
number_of_class = [temp.count(0), temp.count(1), temp.count(2)]
print(number_of_class)

m, n = number_of_class.index(max(number_of_class)), 2-number_of_class[::-1].index(max(number_of_class))
if (m!=n):
    distance_1 = distance_2 = 0
    for i in range(len(distance_list)):
        if (distance_list[i][0]==m):
            distance_1 += distance_list[i][1]
        elif (distance_list[i][0]==n):
            distance_2 += distance_list[i][1]
    distance_1/=number_of_class[m]
    distance_2/=number_of_class[n]
    if distance_1>distance_2:
        print("Bong hoa thuoc loai {}".format(m+1))
    else:
        print("Bong hoa thuoc loai {}".format(n+1))
else:
    print("Bong hoa thuoc loai {}".format(m + 1))


#Cach2 (tối ưu) chưa xét đến k có điểm nào thuộc đường tròn
a, b = map(float,input("Nhap chieu dai va chieu rong cua mot bong hoa: ").split())
#Khởi tạo các biến đếm các loại hoa
flowers_1 = flowers_2 = flowers_3 = 0
#Khởi tạo biến tổng khoảng cách của từng loại hoa
sum_of_dis_1 = sum_of_dis_2 = sum_of_dis_3 = 0

for i in range(150):
    #Nếu khoảng cách một hoa tới hoa đang xét bé hơn hoặc bằng bán kính đường tròn
    if (x[i][0]-a)*(x[i][0]-a)+(x[i][1]-b)*(x[i][1]-b)<=4:
        #Nếu là loại hoa 1
        if y[i]==0:
            #Tăng biến đếm loại hoa 1 lên 1
            flowers_1+=1
            #Tính tổng khoảng cách của loại hoa 1
            sum_of_dis_1+=(x[i][0]-a)*(x[i][0]-a)+(x[i][1]-b)*(x[i][1]-b)
        elif y[i]==1:
            flowers_2+=1
            sum_of_dis_2+=(x[i][0]-a)*(x[i][0]-a)+(x[i][1]-b)*(x[i][1]-b)
        else:
            flowers_3+=1
            sum_of_dis_3+=(x[i][0]-a)*(x[i][0]-a)+(x[i][1]-b)*(x[i][1]-b)

#Tạo ra danh sách khoảng cách với các phần tử của nó lần lượt bao gồm các giá trị: số hoa, khoảng cách trung bình, loại hoa 
distance_list = [(flowers_1, sum_of_dis_1/flowers_1,1),(flowers_2, sum_of_dis_2/flowers_2,2),(flowers_3, sum_of_dis_3/flowers_3,3)]
#Sắp xếp danh sách theo thứ tự giảm dần với cột số hoa
distance_list.sort(reverse=True)

#Nếu số hoa của phần tử đầu bằng với số hoa phần tử 2 ( có 2 class có số điểm bằng nhau)
if distance_list[0][0] == distance_list[1][0]:
    #Xét đến khoảng cách trung bình của chúng
    if distance_list[0][1] > distance_list[1][1]:
        print("Bong hoa thuoc loai {}".format(distance_list[0][2]))
    else:
        print("Bong hoa thuoc loai {}".format(distance_list[1][2]))
else:
    print("Bong hoa thuoc loai {}".format(distance_list[0][2]))
'''