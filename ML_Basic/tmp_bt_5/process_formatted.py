from tabulate import tabulate
import pandas as pd
import numpy as np

df = pd.read_csv(filepath_or_buffer="cities_population_formatted.csv", header=0)
new_df = tabulate(df, headers='keys', tablefmt='psql')

'''
# Sắp xếp và in 10 tên thành phố có dân số lớn nhất

df_asc = df.sort_values(by = ['Population'], inplace=False, ascending=False)
print(tabulate(df_asc.head(10), headers='keys', tablefmt='psql'))
    #Cach2
print(df.nlargest(10,'Population'))

# Sắp xếp và in 10 tên thành phố có dân số nhỏ nhất

#In ra danh sách được sắp xếp từ dưới lên, lấy 10 thành phố
print(tabulate(df_asc[::-1].head(10), headers='keys', tablefmt='psql'))
    #Cach2
print(df.nsmallest(10,'Population'))
'''

'''
#In ra tên Quốc gia có tối thiểu 3 thành phố trong danh sách

#Thêm một cột tổng thành phố bằng cách group các nước lại và đếm
df['Number_of_Cities'] = df.groupby(by='Country')['City'].transform('count')
#In ra 2 cột Tên nước và tổng số thành phố với điều kiện tổng số thành phố >=3 và loại bỏ các dòng lặp lại, với index được reset lại ( đếm từ 0 )
print(df[['Country','Number_of_Cities']][(df.Number_of_Cities>=3)&(~df.duplicated('Country'))].reset_index(drop=True))

    #Cách 2
Number_of_Cities = df.groupby('Country')['City'].count()
print(Number_of_Cities[Number_of_Cities >= 3].reset_index())
'''

'''
#In ra top 5 quốc gia có nhiều thành phố

#Thêm một cột tổng thành phố bằng cách group các nước lại và đếm
df['Number_of_Cities'] = df.groupby(by='Country')['City'].transform('count')
#Tạo một dfframe mới là dfframe được sort theo cột tổng số thành phố
new_df = df.sort_values(by='Number_of_Cities', ascending=False, inplace=False)
#In ra 5 dòng đầu tiên của dfframe với 2 cột là tên nước và tổng số thành phố, với reset_index để khởi tạo lại cột index
print(new_df[['Country','Number_of_Cities']][(~new_df.duplicated('Country'))].reset_index(drop=True).head(5))
    #Cách 2
Number_of_Cities = df.groupby('Country')['City'].count().reset_index()
Sorted_No_Cities = Number_of_Cities.sort_values(by='City',ascending=False, inplace=False).reset_index(drop=True)
print(Sorted_No_Cities.head(5))
'''

'''
#In ra các thành phố có dân số & diện tích đều nằm trong Top 20

#Gộp 2 dfframe với dfframe đầu là danh sách dân số lớn nhất trong top 20 và dfframe th hai là diện tích các vùng đất theo km^2
#Gộp 2 dfframe bằng Merge với how='inner' nhằm chọn những dòng dữ liệu có cột 'Population' giống nhau
top_20 = df.nlargest(20,'Population').merge(df.nlargest(20,'Area KM2'), on='Population', how='inner')
print(tabulate(top_20, headers='keys', tablefmt='psql'))
'''

'''
#Thống kê mật độ dân số theo quốc gia

#Tính tổng dân số theo các nước
total_population = df.groupby(by = 'Country')['Population'].sum()
print(total_population)
#Tính tổng diện tính theo các nước
total_area = df.groupby(by = 'Country')['Area KM2'].sum()
print(total_area)
#Tạo dfframe mật độ = tổng dân số / diện tích trên từng nước
density_of_country = total_population/total_area
print(density_of_country)
'''


#Thống qua các thành phố có dân số lớn nhất của từng quốc gia (chỉ tính của những quốc gia có 2 thành phố xuất hiện trở lên trong bảng)

#Tạo ra một dfframe mới bằng cách group theo Country và tạo một filter với mỗi quốc gia có trên 2 dòng ( 2 thành phố trở lên ) sẽ được giữ lại
new_df = df.groupby(by='Country').filter(func= lambda x: len(x)>=2)
#Tạo một dfframe mới sort trên dfframe vừa tạo ra theo dân số group theo Country, và lấy ra dòng dữ liệu đầu tiên trên mỗi Quốc gia (là thành phố đông dân nhất)
result = new_df.sort_values("Population", ascending=False).groupby("Country").first()
print(tabulate(result, headers='keys', tablefmt='psql'))


