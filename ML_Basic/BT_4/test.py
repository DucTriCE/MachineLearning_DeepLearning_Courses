import pandas as pd
import numpy as np
from tabulate import tabulate

df = pd.read_csv("project_df.csv", header=0)
new_df = tabulate(df,headers= 'keys', tablefmt= 'psql')

#Thống kê trình độ học vấn
print(df['educational_level'].value_counts(normalize=True))

#In 20 khách hàng thu nhập cá nhân cao nhất
tmp_df = df.sort_values(by=['annual_income'], inplace=False, ascending=False)
print(tmp_df[['customer_id','annual_income','year_of_birth']].head(20))
    #Cach2
# tmp_df = df.nlargest(20,'annual_income')
# print(tmp_df[['customer_id','annual_income']])

#In ra khách hàng sinh sau năm 1960 và thu nhập >50000
new_df = df[(df.year_of_birth>1960)&(df.annual_income>50000)]
print(new_df)

#In 20 khách hàng thu nhập cao nhất có lương trên 50000$ và sinh năm sau 1960
tmp_df = df.sort_values(by=['annual_income'], inplace=False, ascending=False)
tmp_df = tmp_df[(tmp_df.year_of_birth>1960)&(tmp_df.annual_income>50000)]
print(tmp_df[['customer_id','annual_income']].head(20))

#In khách hàng có tình trạng hôn nhân đã kết hôn hoặc đã ly hôn
new_df = df[(df.marital_status=='Married')|(df.marital_status=='Divorced')]
print(new_df[['customer_id','marital_status']])

#In ra mức thu nhập trung bình theo học vấn
mean_values_edu = df.groupby(by = 'educational_level')[['annual_income']].mean()
print(tabulate(mean_values_edu,headers= 'keys', tablefmt= 'psql'))

#In ra mức thu nhập trung bình theo học vấn và tình trạng hôn nhân
mean_values_edu = df.groupby(by = ['educational_level','marital_status'])[['annual_income']].mean()
print(tabulate(mean_values_edu,headers= 'keys', tablefmt= 'psql'))
    #Cach2
# pivot_table = pd.dfFrame(pd.pivot_table(df, values=['annual_income'], index=['marital_status'], columns=['educational_level'], aggfunc={'annual_income': np.mean}, fill_value=0))
# print(pivot_table)

mean_values_edu = df.groupby(by = ['marital_status'])[['annual_income']].mean()
print(tabulate(mean_values_edu,headers= 'keys', tablefmt= 'psql'))

# PHD --> Highest Salary
#
# Basic --> Single
# Graduation --> Relationship
# HighSchool --> Relationship
# Master --> Single
# PHD --> Relationship
#
# Widowed --> Highest Salary
