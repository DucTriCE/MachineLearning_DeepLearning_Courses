from tabulate import tabulate
import pandas as pd
import numpy as np

df = pd.read_csv(filepath_or_buffer="StudentScore.csv", header=0)

#Thêm một cột 'id' vào bên trái với số id bắt đầu từ 1
df.insert(loc=0, column='id', value=df.index + 1)


#Task 1
'''
math_reading_top100 = df.nlargest(100,'math score').merge(df.nlargest(100,'reading score'), on='id', how='inner')
print(math_reading_top100[['id']])
'''

#Task 2
'''
#Reset index vì nếu không có sẽ dẫn đến in ra dữ liệu với index gốc của df
new_df = df.groupby(by='race/ethnicity').apply(lambda x: x.nlargest(20, 'writing score')).reset_index(drop = True)
print(tabulate(new_df[['race/ethnicity','id','writing score']], headers='keys', tablefmt='psql'))
'''

#Task 3
'''
new_df = df[df['race/ethnicity'] == 'group A'].groupby("parental level of education")['id'].count()
print(new_df)
'''

#Task 4
'''
new_df = df
new_df['average'] = new_df[['math score', 'reading score', 'writing score']].mean(axis=1)
new_df = new_df.sort_values('average', ascending = False).groupby('parental level of education')
print(tabulate(new_df, headers='keys', tablefmt='psql'))
'''

#Task 5
'''
new_df = df
new_df['average'] = new_df[['math score', 'reading score', 'writing score']].mean(axis=1)
new_df = new_df.sort_values('average', ascending = False).groupby('lunch')
print(tabulate(new_df, headers='keys', tablefmt='psql'))
'''

#Task 6
'''
new_df = df.groupby('race/ethnicity').apply(lambda x: x.nlargest(10, 'math score')).reset_index(drop=True)
print(tabulate(new_df[['race/ethnicity','id', 'math score']], headers='keys', tablefmt='psql'))
'''

#Task 7
'''
new_df = df
new_df['average'] = new_df[['math score', 'reading score', 'writing score']].mean(axis=1)
new_df_completed = new_df[new_df['test preparation course']=='completed'].groupby(['race/ethnicity','parental level of education','lunch'])['average'].mean()
new_df_none = new_df[new_df['test preparation course']=='none'].groupby(['race/ethnicity','parental level of education','lunch'])['average'].mean()

print(new_df_completed, new_df_none)
'''

