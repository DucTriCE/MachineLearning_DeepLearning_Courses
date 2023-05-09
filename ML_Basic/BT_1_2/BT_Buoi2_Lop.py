tien_luong, tuoi, tt_honnhan, so_con = map(int, input().split())
thue = 0
if tt_honnhan==1:
    thue+=0.05
elif tt_honnhan==2:
    thue+=0.03
else:
    thue+=0.02

if so_con==1:
    thue-=0.03
elif so_con==2:
    thue-=0.05
elif so_con>=3:
    thue=thue-0.05-(so_con-2)*0.01

if 10<tien_luong<20:
    thue+=0.1
elif tien_luong>=20:
    thue_luytien = (tien_luong%20)*0.001
    if thue_luytien>0.05:
        thue_luytien=0.05
    thue+=0.1+thue_luytien

if 18<=tuoi<=30:
    thue+=0.06
elif 31<=tuoi<=45:
    thue+=0.09
elif 46<=tuoi<=60:
    thue=thue+0.09-(tuoi-45)*0.001
else:
    thue+=0.04

print(tien_luong-tien_luong*thue)




