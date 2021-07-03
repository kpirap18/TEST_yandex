from statistics import mean

file_1 = open("t1.txt", "r")

n = file_1.readline()
res_time = []
j = 0
for line in file_1:
    a = line.split()
    # print(a)
    time1 = a[3].split('-')
    time1.extend(a[4].split(':'))
    time2 = a[5].split('-')
    time2.extend(a[6].split(':'))

    for i in range(len(time1)):
        time1[i] = int(time1[i])
        time2[i] = int(time2[i])
    
    time1_int = (((((time1[0] * 12 + time1[1]) * 30) + time1[2]) * 24 + time1[3]) * 60 + time1[4]) * 60 + time1[5]
    time2_int = (((((time2[0] * 12 + time2[1]) * 30) + time2[2]) * 24 + time2[3]) * 60 + time2[4]) * 60 + time2[5]
    buf = float(a[2])

    res_time.append((time2_int - time1_int) / buf)
    
res_time_mean = mean(res_time)
print(res_time_mean)
print(res_time_mean / 30)

file_1.close()