file_1 = open('t2.csv', 'r')

n = file_1.readline()
res_d = dict()

for line in file_1:
    a = line.split()
    res_d[a[0]] = 0
file_1.close()

file_1 = open('t2.csv', 'r')
n = file_1.readline()
for line in file_1:
    a = line.split()
    if a[3] != a[4]:
        res_d[a[0]] += 1
file_1.close()

sort_res_d = {}
sort_res_d_val = sorted(res_d, key = res_d.get)
for w in sort_res_d_val:
    sort_res_d[w] = res_d[w]

bad_res = sort_res_d.popitem()
print(bad_res)