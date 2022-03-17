# БИНАРНЫЙ ПОИСК
def BinarySearch(lys, val):
    first = 0
    last = len(lys)-1
    index = -1
    while (first <= last) and (index == -1):
        mid = (first+last)//2
        if lys[mid] == val:
            index = mid
        else:
            if val<lys[mid]:
                last = mid -1
            else:
                first = mid +1
    return index



############################################################
# Пересечение списков
a = [1, 2, 3, 4, 5, 1, 3, 4]
b = [1, 2, 3, 4, 6, 9, 8, 0, -8]
d = {}

for el in b:
    d[el] = 0

for el in b:
    d[el] += 1

result = []

for el in a:
    if el in b:
        count = d[el]
        if count > 0:  
            result.append(el)  
            d[el] -= 1  

print(result)


# a = [1, 2, 3, 4, 5, 1, 3, 4]
# b = [1, 2, 3, 4, 6, 9, 8, 0, -8]
# c = []
# for i in a:
#     for j in b:
#         if i == j:
#             c.append(i)
#             break
 
# print(c)


############################################################
# RLE

def rle_encode(data):
    en = ''
    prev_char = ''
    char = ''

    if not data:
        return ''

    for char in data:
        if char != prev_char:
            if prev_char:
                if count == 1:
                    en += prev_char
                else:
                    en += prev_char + str(count)
            count = 1
            prev_char = char
        else:
            count += 1
    else:
        en += prev_char + str(count)
        return en

print(rle_encode('AAADDDFFFRFFFREEEE'))


############################################################
# Сгруппировать список в 1-4 например
def repr(group_start, group_end) -> str:
    # это просто правильно печатает группу
    
    if group_start == group_end:
        return str(group_end)

    return f'{group_start}-{group_end}'


def squeeze(numbers) -> str:
    if not numbers:  # граничный случай
        return ''

    numbers_ = sorted(numbers)  # сначала располагаем по порядку
    groups = []  # тут будем хранить группы

    last_group_start = None
    last_group_end = None

    for n in numbers_:
        # это первая итерация, просто говорим, что группа началась и закончилась
        if last_group_end is None:
            last_group_start = n
            last_group_end = n

        # если предыдущая группа отличается от текущего числа на 1, 
        # то это число входит в группу, то есть становится концом группы
        elif last_group_end == n - 1:
            last_group_end = n

        # иначе мы понимаем, что группа закончилась,
        # мы её запоминаем и начинаем новую
        else:
            groups.append(repr(last_group_start, last_group_end))
            last_group_start = n
            last_group_end = n

    else:
        # посленюю группу придётся обработать вручную
        groups.append(repr(last_group_start, last_group_end))

    return ','.join(groups)



##########################################################
# 0 и 1
def get_one_length(lst):
    one_length = 0
    lengths = []
    for elem in lst + [0]:
        if elem == 1:
            one_length += 1
        else:
            lengths.append(one_length)
            one_length = 0
    print(lengths)
    return lengths


def calc_max_one_length(lst):
    one_lengths = get_one_length(lst)
    if len(one_lengths) == 1:
        return one_lengths[0] - 1
    else:
        i = 0
        max_length = 0
        while i < len(one_lengths) - 1:
            if one_lengths[i] + one_lengths[i+1] > max_length:
                max_length = one_lengths[i] + one_lengths[i+1]
            i += 1
        return max_length

lst = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
print(str(lst) + " -> " + str(calc_max_one_length(lst)))



########################################################
# сумма трех равна числу
from bisect import bisect_left


def find_triplet_sum(a, x):
    a.sort()
    lo, hi = 0, len(a) - 1
    while lo + 1 < hi:
        i = bisect_left(a, x - a[lo] - a[hi], lo + 1, hi)  # binary search
        assert lo < i
        triplet_sum = a[lo] + a[i] + a[hi]
        if i == hi or triplet_sum < x:  # sum is too small
            lo += 1
        elif triplet_sum > x:  # sum is too big
            hi -= 1
        else:  # found
            assert lo < i < hi and triplet_sum == x
            return lo, i, hi
    raise ValueError("Can't find i,j,k such that: a[i] + a[j] + a[k] == x.")



##########################################################
# гостиница 
from collections import defaultdict

def max_num_guests(guests):

    res = 0

    # для каждого дня посчитаем, сколько приехало и сколько отъехало
    arriving = defaultdict(int)
    leaving = defaultdict(int)

    for guest in guests:  # O(n)
        arriving[guest[0]] += 1
        leaving[guest[1]] += 1

    current = 0
    # едем по дням в порядке увеличения, добавлем приехавших и убавляем уехавших,
    # считаем сколько стало
    for day in sorted(set(arriving.keys()).union(set(leaving.keys()))):  # O(n*log(n)) + O(n)
        current -= leaving[day]
        current += arriving[day]

        if current > res:
            res = current

    return res


a = [ (1, 2), (1, 3), (2, 4), (2, 3)]
print(max_num_guests(a))



########################################################
# сгруппировать 
# def group_words(words):

#     groups = defaultdict(list)

#     for word in words:  # O(n)
#         key = sorted(word)
#         groups[key].append(word)

#     return [sorted(words) for words in groups.values()] 


def group_words(words):
    result = []
    words_dict = {}
    for word in words:
        sorted_word = "".join(sorted(word))
        if not sorted_word in words_dict.keys():
            words_dict[sorted_word] = []
        words_dict[sorted_word].append(word)
    for key, value in words_dict.items():
        result.append(value)
    return result

a =  ["eat", "tea", "tan", "ate", "nat", "bat"]
print(group_words(a))



######################################################
# слияние отрезков
def merge(ranges):

    if not ranges:
        return []

    result_ranges = []
    last_range = None  # последний отрезок, что мы видели

    for rng in sorted(ranges):  # обязательно сортируем

        if not last_range:
            last_range = rng
            continue

        # если начало текущего отрезка меньше конца предыдущего
        if rng[0] <= last_range[1]:
            # расширяем предыдущий отрезок на текущий
            last_range = (last_range[0], max(rng[1], last_range[1]))

        # старый отрезок всё, начинаем новый
        else:
            result_ranges.append(last_range)
            last_range = rng

    else:
        # граничный случай для последнего элемента
        result_ranges.append(last_range)

    return result_ranges


a = [[1, 3], [100, 200], [2, 4], [101, 221]]
print(merge(a))


####################################################
# точки и симметрия


####################################################
# если из первой строки можно получить вторую, совершив не более 1 изменения 
def create_matrix(n, m):
    matrix = [[0] * m for i in range(n)]

    # заполняем 0 строку
    for j in range(m):
        matrix[0][j] = j

    # заполняем 0 столбец
    for i in range(n):
        matrix[i][0] = i

    return matrix

def damerau_levenshtein_recursive(str1, str2, out_put = False):
    n = len(str1)
    m = len(str2)

    if n == 0 or m == 0:
        if n != 0:
            return n
        if m != 0:
            return m
        return 0

    change = 0
    if str1[-1] != str2[-1]:
        change += 1

    if n > 1 and m > 1 and str1[-1] == str2[-2] \
        and str1[-2] == str2[-1]:
        min_ret = min(damerau_levenshtein_recursive(str1[:n - 1], str2) + 1,
                      damerau_levenshtein_recursive(str1, str2[:m - 1]) + 1,
                      damerau_levenshtein_recursive(str1[:n - 1], str2[:m - 1]) + change,
                      damerau_levenshtein_recursive(str1[:n - 2], str2[:m - 2]) + 1)
    else:
        min_ret = min(damerau_levenshtein_recursive(str1[:n - 1], str2) + 1,
                      damerau_levenshtein_recursive(str1, str2[:m - 1]) + 1,
                      damerau_levenshtein_recursive(str1[:n - 1], str2[:m - 1]) + change)
    return min_ret

print(damerau_levenshtein_recursive('qwerty', 'qwrety'))


#######################################################
# кол-во слов в файле
filename = 'test.txt'
def counter():
    with open(filename) as file:    #открываем через менеджер контекста, filename определим позже
        text = file.read()    #считываем содержимое
    text = text.replace("\n", " ")
    text = text.replace(",", "").replace(".", "").replace("?", "").replace("!", "").replace("—", "")
    text = text.lower()    #убираем верхний регистр
    words = text.split() 


    nonrep_words = list()
    for word in words:
        if word in nonrep_words:    #проверка, "есть ли данный элемент уже в списке?"
            pass    #если есть, то ничего не делаем
        else:
            nonrep_words.append(word) 

    return nonrep_words

print(counter())



#################################################################
from re import search
s1 = "catrew"
s2 = "ctr"

count = 0
for i in range(len(s1)):
    if count < len(s2) and s2[count] == s1[i]:
        count += 1

if count == len(s2):
    print("True")
else:
    print("False")

s = [1, 2, 3, 4, 5, 1, 2, 3, 4]
print(s.reduce(__xor__))






"""
Даны две строки строчных латинских символов: строка J и строка S. Символы, входящие в строку J,
— «драгоценности», входящие в строку S — «камни». Нужно определить, какое количество символов из S
одновременно являются «драгоценностями». Проще говоря, нужно проверить, какое количество символов из S
входит в J.
Это разминочная задача, к которой мы размещаем готовые решения. Она очень простая и нужна для того,
чтобы вы могли познакомиться с нашей автоматической системой проверки решений. Ввод и вывод осуществляется
через файлы, либо через стандартные потоки ввода-вывода, как вам удобнее."""
import sys

j = sys.stdin.readline().strip()
s = sys.stdin.readline().strip()

result = 0
for ch in s:
    if ch in j:
        result += 1

print(result)










"""
Требуется найти в бинарном векторе самую длинную последовательность единиц и вывести её длину.
Желательно получить решение, работающее за линейное время и при этом проходящее по входному
массиву только один раз.
"""
import sys

count = sys.stdin.readline().strip()

result = 0
max = 0
for i in range(int(count)):
    el = sys.stdin.readline().strip()
    if el == '1':
        result += 1
    else:
        if max < result:
            max = result
        result = 0

print(max if max > result else result)










"""
Дан упорядоченный по неубыванию массив целых 32-разрядных чисел.
Требуется удалить из него все повторения.
Желательно получить решение, которое не считывает входной файл целиком в память, т.е.,
использует лишь константный объем памяти в процессе работы.
"""
import sys

count = sys.stdin.readline().strip()

ch = -1
res = []
for i in range(int(count)):
    el = sys.stdin.readline().strip()
    if el != ch:
        res.append(el)
    ch = el

for i in res:
    print(i)








"""
Дано целое число n. Требуется вывести все правильные скобочные последовательности длины 2 ⋅ n,
упорядоченные лексикографически (см. https://ru.wikipedia.org/wiki/Лексикографический_порядок).
В задаче используются только круглые скобки.
Желательно получить решение, которое работает за время, пропорциональное общему количеству
правильных скобочныхпоследовательностей в ответе, и при этом использует объём памяти,
пропорциональный n.
"""
import sys
n = int(sys.stdin.readline().strip())


def foo(s, l, r, pairs):
    if l == pairs and r == pairs:
        print(s)
    else:
        if l < pairs:
            foo(s + '(', l + 1, r, pairs)
        if r < l:
            foo(s + ')', l, r + 1, pairs)


foo('', 0, 0, n)


АНАГРАММЫ
import sys
s1 = sys.stdin.readline().strip()
s2 = sys.stdin.readline().strip()

l1 = list(s1)
l1.sort()
l2 = list(s2)
l2.sort()

print(1 if l1 == l2 else 0)



import sys
s1 = sys.stdin.readline().strip()
s2 = sys.stdin.readline().strip()
l1 = list(s1)
l2 = list(s2)

while l1:
    li = l1.pop()
    if li in l2:
        l2.remove(li)
    else:
        print(0)
        sys.exit()
print(1 if not l2 else 0)



import collections
import sys
s1 = sys.stdin.readline().strip()
s2 = sys.stdin.readline().strip()
l1 = list(s1)
l2 = list(s2)

c1 = collections.Counter()
c2 = collections.Counter()

for word in l1:
    c1[word] += 1

for word in l2:
    c2[word] += 1

print(1 if c1 == c2 else 0)
""""""""""""""""""""""""""""""""""""""""""""""""



import sys

int_list = []
t = [0] * 101
n = int(sys.stdin.readline().strip())
for i in range(int(n)):
    s = sys.stdin.readline().strip()
    try:
        num = int(s[:s.find(' ')])
    except ValueError:
        continue
    for index, value in enumerate(s.split(' ')):
        if index == 0:
            continue
        elif index == num + 1:
            break
        try:
            t[int(value)] += 1
        except ValueError:
            pass
    del s

for i in range(101):
    if t[i]:
        print(' '.join([str(i)] * t[i]), end=' ')