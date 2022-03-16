# Бинарная вставка

def insertion_binary(data):
	for i in range(len(data)):
		k = False
		key = data[i]
		lo, hi = 0, i - 1
		while lo < hi:
			mid = lo + (hi - lo) // 2
			if key < data[mid]:
				hi = mid
			else:
				lo = mid + 1		
		for j in range(i, lo , -1):
			data[j] = data[j - 1]; k = True
		if k: data[lo] = key
	return data

# бабл с флагом
def bubble_sort(nums):  
    # Устанавливаем swapped в True, чтобы цикл запустился хотя бы один раз
    swapped = True
    while swapped:
        print('jjj',nums)
        swapped = False
        for i in range(len(nums) - 1):
            print('uuu',nums)
            if nums[i] > nums[i + 1]:
                # Меняем элементы
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
                # Устанавливаем swapped в True для следующей итерации
                swapped = True

b = [1, 2, 3, 4, 6, 9, 8, 0, -8]
print(insertion_binary(b))


# быстрая
def partition(nums, low, high):  
    # Выбираем средний элемент в качестве опорного
    # Также возможен выбор первого, последнего
    # или произвольного элементов в качестве опорного
    pivot = nums[(low + high) // 2]
    i = low - 1
    j = high + 1
    while True:
        i += 1
        while nums[i] < pivot:
            i += 1

        j -= 1
        while nums[j] > pivot:
            j -= 1

        if i >= j:
            return j

        # Если элемент с индексом i (слева от опорного) больше, чем
        # элемент с индексом j (справа от опорного), меняем их местами
        nums[i], nums[j] = nums[j], nums[i]

def quick_sort(nums):  
    # Создадим вспомогательную функцию, которая вызывается рекурсивно
    def _quick_sort(items, low, high):
        if low < high:
            # This is the index after the pivot, where our lists are split
            split_index = partition(items, low, high)
            _quick_sort(items, low, split_index)
            _quick_sort(items, split_index + 1, high)

    _quick_sort(nums, 0, len(nums) - 1)


# Шелла
def shell(mass):
    len_mass = len(mass)

    t = len_mass // 2
    while t > 0:
        for i in range(len_mass - t):
            j = i
            while j > -1 and mass[j] > mass[j + t]:
                mass[j], mass[j + t] = mass[j + t], mass[j]
                j -= 1
        t //= 2
    return mass


# вставками
def insertion_sort(nums):  
    # Сортировку начинаем со второго элемента, т.к. считается, что первый элемент уже отсортирован
    for i in range(1, len(nums)):
        item_to_insert = nums[i]
        # Сохраняем ссылку на индекс предыдущего элемента
        j = i - 1
        # Элементы отсортированного сегмента перемещаем вперёд, если они больше
        # элемента для вставки
        while j >= 0 and nums[j] > item_to_insert:
            nums[j + 1] = nums[j]
            j -= 1
        # Вставляем элемент
        nums[j + 1] = item_to_insert