import random
from datetime import datetime

def k_partition(sort_list, low, high, fn):
    i = (low - 1)
    pivot = sort_list[high]
    for j in range(low, high):
        if fn(sort_list[j]) <= fn(pivot):
            i += 1
            sort_list[i], sort_list[j] = sort_list[j], sort_list[i]
    sort_list[i + 1], sort_list[high] = sort_list[high], sort_list[i + 1]
    return (i + 1)


def k_quick_sort(sort_list, low, high, fn):
    if low < high:
        pi = k_partition(sort_list, low, high, fn)
        k_quick_sort(sort_list, low, pi - 1, fn)
        k_quick_sort(sort_list, pi + 1, high, fn)


def k_get(e):
    return e


lst = []

start = datetime.now()

for i in range(1000000):
    lst.append(random.randint(0, 1000000))

lst_reg = lst

end = datetime.now()
print('time to generate: {0}'.format(end - start))

if False:
    size = int(input("Enter size of the list: "))
    for i in range(size):
        elements = int(input("Enter an element"))
        lst.append(elements)

#quick sort
start = datetime.now()

low = 0
high = len(lst) - 1
k_quick_sort(lst, low, high, k_get)

end = datetime.now()
print('time to quick sort: {0}'.format(end - start))

#reg sort
start = datetime.now()

end = datetime.now()
print('time to regular sort: {0}'.format(end - start))

#print(lst)
