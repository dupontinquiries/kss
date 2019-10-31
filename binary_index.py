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


def kBinaryIndex(l, cutoff, fn):
    l = []
    
    return False


lst = []

start = datetime.now()

for i in range(1000000):
    lst.append(random.randint(0, 1000000))

end = datetime.now()
print('time to generate: {0}'.format(end - start))

if False:
    size = int(input("Enter size of the list: "))
    for i in range(size):
        elements = int(input("Enter an element"))
        lst.append(elements)

start = datetime.now()

low = 0
high = len(lst) - 1
k_quick_sort(lst, low, high, k_get)

end = datetime.now()
print('time to sort: {0}'.format(end - start))

print('\n\nSearching...\n\n')

cutoff = 9000

start = datetime.now()

kBI = kBinaryIndex(lst, cutoff, k_get)

end = datetime.now()
print('time to binary index: {0}'.format(end - start))

print(kBI)

print('Original list length: {0}\nNew list length: {1}'.format(len(lst), len(kBI)))
