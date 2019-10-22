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


lst = [43, 12, 53, 39, 1, 2, 2, 103, 72]
if False:
    size = int(input("Enter size of the list: "))
    for i in range(size):
        elements = int(input("Enter an element"))
        lst.append(elements)
low = 0
high = len(lst) - 1
k_quick_sort(lst, low, high, k_get)
print(lst)
