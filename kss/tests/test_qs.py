def k_qs(arr, fn):
    # helper
    def k_hqs(items, low, high):
        # split around pivot
        split = partition(items, low, high, fn)
        items = k_hqs(items, low, split)
        items = k_hqs(items, split + 1, high)
        return items

    return k_hqs(arr, 0, len(arr) - 1)


# swapping function
def partition(nums, low, high, fn):
    pivot = nums[(low + high) // 2]
    i = low - 1
    j = high + 1
    while True and 0 < i < len(nums) - 1 and 0 < j < len(nums) - 1:
        i += 1
        while fn(nums[i]) < fn(pivot):
            i += 1

            j -= 1
            while fn(nums[j]) > fn(pivot):
                j -= 1

            if i >= j:
                return j


def get_v(e):
    return e


num_list = [1, 86, 1, 12, 32, 45, 2, 85, 3, 19, 35, 87, 4, 8]
print('before: ' + str(num_list))
num_list = k_qs(num_list, get_v)
print('after: ' + str(num_list))
