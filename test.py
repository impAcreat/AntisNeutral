def get_map_index_for_sub_arr(sub_arr, raw_arr):
    # 初始化 map_arr，与 raw_arr 形状相同，初始值为 -1
    map_arr = np.zeros(raw_arr.shape)
    map_arr.fill(-1)

    # 遍历 sub_arr，记录每个元素在 sub_arr 中的索引
    for idx in range(sub_arr.shape[0]):
        map_arr[sub_arr[idx]] = idx

    return map_arr


import numpy as np

raw_arr = np.array([10, 20, 30, 40, 50])
sub_arr = np.array([30, 50])

map_arr = get_map_index_for_sub_arr(sub_arr, raw_arr)
print(map_arr)
