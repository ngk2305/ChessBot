arr = ['a', 'b', 'c', 'd', 'e']
brr = ['l','m','n','o','p']

def putToFront(array,element,now):
    array.remove(element)
    index_now = arr.index(now)+1
    array.insert(index_now, element)
    return array

print(putToFront(arr,'e','c'))




# Iterate through the array backwards, taking only the odd positions
for i in range(len(arr) - 2, -1, -2):  # Start from the last element, step backwards by 2
    print(arr[i])
