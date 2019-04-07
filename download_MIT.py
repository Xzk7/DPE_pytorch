from urllib.request import urlretrieve
import os
import requests

file1 = open('/home/user300/XZK/DPE_pytorch/MIT_datasets/train_input.txt', 'r')
file2 = open('/home/user300/XZK/DPE_pytorch/MIT_datasets/train_label.txt', 'r')
file3 = open('/home/user300/XZK/DPE_pytorch/MIT_datasets/download1.txt', 'r')
file4 = open('/home/user300/XZK/DPE_pytorch/MIT_datasets/download2.txt', 'r')
A_name_list = file1.readlines()
C_name_list = file2.readlines()
name_list1 = file3.readlines()
name_list2 = file4.readlines()
url1 = 'https://data.csail.mit.edu/graphics/fivek/img/dng/'
url2 = 'https://data.csail.mit.edu/graphics/fivek/img/tiff16_c/'
path1 = '/home/user300/XZK/DPE_pytorch/MIT_datasets/A'
path2 = '/home/user300/XZK/DPE_pytorch/MIT_datasets/C'
name1 = []
for items in name_list1:
    name1.append(items[0:5])
name2 = []
for items in name_list2:
    name2.append(items[0:5])

'''for items in A_name_list:
    if(items[0:5] in name1):
        index = name1.index(items[0:5])
        length = len(name_list1[index])
        url = url1+name_list1[index][:length-1]+'.dng'
        urlretrieve(url, os.path.join(path1, items[0:5]+'.dng'))
        print(url)
    elif (items[0:5] in name2):
        index = name2.index(items[0:5])
        length = len(name_list2[index])
        url = url1 + name_list2[index][:length - 1] + '.dng'
        urlretrieve(url, os.path.join(path1, items[0:5] + '.dng'))
        print(url)
    else:
        pass

    print("image "+items[0:5]+".dng downloaded")'''

for items in C_name_list:
    if(items[0:5] in name1):
        index = name1.index(items[0:5])
        length = len(name_list1[index])
        url = url2+name_list1[index][:length-1]+'.tif'
        urlretrieve(url, os.path.join(path2, items[0:5]+'.tif'))
    elif (items[0:5] in name2):
        index = name2.index(items[0:5])
        length = len(name_list2[index])
        url = url2 + name_list2[index][:length - 1] + '.tif'
        urlretrieve(url, os.path.join(path2, items[0:5] + '.tif'))
    else:
        pass

    print("image " + items[0:5] + ".tif downloaded")

'''url_list = file1.readlines()
name_list = file2.readlines()
root = '.\MIT_datasets'
for index in range(563,len(url_list)):
    urlretrieve(url_list[index], os.path.join(root, name_list[index][0:len(name_list[index])-1])+'.jpg')'''
