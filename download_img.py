from urllib.request import urlretrieve
import os
import requests

file1 = open('/home/user300/XZK/DPE_pytorch/HDR_datasets/hdr_url.txt', 'r')
file2 = open('/home/user300/XZK/DPE_pytorch/HDR_datasets/hdr_new.txt', 'r')
url_list = file1.readlines()
name_list = file2.readlines()
root = '.\HDR_datasets'
for index in range(563,len(url_list)):
    urlretrieve(url_list[index], os.path.join(root, name_list[index][0:len(name_list[index])-1])+'.jpg')
