import os

file1 = open('C:/Users/win8/Desktop/Xzk/工程/Photo_Enhancement/DPE_code_tensorflow/HDRs from Flickr/hdr.txt', 'r')
file2 = open('C:/Users/win8/Desktop/Xzk/工程/Photo_Enhancement/DPE_code_tensorflow/HDRs from Flickr/hdr_id.txt', 'r')
file3 = open('C:/Users/win8/Desktop/Xzk/工程/Photo_Enhancement/DPE_code_tensorflow/HDRs from Flickr/hdr_del.txt', 'w')
lines1 = file1.readlines()
lines2 = file2.readlines()
print(len(lines1))
'''print(str.find(lines2[1], '_'))
for index in range(len(lines1)):
    while not (lines2[index][str.find(lines2[index], '_')+1 : len(lines2[index])-1] in lines1[index]):
        del lines2[index]

with open('C:/Users/win8/Desktop/Xzk/工程/Photo_Enhancement/DPE_code_tensorflow/HDRs from Flickr/hdr_new.txt', 'w') as file3:
    for line in lines2:
        file3.write(line)

    file3.close()

with open('C:/Users/win8/Desktop/Xzk/工程/Photo_Enhancement/DPE_code_tensorflow/HDRs from Flickr/hdr_url.txt', 'w') as file4:
    for line in lines1:
        file4.write(line)

    file4.close()

with open('C:/Users/win8/Desktop/Xzk/工程/Photo_Enhancement/DPE_code_tensorflow/HDRs from Flickr/hdr_new.txt', 'r') as file5:
    lines5 = file5.readlines()
    file4 = open('C:/Users/win8/Desktop/Xzk/工程/Photo_Enhancement/DPE_code_tensorflow/HDRs from Flickr/hdr_url.txt', 'r')
    lines4 = file4.readlines()
    for index in range(len(lines4)):
        print(lines5[index][lines5[index].find('_')+1: len(lines5[index])-1] in lines4[index])

    file4.close()
    file5.close()'''
