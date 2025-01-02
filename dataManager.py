import os

list = os.listdir('./labeled/G9')
count = 0

for img in list:
    #for ch in img:
     #   print(ch)
    print(img[:2])
    
    if img[:2] == 'G9':
        count += 1

print(count)

print(list)