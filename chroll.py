import os, re
import shutil
import random

"""shutil_dir = 'C:\\Users\\HP\\Desktop\\garbage_classification\\trash'
original_dir = 'C:\\Users\\HP\\PycharmProjects\pythonProject\\Garbage classification\\trash'
"""

original_dir = 'C:\\Users\\HP\\PycharmProjects\pythonProject\\Garbage classification\\'
test_dir = 'C:\\Users\\HP\\PycharmProjects\pythonProject\\test'
train_dir = 'C:\\Users\\HP\\PycharmProjects\pythonProject\\train'
val_dir = 'C:\\Users\\HP\\PycharmProjects\pythonProject\\val'
i = 0

"""p = re.compile('[a-zA-Z]+')
num = str(i+len(os.listdir(original_dir)))
"""
for folder in os.listdir(original_dir):


    file_dir = os.path.join(original_dir, folder)
    num = len(os.listdir(file_dir))
    if os.path.isdir(os.path.join(test_dir, folder)) == 0:
        os.mkdir(os.path.join(test_dir, folder))

    if os.path.isdir(os.path.join(train_dir, folder)) == 0:
        os.mkdir(os.path.join(train_dir, folder))

    if os.path.isdir(os.path.join(val_dir, folder)) == 0:
        os.mkdir(os.path.join(val_dir, folder))


    for a in random.sample(os.listdir(file_dir),len(os.listdir(file_dir))):
        src = os.path.join(file_dir, a)
        dst = os.path.join(train_dir, os.path.join(folder, a))
        shutil.move(src, dst)
        if i == int(num * 0.6):
            break
        i = i + 1
    i = 0

    for a in random.sample(os.listdir(file_dir),len(os.listdir(file_dir))):
        src = os.path.join(file_dir, a)
        dst = os.path.join(test_dir, os.path.join(folder, a))
        shutil.move(src, dst)
        if i == int(num * 0.1):
            break
        i = i + 1
    i = 0

    for a in random.sample(os.listdir(file_dir),len(os.listdir(file_dir))):
        src = os.path.join(file_dir, a)
        dst = os.path.join(val_dir, os.path.join(folder, a))
        shutil.move(src, dst)
        if i == int(num * 0.3):
            break
        i = i + 1
    i = 0
# test


"""for i in os.listdir('.\\Garbage classification'):
    for a in os.listdir('.\\Garbage classification\\{}'.format(i)):
        print(a)
        src = os.path.join(os.listdir('.\\Garbage classification\\{}'.format(i)), a)
        dst = os.path.join(os.listdir('.\\Garbage classification\\{}'.format(i)), )
"""
