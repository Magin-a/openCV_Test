import os
import glob
#폴더에 들어있는 모든 사진 이름을 리스트로 반환(경로까지 모두 포함한 상태)
path_jpg = glob.glob("./Desktop/test/*.jpg")

#모든 사진의 이름을 TXT파일에 저장
with open("./Desktop/new_name.txt", 'w') as f:
    lines = path_jpg
    for line in lines:
        f.write(line+"\n")

#TXT에 있는 경로의 일부분을 삭제해주는 과정
with open("./Desktop/new_name.txt", 'r') as f:
    lines = f.readlines()

with open("./Desktop/new_name.txt", 'w') as f:
    for line in lines:
        f.write(line.replace("./Desktop/",''))
    f.close()

print("finish")


