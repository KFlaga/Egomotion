import re
import os.path as path
import os
from shutil import copyfile


image_file_format = "frame-(?P<number>\d+)\.color\.png"
camera_file_format = "frame-(?P<number>\d+)\.pose\.txt"


def get_files(directory):
    images = {}
    cameras = {}
    files = os.listdir(directory)
    for file in files:
        file_name = path.basename(file)
        match = re.match(image_file_format, file_name)
        if match:
            number = int(match.group('number'))
            images[number] = file
        else:
            match = re.match(camera_file_format, file_name)
            if match:
                number = int(match.group('number'))
                cameras[number] = file
    
    return (images, cameras)


def create_directory(path):
    try:
        os.makedirs(path)
    except os.error:
        pass
    return path


def main():
    dirs = ["kitchen2study"]
    abs_dirs = [path.abspath(x) for x in dirs]

    for dir in abs_dirs:
        print "Reading dataset ", dir
        images, cameras = get_files(dir)
        frames = []
        for number in images:
            if number in cameras:
                frames.append( (number, images[number], cameras[number]) )
        frames.sort(key = lambda f: f[0])
        
        out_dir = "common_" + path.basename(dir)
        images_dir = create_directory(path.join(out_dir, "images"))
        cameras_dir = create_directory(path.join(out_dir, "cameras"))

        print "Saving dataset ", out_dir
        i = 0;
        for frame in frames:
            copyfile(path.join(dir, frame[1]), path.join(images_dir, str(i) + ".png"))
            copyfile(path.join(dir, frame[2]), path.join(cameras_dir, str(i) + ".txt"))
            i = i + 1


if __name__ == "__main__":
    main()
