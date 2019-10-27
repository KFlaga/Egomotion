import re
import os.path as path
import os
from shutil import copyfile


image_file_format = "(?P<number>\d+)\.jpg"


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

    return images


def create_directory(path):
    try:
        os.makedirs(path)
    except os.error:
        pass
    return path


def create_camera_file(camera, path):
    with open(path, 'w') as f:
        f.write(camera)


def read_cameras(path):
    content = []
    with open(path, 'r') as f:
        content = f.readlines()

    cameras = {}
    line_count = len(content)
    for i in range(int(line_count / 5)):
        header = content[i * 5]
        frame_number = int(header.split()[2])
        cameras[frame_number] = "".join(content[k] for k in range(i * 5 + 1, i * 5 + 5))
    return cameras


def main():
    dirs = ["office", "office2", "livingroom", "livingroom2"]
    abs_dirs = [path.abspath(x) for x in dirs]

    for dir in abs_dirs:
        print "Reading dataset ", dir
        images = get_files(dir)
        cameras = read_cameras(path.join(dir, "camera.txt"))
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
            copyfile(path.join(dir, frame[1]), path.join(images_dir, str(i) + ".jpg"))
            create_camera_file(frame[2], path.join(cameras_dir, str(i) + ".txt"))
            i = i + 1


if __name__ == "__main__":
    main()
