import os


def main():
    base_dir = "/home/lichengzhi/CV_ToolBox/data/2019.8.1/Annotations"
    for r, dirs, files in os.walk(base_dir):
        for file in files:
            with open(os.path.join(r, file), "r+") as f:
                content = f.read()
                f.seek(0, 0)
                f.write('<?xml version="1.0"?>\n' + content)


if __name__ == "__main__":
    main()
