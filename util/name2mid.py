import openpyxl
from mmdet.datasets.shell import ShellDataset

source_dir = "Name2MID.xlsx"


def main():
    mmap = dict()
    source_wb = openpyxl.load_workbook(source_dir, read_only=True)
    sheet = source_wb["Sheet1"]
    for row in sheet.rows:
        name = str(row[0].value)
        chn = str(row[3].value)
        if chn is not 'None':
            mmap[name] = chn
        else:
            mmap[name] = "Unknown"
    for name in ShellDataset.CLASSES:
        # print("%s: %s" % (name, mmap[name]))
        print("\'%s\'," % mmap[name])


if __name__ == "__main__":
    main()
