import rarfile

if __name__ == "__main__":
    path = "./TNO.rar"

    path2 = "./TNO"

    rf = rarfile.RarFile(path)

    rf.extractall(path2)