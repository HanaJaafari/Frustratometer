import dca_frustratometer
print(dir)
print("Module Structure\n")
for submodule in dir(dca_frustratometer):
    #print(dca_frustratometer.__dict__)
    if "__" not in submodule:
        m=dca_frustratometer.__dict__[submodule]
        print(f'{submodule}: {" ".join(a for a in dir(m) if "__" not in a)}')