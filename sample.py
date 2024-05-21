import frustratometer
print(dir)
print("Module Structure\n")
for submodule in dir(frustratometer):
    #print(dca_frustratometer.__dict__)
    if "__" not in submodule:
        m=frustratometer.__dict__[submodule]
        print(f'{submodule}: {" ".join(a for a in dir(m) if "__" not in a)}')

if __name__ == "__main__":
    pdb_file = frustratometer._path.parent/'tests'/'data'/'1r69.pdb'
    pdb_structure = frustratometer.Structure.full_pdb(str(pdb_file))
    awsem = frustratometer.AWSEM(pdb_structure)
    print(awsem.configurational_frustration())