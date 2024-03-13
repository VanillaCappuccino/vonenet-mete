import os, sys

if not os.path.isdir("testing"):
    os.mkdir("testing")
else:
    print("Directory already created by Docker")

with open("testing/hello.txt", "w") as f:
    f.write("y u no work?")

f.close()