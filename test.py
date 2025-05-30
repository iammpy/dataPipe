import os


if "__file__" in globals():
    os.chdir(os.path.join(os.path.dirname(__file__)))
print("Current working directory:", os.getcwd())
print("Absolute path of current directory:", os.path.abspath('.'))