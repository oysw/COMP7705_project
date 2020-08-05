# How to use this code

First of all, in order to run the scripts successfully, run

~~~bash
pip install -r requirements.txt
~~~

Then edit the dataloader.py file.

~~~python
if __name__ == "__main__":
    generate("result", 400, 1)
~~~

Note that, the first argument is the directory name where the generated data will be saved.

The second argument is the total number of labels you want to generate.

The last parameter is according to the computer. It means the number of sub-tasks that the whole task will be splited into. For example, if the second argument is 400 and the third argument is 4, the script will generated 100 labels each time and append the 100 labels to the file on the disk.