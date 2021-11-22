# Sequence competition

Each team **X** should submit their own **leikmadur_teymi_X.py** class file, see example of *Random* player and *SelfPlay100* player within leikamdur_teymi_7.py
I assume all the binary files you need to load will be in the folder **./teymiX/** 

To try out you player you can use the compete.py file, you will need to add the lines to the code

<code>
from leikmadur_teymi_X import * 
</code>

<p>
Then add to the bottom of the file:
</p>

<p>
  
model_1 = []  
model_1.append(leikmadur_teymi_Xa(1))  
model_1.append(leikmadur_teymi_Xb(1))  
  
model_2 = []  
model_2.append(leikmadur_teymi_Xa(2))  
model_2.append(leikmadur_teymi_Xb(2))  
</p>

Here I assume you have two classes in your file **leikmadur_teymi_Xa** and **leikmadur_teymi_Xb**

You can then just run the code as follows:

<code>
python3 compete.py
</code>
