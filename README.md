# SharjahArabicLibraryCode
sharjah library codes


#### install the library:
```bash
Step # 1: install SharjahArabicLibraryCode the python library
!pip install git+https://github.com/alshargi/SharjahArabicLibraryCode.git

Step # 2: clone the model
!git lfs install
!git clone https://huggingface.co/Alshargi/libraryCodes

```
#### Demo of some of the features:
```python
from SharjahArabicLibraryCode import findCongCode 
import joblib

def loadMosels(mpath):
    all_models = joblib.load(mpath)
    print("Models loaded")
    return all_models

    
    
# Load the model:
model = loadMosels('/content/libraryCodes/libraryCodes.pkl')

text = "جراحة المسالك البولية"
text = "تقافة الفكر عند ارسطو وعظماء الامة"


result = findCongCode(text, model)
for i in result:
    print(i)
    

```

#### Result
```bash


Top Result:	B  PHILOSOPHY. PSYCHOLOGY. RELIGION  N#: 1
Result:	BP  ISLAM. BAHAISM. THEOSOPHY, ETC.  N#: 16



```





