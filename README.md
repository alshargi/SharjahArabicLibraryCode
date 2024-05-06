# SharjahArabicLibraryCode
sharjah library codes



transformers-cli repo download alshargi/librarycode

git clone https://huggingface.co/Alshargi/libraryCodes


  #SharjahArabicLibraryCode


#### install the library:
```bash
!pip install git+https://github.com/alshargi/SharjahArabicLibraryCode.git

```
#### Demo of some of the features:
```python

from SharjahArabicLibraryCode import findCongCode 
import joblib

def loadMosels():
    all_models = joblib.load('SharjahLibraryOfCongressModel.pkl')
    print("Models loaded")
    return all_models

    
    
sharjamodel = loadMosels()
text = "تاريخ طب الأطفال عند العرب	الطب عند العرب طب الأطفال صحة الأطفال أمراض الأطفال"



result = findCongCode(text, sharjamodel)
for i in result:
    print(i)
    

    

```

#### Result
```bash



```





