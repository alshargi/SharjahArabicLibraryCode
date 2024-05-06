# SharjahArabicLibraryCode
sharjah library codes



transformers-cli repo download alshargi/librarycode

git clone https://huggingface.co/Alshargi/libraryCodes


  #SharjahArabicLibraryCode


#### install the library:
```bash
pip install git+https://github.com/alshargi/SharjahArabicLibraryCode.git
```
#### Demo of some of the features:
```python
from SharjahArabicLibraryCode import get_pred_label
xx = ['ليبراليه شويه صعاليك ',
       ' الاسم شاهين والشكل والتفكير حمار',
      'يلعن امك يا ابن الحمار']
MyResult = get_pred_label(xx)
for i in MyResult:
    subRes =  i.split("\t")
    for s in subRes:
        print(s)
    print("########")

```

#### Result
```bash



```





