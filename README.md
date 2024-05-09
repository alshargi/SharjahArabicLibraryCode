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


text = "تقافة الفكر عند ارسطو وعظماء الامة"

result = findCongCode(text, model)
for i in result:
    print(i)


```


#### Result
```bash
text: تقافة الفكر عند ارسطو وعظماء الامة
Top Result:	B  PHILOSOPHY. PSYCHOLOGY. RELIGION  N#: 1
Result:	BP  ISLAM. BAHAISM. THEOSOPHY, ETC.  N#: 16
% 99.7997  BP  ISLAM. BAHAISM. THEOSOPHY, ETC.
% 0.1015  B1  Philosophy (General)—Periodicals. Serials—English and American
% 0.0356  BF  PSYCHOLOGY
% 0.0277  B8  Philosophy (General)—Periodicals. Serials—Other. By language, A-Z
```



```python
text = "تأديب الموظف وفقا لأحكام القانون الأساسي العام للوظيفة العمومية"
result = findCongCode(text, model)
for i in result:
    print(i)
    
```
#### Result
```bash
Text:  تأديب الموظف وفقا لأحكام القانون الأساسي العام للوظيفة العمومية
Top: K  LAW  N#: 9
Sub: KM  Middle East.  Southwest Asia  N#: 15
Sub_sub: KRM  Islamic Law  N#: 29

Sub_sub Predictions:
% 55.7075  KRM  Islamic Law
% 35.5449  KMV  Islamic Sects and Movements
% 5.5331  KMM  Islamic Moral and Ethical Thought
% 1.5389  KMQ  Islamic Ritual and Worship

Top Predictions:
% 99.7311  K  LAW
% 0.0649  A  GENERAL WORKS
% 0.045  B  PHILOSOPHY. PSYCHOLOGY. RELIGION
% 0.0339  Q  SCIENCE

Sub Predictions:
% 55.9922  KM  Middle East.  Southwest Asia
% 31.98  KR  Law of Scotland Dedicated to legal sources pertaining to the laws of Scotland.
% 7.9953  K3  Law in general. Comparative and uniform law. Jurisprudence—Periodicals—By main entry—C
% 1.201  KS  Law of Hispanic Countries Laws and legal guides related to Spanish-speaking countries in the Americas.
```





