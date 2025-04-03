import requests as r
import pyperclip

url = "http://github.com/mediasample/action/raw/main/graph2.py"
res =r.get(url,allow_redirects=True)
#pyperclip.copy(res.text)
data=res.text
print(data)



