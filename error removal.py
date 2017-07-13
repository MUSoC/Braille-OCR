from textblob import TextBlob
from textblob import Word

f=open ("MUSOC.txt","a+")
with open ("MUSOC.txt","r") as fp:
    content=fp.read()
blob=TextBlob(content)
p=(blob.correct())
open("MUSOC.txt","w+")
f.write(str(p))
f.close()


