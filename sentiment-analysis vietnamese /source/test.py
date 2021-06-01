from pyvi import ViTokenizer
text = "Dai hoc bach Viet nam"
text = ViTokenizer.tokenize(text)
text = ViTokenizer.tokenize(text)
texts = text.split()
texts = [t.replace('_', ' ') for t in texts]
print(texts)
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = set(stopwords.words('vietnam'))
print(stop)