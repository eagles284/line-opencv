from googletrans import Translator

trans = Translator()
text = trans.translate('Saya adalah budak anda')
print(text.text)