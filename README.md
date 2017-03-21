# SorteKat
sortekat er en python modul for å <strong>sort</strong>ere norske nyhetstekster inn i <strong>kat</strong>egorier.
I første omgang er det kun en LinearSVC fra sklearn-pakken, trent på Norske nyhetstekster.

Kategoriene som brukes er:
* Sport
* Kultur
* Økonomi
* Politikk
* Krim
* Vitenskap
* Hverdagsliv
* Sosiale spørsmål
* Ulykker
* Vær

```python
['HVER', 'KRIM', 'KUL', 'OKO', 'POL', 'SOS', 'SPO', 'ULY', 'VEAR', 'VIT']
```
Disse representerer den inderste ringen i en sektorbasert inndeling av nyhetskategorier.
![Sektorbasert inndeling av nyhetskategorier](http://stavelin.com/uib/nyhetskategorier/sektor.png)



## Bruk
* Last ned stortekat og legg på din `PYTHONPATH`.
* Unzip saved_models.zip slik at du får en mappe (saved_models) med .pkl-filer

```python
from sortekat import SorteKat
# opprett et object (dette tar litt tid, modellene lastes fra disk)
clf = SorteKat()

texts = ["Statsminister Stoltenberg besøkte bedrifter på vestlandet.",
        "Brann vant kveldens kamp not Drammen.",
        "Arbeidsledighetstallene går ned viser nye tall fra SSB.",
        "Ny forskning viser at profesorer forsker mer."]

# Klassifiser én...
pred = clf.predict_one(texts[0])
print(pred, "==>", texts[0])
#Index(['POL'], dtype='object') ==> Statsminister Stoltenberg besøkte bedrifter på vestlandet.

# .. eller mange av gangen. Dette er raskt.
preds = clf.predict_many(texts)
for p, t in zip(preds, texts):
    print(p,"==>", t)

#POL ==> Statsminister Stoltenberg besøkte bedrifter på vestlandet.
#SPO ==> Brann vant kveldens kamp not Drammen.
#SOS ==> Arbeidsledighetstallene går ned viser nye tall fra SSB.
#VIT ==> Ny forskning viser at profesorer forsker mer.
```

## Install & Avhengigheter
* python3
* sklearn


## Data
sortekat er trent på data fra ulike forskninsprosjekter, blandt annet [Norsk Avis Korpus](http://avis.uib.no/), [NRKs nyhetstilbud på nett i 2009](http://www.medietilsynet.no/no/Nyheter/Nyhetsarkiv/Nyheter-2010/Juni-2010/Allmennkringkastingsordningen-star-fortsatt-sterkt1/) og ulike deler av [Helle Sjøvaags forskning](http://www.uib.no/personer/Helle.Sj%C3%B8vaag#uib-tabs-publikasjoner)

## Presisjon
Modellen er trent på 11943. Validering og søking etter hyperparametre er gjort med en normal 75/25 split.
Helt til sist har jeg en 95/5 split slik at jeg maksimerer datsettet som brukes til trening, men fortsatt har et lite knippe (5%) out-of-sample data som kan gi en liten pekepinn på den ferdige modellen. Under finner du denne, den samsvarer fint med 75/25-splitten. (ta denne med en klype salt, som du ser er det veldig få tekster pr kategori her).

####Accuracy:
Tested on *only* 5% of the dataset (629 text)
precision_score:        0.936446146838   
recall:                 0.936406995231
f1:                     0.936219534836   
f1_macro:               0.920418777686

             precision    recall  f1-score   support

       HVER       0.90      0.93      0.92        99
       KRIM       0.92      0.92      0.92        38
        KUL       0.96      0.96      0.96       113
        OKO       0.90      0.88      0.89        42
        POL       0.95      0.94      0.94       118
        SOS       0.90      0.92      0.91        48
        SPO       0.97      0.99      0.98       117
        ULY       0.89      0.83      0.86        30
       VEAR       0.94      0.89      0.91        18
        VIT       1.00      0.83      0.91         6

avg / total       0.94      0.94      0.94       629




### Kontakt
- [Eirik Stavelin](http://www.uib.no/personer/Eirik.Stavelin)
