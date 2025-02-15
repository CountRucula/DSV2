# Digitale Signal Verarbeitung 2 (DSV2)
Diese Repository enthält meine Source Code zu den Praktikas, Übungen und SEP's für das Modul DSV2.

## Content
Das Repo is in drei Bereiche unterteilt:
  - [Praktika](#praktika)
  - [Übungen](#übungen)
  - [Prüfungen](#prüfungen)


### Praktika
Die Lösung der Praktikas sind entweder mit python script oder mit eine Jupyter-Notebook gemacht worden.
Alle nötigen Dateien (Anleitung, Daten, ...) befinden sich jeweils in den entsprechenden Unterördner.

|  Nr. | Praktikum                                                                                                                                                               |
| ---: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|    1 | [Digitale Signalverarbeitung mit Python](https://github.com/CountRucula/DSV2/tree/main/Praktika/Praktikum%2001/DSV2_Lab1.ipynb)                                         |
|    2 | [Distanzmessung mit Pseudo-Noise](https://github.com/CountRucula/DSV2/tree/main/Praktika/Praktikum%2002/distanzmessung.py)                                              |
|    3 | [Raumidenfikation](https://github.com/CountRucula/DSV2/tree/main/Praktika/Praktikum%2003/impulse_response_raum.py)                                                      |
|    4 | [LMS Algorimthus](https://github.com/CountRucula/DSV2/tree/main/Praktika/Praktikum%2004/sysId.py)                                                                       |
|    5 | [Active Noise Cancelling](https://github.com/CountRucula/DSV2/tree/main/Praktika/Praktikum%2005/anc.py)                                                                 |
|    6 | [RLS Algorimthus](https://github.com/CountRucula/DSV2/tree/main/Praktika/Praktikum%2006/anc_rls_template.py)                                                            |
|    7 | [Höhenmessung und Sensorfusion](https://github.com/CountRucula/DSV2/tree/main/Praktika/Praktikum%2007/praktikum_07.ipynb)                                               |
|    8 | [Kalman Filter und Sensorfusion](https://github.com/CountRucula/DSV2/tree/main/Praktika/Praktikum%2008/praktikum_08.ipynb)                                              |
|    9 | [Nichtlineare Kalman Filter: Tracking eines Roboters mit Differentialantrieb](https://github.com/CountRucula/DSV2/tree/main/Praktika/Praktikum%2009/praktikum_09.ipynb) |
|   10 | [Linear Logistic Regression](https://github.com/CountRucula/DSV2/tree/main/Praktika/Praktikum%2010/praktikum_10.ipynb)                                                  |
|   11 | [Deep Neural Networks](https://github.com/CountRucula/DSV2/tree/main/Praktika/Praktikum%2011/train_two_layer_network_template.py)                                       |
|   13 | [CNN Keyword Detection](https://github.com/CountRucula/DSV2/tree/main/Praktika/Praktikum%2013/train_network.m)                                                          |

### Übungen
- [Übung 01](https://github.com/CountRucula/DSV2/tree/main/Übungen/Übung%2001.ipynb)
- [Übung 04](https://github.com/CountRucula/DSV2/tree/main/Übungen/Übung%2004.ipynb)
- [Übung 05](https://github.com/CountRucula/DSV2/tree/main/Übungen/Übung%2005.ipynb)
- [Übung 06](https://github.com/CountRucula/DSV2/tree/main/Übungen/Übung%2006.ipynb)
- [Übung 07](https://github.com/CountRucula/DSV2/tree/main/Übungen/Übung%2007.ipynb)
- [Zwischenprüfung FS2021](https://github.com/CountRucula/DSV2/tree/main/Übungen/Zwischenprüfung%20FS2021.py)
- [SEP FS2019](https://github.com/CountRucula/DSV2/tree/main/Übungen/SEP%20FS2019.ipynb)
- [SEP FS2021](https://github.com/CountRucula/DSV2/tree/main/Übungen/SEP%20FS2021.ipynb)

### Prüfungen
- [Zwischenprüfung](https://github.com/CountRucula/DSV2/tree/main/Übungen/Zwischenprüfung.py)
- [SEP](https://github.com/CountRucula/DSV2/tree/main/SEP/SEP.ipynb)


## Requirements
Für die Ausführung des Praktika und Übungs code, wird Python 3.X mit conda benötigt. 
Für das Praktikum 13 wird zusätzlich Matlab benötigt. Evtl. funktioniert es auch mit Octave.

Python modules Installation via conda:
```sh
conda env create -f environment.yml
```