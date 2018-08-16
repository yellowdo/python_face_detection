###### Anaconda Download

https://repo.continuum.io/archive/

###### Andaconda 설치후 업데이트

```
conda update -n base conda
conda update conda
conda update anaconda
```

###### python update (python 리스트 확인 후 원하는 버전 설치)

```
conda search python
conda install python=3.6.5
```

###### pip 다운로드 서버

- %APPDATA%\pip\pip.ini 경로 파일 생성후 아래 내용 기입

```
[global]
index-url=http://ftp.daumkakao.com/pypi/simple
trusted-host=ftp.daumkakao.com
;proxy 설정
;proxy=http://xxx.xxx.xxx.xxx:xxxx
```

###### pip command line 이용할 경우

`pip install XXXXX -i http://ftp.daumkakao.com/pypi/simple --trusted-host ftp.daumkakao.com`

###### env create

`conda create --name py36 python=3.6`

###### env remove

`conda env remove --name py36`

###### env activate

`activate env_face`

###### env 설정

```
conda install tensorflow opencv
pip install imutils playsound cython mkl cmake
pip install numpy scipy matplotlib scikit-learn jupyter
pip install opencv-contrib-python
pip install dlib
pip install pyserial cx-oracle pyqt5 pyqt5-tools
```

###### dlib 강제 설치

- 버전확인 https://anaconda.org/conda-forge/dlib/files

`conda install -c conda-forge dlib=19.10`

- Download https://pypi.org/project/dlib/#files