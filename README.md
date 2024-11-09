# COFFE_PREDICT

## 1. REQUIREMENT
1. **Python 3.12+**
1. **Poetry**
1. **Edge browser**


## 2. Installation
### 2.1 Install pipx
[pipx Document](https://pipx.pypa.io/stable/installation/)

- **On Windows**

Install via pip (requires pip 19.0 or later)

``` powershell 
py -m pip install --user pipx
```

move to <USER folder>\AppData\Roaming\Python\Python3x\Scripts

``` powershell
.\pipx.exe ensurepath
```
This will add both the above-mentioned path and the %USERPROFILE%\.local\bin folder to your search path. Restart your terminal session and verify that pipx runs correctly.

### 2.2 Installing Poetry
[Poetry Document](https://python-poetry.org/docs/#installing-with-pipx)
``` powershell
pipx install poetry
```

### 2.3 Config Poetry to create venv at root
run this command to make poetry to create venv at root
``` powershell
poetry config virtualenvs.in-project true
```

### 2.4 Installing dependencies
move to root project
``` powershell
poetry install
```

### 2.5 Access the venv
if you are not using **vscode** *(vscode will auto load the venv for python 3.12+)*
1. At root project

1. CMD
    ``` bash
    .venv\Scripts\activate.bat
    ```
2. Powershell
    ``` powershell
    .venv\Scripts\activate.ps1
    ```
### 2.6 Running API service
[FastAPI Document](https://fastapi.tiangolo.com/)
``` powershell
fastapi dev API_main.py
```
To access API UI documentation
```
localhost:8000/docs
```

### 2.7 Running service (optional since you can run it through the API UI)
``` powershell
py console.py
```
--> this service update data and train AI Model every 6AM


