# Senior_Capstone

## Installation:
For this project you must have the following software. 
NOTE: This is only for Windows. CU Boulder only offers labview student suite for free which is only for Windows. You can technically run this in Linux if you spend $500 a month for the regular version on Linux.

Labview and Python. You must make sure that the bit value is the same for both. Either both are 64 bit or 32 bit. Also, currently only Python 3.8.10 works without issue.

Labview installation via CU Boulder:  https://oit.colorado.edu/software-hardware/software-catalog/labview

Python 3.8.10: https://www.python.org/downloads/release/python-3810/

Ni DAQmx: https://www.ni.com/en-us/support/downloads/drivers/download.ni-daqmx.html#428058

IDE: Pycharm: https://www.jetbrains.com/pycharm/download/#section=windows

Pip3 for windows: https://pip.pypa.io/en/stable/installation/

Check if you already have pip3: ``` pip3 --version ```

### pipenv for package mangement
We used pipenv for managing our dependicies and makeing sure the correct version of python is running.

To install pipenv: ``` pip install --user pipenv ```

Inisde the git repository 

To enter virtual enviroment: ``` pipenv shell ```

To download dependicies: ``` pipenv install ```


## Troubleshooting:
You may have an issue where you have already installed python and it is taking your newest version instead of python 3.8.10 To fix this search “System variables” in the windows search bar. Then click on environment variables. Change the python path you currently have to the python path where you have python 3.8.10.


