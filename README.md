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

To install pipenv: ``` pip3 install --user pipenv ```

Inisde the git repository 

To download dependicies: ``` pipenv install ```

To enter virtual enviroment: ``` pipenv shell ```


### Required Hardware

Magnetometer: Mag 612 FluxGate Magnetometer \
For more info about the magnetometer please look here: https://drive.google.com/file/d/1OGLSczA6ZfqLArzqDdKDe62jSup29FKf/view?usp=sharing

NI Chassis

NI 9263: Output drives the coil to the PCB

NI 9239: Input for the magnetometer

PCB Board: Connect soldered wires to 15 pin connectors, and 15 pin connector to NI 9263 

## Troubleshooting:
You may have an issue where you have already installed python and it is taking your newest version instead of python 3.8.10 To fix this search “System variables” in the windows search bar. Then click on environment variables. Change the python path you currently have to the python path where you have python 3.8.10.

You may encounter a problem where you install pipenv and everything went well but Windows says it can't find the command. The virtualenv package and pipenv could be on differnt versions. Look at this: https://stackoverflow.com/questions/46041719/windows-reports-error-when-trying-to-install-package-using-pipenv


## Code Documentation

To get magnetometer readings run main.py

### Main.py

This file creates an instance of the magnetometer class. Which then kicks off the data colleciton

### Magnetometer.py

```python
   Magnetometer(self, device_name, channel_names=None, sample_rate=200, buffer_size=200, log_path='../logs/')
```

#### Class variables 
``` python
"""
self.device_name: Name of device in NI MAX
self.channel_names: Optional, list of channels to use. Default: ["/ai0", "/ai1", "/ai2"]
sample_rate: Optional, sample rate in hz. Default: 200
"""
```



