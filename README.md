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

This class connects to the DAQ as an input to the magnetometer. All magnetometer functions are in this class. \
All functions calls to the DAQ are from the nidaqmx library. You can find out more here: \ 
https://nidaqmx-python.readthedocs.io/en/latest/ 
This class relies on multithreading proccessing. We createda producer and consumers.
The producer is a common buffer that has data being stored with a size declared in buffer_size
The consumer is a queue that takes from the producer and logs the data

```python
   Magnetometer(self, device_name, channel_names=None, sample_rate=200, buffer_size=200, log_path='../logs/')
```

#### Class variables 
``` python
self.device_name: Name of device in NI MAX
self.channel_names: Optional, list of channels to use. Default: ["/ai0", "/ai1", "/ai2"]
self.log_path: The file path for logging the magentometer readings
self.device_name: The name for the NIDAQ you created
self.channels: The channels that you're using in the DAQ. It will always be  in the format "/ai#"
self.n_channels: The number of channels you created
self.Runnning: Boolean that checks if the magnetomter class is already running
self.task: An instance of the nidaqmx.task.Task. Does the data aquisition
self.reader: An instance of nidaqmx stream reader. For this task its an input type since its used as input for the magnetometer
self.daq_out_queue: The producer queue used to collect the data
self.log_queue: A consumer that takes the data and logs it to a .csv file
self.consumer: a 2d array of all the consumer queues
self.con_man_thread: The manager thread used for the consumers
self.log_thread: The thread used for logging

```

#### __init__
``` 
Intializes everything needed for data collection. The process for this is
1. Reset the Daq. Stored in device
2. Set up the task by adding the voltage channel, data aquisition type, and reader. Stored in self.task and self.reader
In this case we are doing Conintuous aquisitions and in stream readings
3. Setup producer
4. Setup consumer
```
#### close
``` python
close(self)
Checks if theres a Task opened and if closes if there is one
```

#### start:

```python
start(self, log_filename=None)
1. Sets up the logging and consumer manager threads with the functions __log_consumer and __consumer_manager respectively
2. Starts the consumer and producer tasks
```

#### stop

``` python
stop(self)
Stops magnetometer reading and logging, and sends poison pill to all consumers
```

#### add_consumer

``` python
add_consumer(self, q)
Adds a queue to be populated by the producer
param q: Queue to be populated
```

#### __daq_producer
``` python
 __daq_producer(self, task_idx, event_type, n_samples, callback_data=None)
 1. Intializes a variable buffer with 0's
 2. Reads data from self.reader and puts data into bufferr
 3. Pushes the tuple (time, number of readings from each channel, x,y,z readings) without blocking
```

#### __consumer_manager:
``` python
 __consumer_manager(prod_q, cons_qs):
 1. Checks if there is any data in the producer q
 2. Calls the other consumer threads and gives the data to each consumer
 3. If there is no data left in the producer q it sends a poison pill
```

#### __log_consumer

``` python
__log_consumer(filepath, log_queue)
1. Get data from the log_queue
2. If there is data, push the tuple (time, x,y,z readings) into the dataframe
3. Otherwise use a poison pill

```

### Analysis.py

   #### resample_dataframe
   ``` python
    resample_dataframe(df, sample_rate, agg_func='mean', trim=False)
    Aggregates a timeseries of observations into a timeseries of uniform sample rate and returns the re-sampled dataframe

    param df: Dataframe of observations whose index is the elapsed seconds passed when the observation was recorded.
    param sample_rate: Number of samples to aggregate to in each second. (Hz)
    param agg_func: Function(s) to aggregate the sub-samples. Default is 'mean', can be a list of any agg functions.
    param trim: Defaults to false. Choose to remove the final window from the dataframe if it is not fully saturated with samples.
    
   ```
   
   ###### _sub_sample
    ``` python
    _sub_sample(w_df) 
    Note: Inside the function resample_dataframe
    Create a parameterization of the data and have the interval be in terms of a percentage between (0, 1]
    Forumla: w_df['sample'] = ((w_df.index - w_df.index.min()) * sample_rate).astype(int)
    ```
   #### usgs_dataframe
   ``` python
   usgs_dataframe(usgs_filepath)
   Reads a USGS json data file and returns it as a dataframe with timestamped indices and each row is a recorded channel

   param usgs_filepath: Data file to read
   ```
    
   #### merge_log_usgs
   ``` python
   merge_log_usgs(dt_identifier, sample_rate=240)
   Merges datasets from concurrent magnetometer and USGS readings into a single dataframe with constant units and time intervals
   param dt_identifier: Experiment date string used to identify which files to load.
   param sample_rate: Sample rate to aggregate the magnetometer readings at.
   return: Dataframe indexed by elapsed seconds and sample number. Column names prefixed
          by `USGS - ` are from the USGS dataset. Otherwise, the column is from the magnetometer
          readings dataset. All values in the dataframe are in terms of Tesla.
   ```
   #### plot_log
    ``` python
    plot_log(log_path, save=False):
    Reads data from log file and plots the data

    param log_path: Path to log file
    param save: Whether to save or display plot. Default: False (display)
    ```
   #### fft_signal
    ``` python
    fft_signal(signal, sampling_rate):
    Simple helper function to compute the fft of a signal with a known sampling rate.

    De-means the signal, then computes the frequency space and frequency amplitudes
    in terms of input signal density. I.e. if the signal is in volts, the output of the
    frequency amplitudes will be in units of squared volts.

    :param signal: Signal timeseries to analyze.
    :param sampling_rate: Sampling rate of the signal, in Hz
    :return: Tuple of the frequency space and associated frequency amplitudes
    ```
   #### plot_log
    ``` python
    plot_log_fft(log_path, save=False, max_freq=60):
    Reads data from log file, computes and plots fft

    :param log_path: Path to log file
    :param save: Whether to save or display plot. Default: False (display)
    :param max_freq: Show up to this frequency on plot (Hz).
    ```
### Simulation.py
Field equations for creating a magnetic field given by one or more rectangular loops of wire in the same plane
Adapted from mathmatica file given by Svenja: https://drive.google.com/file/d/1FrGWrs4JCh3DOKBoIA7Rn8GUHFDU2A24/view?usp=sharing
For background info: https://drive.google.com/file/d/1ITCqS9WszL4EksHgJJAhwlwrAfNrGPrn/view?usp=sharing

