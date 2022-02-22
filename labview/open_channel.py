import nidaqmx

device_name = 'cDAQ1Mod4'
channel_names = 'ao0'

with nidaqmx.Task() as task:
    task.ao_channels.add_ao_voltage_chan(f'{device_name}/{channel_names}')

    oclce = task.out_stream.overcurrent_chans_exist
    print('Open loop channels exists:', oclce)

    if oclce:
        for s in task.out_stream.overcurrent_chans:
            print('Open channel:', s)

    # print('1 Channel 1 Sample Write: ')
    # print(task.write(1.0))
    # task.stop()
    #
    # print('1 Channel N Samples Write: ')
    # print(task.write([1.1, 2.2, 3.3, 4.4, 5.5], auto_start=True))
    # task.stop()
