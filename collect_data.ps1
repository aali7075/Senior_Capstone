$ts = Get-Date -Format "yyyy-MM-ddTHH-mm-ss"
$fname = "C:\\Users\\Lab\\Desktop\\Senior_Capstone\\logs\\python\\python_" + $ts + ".log"

C:\Users\Lab\.virtualenvs\Senior_Capstone-ZalnXfov\Scripts\python.exe C:\Users\Lab\Desktop\Senior_Capstone\src\main.py > $fname
