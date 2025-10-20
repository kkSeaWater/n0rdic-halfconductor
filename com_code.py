import time, serial.tools.list_ports as lp
def snap(): 
    return {p.device:(p.description, p.vid, p.pid, p.manufacturer, p.hwid) for p in lp.comports()}
print("Unplug/plug your PPK2 once. Watching ports…")
before = snap()
while True:
    time.sleep(0.5)
    now = snap()
    add = {k:v for k,v in now.items() if k not in before}
    rem = {k:v for k,v in before.items() if k not in now}
    if add or rem:
        print("Removed:", rem)
        print("Added  :", add)
        before = now