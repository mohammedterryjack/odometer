from microbit import accelerometer, display, sleep

with open("data.txt","w") as data:
    for _ in range(100):
        readings = accelerometer.get_values()
        data.write(str(readings)+"\n")
        sleep(100)

display.scroll('Done!', wait=False, loop=True)