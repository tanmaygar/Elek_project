import RPi.GPIO as GPIO
import time

PIR_PIN = 4
LED_PIN= 12

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN,GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Setting mode to down because of pull down resistor

 #the time we give the sensor to calibrate (10-60 secs according to the datasheet) 
calibrationTime = 30   #the time when the sensor outputs a low impulse 

 #before we assume all motion has stopped 
pause = 5 
lockLow = True  
 # # # # # # # # # # # # # #/  #SETUP 
print("...........................\n") 
GPIO.setup(LED_PIN, GPIO.OUT)  
GPIO.output( PIR_PIN, GPIO.LOW)   #give the sensor some time to calibrate 
print("calibrating sensor ")  
for i in range(calibrationTime):
    print(".")  
    time.sleep(1)  
print(" done")  
print("SENSOR ACTIVE")  
time.sleep(0.05)
 # # # # # # # # # # # # # #  #LOOP 
while True: 
  if(GPIO.input( PIR_PIN) == GPIO.HIGH):
    GPIO.output(LED_PIN, GPIO.HIGH)   #the led visualizes the sensors output pin state 
    if(lockLow):
    #makes sure we wait for a transition to LOW before any further output is made: 
      lockLow = False  
      print("---")  
      print("Sensing Initiated ")  
      # Do mask and temperature sensing
      time.sleep(0.05)
    takeLowTime = True  
  
  if(GPIO.input( PIR_PIN) == GPIO.LOW): 
    GPIO.output(LED_PIN, GPIO.LOW)   #the led visualizes the sensors output pin state 
    if(takeLowTime): 
      lowIn #the amount of milliseconds the sensor has to be low 
      lowIn = time.time()  #save the time of the transition from high to LOW 
      takeLowTime = False   #make sure this is only done at the start of a LOW phase 
     #if the sensor is low for more than the given pause, 
     #we assume that no more motion is going to happen 
    if((not lockLow) and time.time() - lowIn > pause): 
       #makes sure this block of code is only executed again after 
       #a new motion sequence has been detected 
      lockLow = True  
      print("Sensing Done ")   #output  
      time.sleep(0.05)  

GPIO.cleanup()