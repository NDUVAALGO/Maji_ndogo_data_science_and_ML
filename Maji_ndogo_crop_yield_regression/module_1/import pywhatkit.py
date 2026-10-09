import time

# REDUCED: Changed from 67999 seconds (~19 hours) to 5 seconds
time.sleep(5) 

print("Opening WhatsApp...")

import pywhatkit

# Sends the message immediately (with an 11-second page-load delay)
pywhatkit.sendwhatmsg_instantly("+254117177691", "Hello, this is a test message!", 11)
