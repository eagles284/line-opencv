import aiml
import os

kernel = aiml.Kernel()
# Create the kernel and learn AIML files
if os.path.isfile("brain.brn"):
    kernel.bootstrap(brainFile = "brain.brn")
else:
    kernel.bootstrap(learnFiles = "std-startup.xml", commands = "tolong load chatlist saya")
    kernel.saveBrain("brain.brn")

# Press CTRL-C to break this loop
while True:
    print(kernel.respond(input("Enter your message >> ")))