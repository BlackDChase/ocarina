import midi_in
import notes_out
import model
import sys
import player
import numpy as np
dic = {}

### trial
data,dic = midi_in.transform_test(dic)
#print(np.shape(data))
data = notes_out.prep_files(data,dic)
#print(np.shape(data))
player.make(data)
#print(np.shape(data))
sys.exit()      





### Actual
n_files_to_load = 0

# to load all the files
# n_files_to_load = 0     

print("Reading midi files")
data,dic = midi_in.transform(n_files_to_load,dic)

print("Running model")
data = model.creator(data,len(dic))
# might have error for n_files_to_load = 0 , coz of more number of n_keys (not sure why)

print("Conveting the files to notes")
notes = notes_out.prep_files(data,dic)
# Make music function incomplete

print("Converting notes midi and playing")
# Player not made
player.make(notes)

