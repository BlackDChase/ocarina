import midi_in
import midi_out
import model
import sys

dic = {}
n_files_to_load = 10 

# to load all the files
# n_files_to_load = 0     

print("Reading midi files")
dic,data = midi_in.transform(n_files_to_load,dic)

print("Running model")
data = model.creator(data,len(dic))
# Training part incomplete, random99 has to be edited

print("Conveting the files back to midi")
music = midi_out.prep_files(data,dic)
# Make music function incomplete

print("Playing Music")
# Player not made
