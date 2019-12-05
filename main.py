import midi_in
import midi_out
import model
import sys

dic = {}
n_files_to_load = 10
dic,data = midi_in.transform(n_files_to_load,dic)

data = model.creator(data)

midi_out.prep_files(data,dic)
sys.exit()
