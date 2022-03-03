from synthesizer.inference import Synthesizer
from pathlib import Path
from IPython.utils import io
from argparse import ArgumentParser
import base64
import pickle
import tensorflow as tf


syn_dir = Path("./synthesizer/saved_models/logs-pretrained/taco_pretrained")
synthesizer = Synthesizer(syn_dir)

def get_mels(text_fragments, embedding):

	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)

	input_text = pickle.loads(base64.b64decode(text_fragments))
	input_embd = pickle.loads(base64.b64decode(embedding))

	with io.capture_output() as captured:
		specs = synthesizer.synthesize_spectrograms(input_text, [input_embd ]*len(input_text))

	with open("generated_mels.txt", "wb") as output:
		pickle.dump((specs), output)




def main():
	parser = ArgumentParser()

	parser.add_argument('-t', '--text_fragments', dest='text_fragments')
	parser.add_argument('-e', '--embeding', dest='embeding')
	#parser.add_argument('-s', '--specs', dest='output_data')

	args = parser.parse_args() 

	print(get_mels(args.text_fragments, args.embeding))

if __name__ == '__main__':
    main()
