--- Requirements ---

tqdm, numpy, pillow, owlready2, pandas, sympy, huggingface-hub

Running the script without the dependencies will result in a prompt giving the option to installing them.

If installing manually, make sure to install on blender's python environment, refer to http://www.codeplastic.com/2019/03/12/how-to-install-python-modules-in-blender/ for instructions.


-- Usage -- 

1. Install blender
2. Run blender -b --python VCAB_generate.py -- [args]

args:

(required) -n X 	-> desired number of images 
--keep 		 	-> Number of images to keep in jpg format, default is 'all'

--textures		-> Vary textures, off by default
--cam			-> Vary camera, choose 'small' or 'large', default is 'none'
--background/-bg 	-> Place background objects, off by default
--contrast/-c		-> Vary image contrast, off by default
--gamma/-g		-> Vary image gamma, off by default

--ontology_path	-> Path to ontology refferenced for image creation, default is 'ontology.owl'. When customizing ontology, it is advisable to use the default one as starting point. 
--output_dir		-> Directory to save dataset to, './results' by default.
--resolution		-> 2 values containing pixel height and width of the images, default is 224*224

--verbose/-v		-> Use if you wish to see blender's render logs

