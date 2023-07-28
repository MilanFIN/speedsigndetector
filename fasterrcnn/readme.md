This implementation is/was trained with a swedish traffic sign dataset that can be found from

https://www.cvl.isy.liu.se/en/research/datasets/traffic-signs-dataset/

The model was only trained to detect speedsigns, and as such the annotations were cropped to only include those.

Both set 1 & 2 annotations were combined to a single file to 

`data/annotations.txt`

and images to 

`data/images`

Running the `parseannotations.py` script will parse the relevant annotations from the `annotations.txt` file and place them
into a folder 

`data/annotations/`

After that the `train.py` script can be run to train a model
