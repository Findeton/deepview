# Deepview: View Synthesis with Learned Gradient Descent

An amateur implementation of Google's Deepview paper on View Synthesis with Multiplane Images (MPI) https://augmentedperception.github.io/deepview/

# Results

An MPI generated with this code with a trained model, at the moment, looks like this:

![MPI](results.png)

# Train

You can also run the following code in [this Colab notebook](https://colab.research.google.com/drive/1tr4N0shbCH2sLfgStZtZcVNPskY5mZOg?usp=sharing). I've also included [a copy of the notebook](colab_notebook.ipynb) in the repo and I've published [a blogpost](https://roblesnotes.com/blog/lightfields-deepview/) as well.

We'll use the [Spaces](https://github.com/augmentedperception/spaces_dataset) and the [Real Estate 10K](https://google.github.io/realestate10k/) datasets to train the model. I'll use the notebook format here so if the line starts with a ! bang it's bash and otherwise it's Python code.

Let's start with the spaces dataset:

    ! git clone https://github.com/Findeton/deepview.git
    ! cd deepview && git checkout . && git checkout main && git pull
    ! git clone https://github.com/augmentedperception/spaces_dataset.git
    ! cd deepview && pip3 install -r requirements.txt
    ! mkdir -p deepview/trained-models
    for i in range(10):
       !  cd deepview && DSET_NAME=spaces:1deterministic SPACES_PATH=/content/spaces_dataset/ python3 train.py

Now let's create an MPI:

    ! cd deepview && DSET_NAME=spaces:1deterministic SCENE_INDEX=1 SPACES_PATH=/content/spaces_dataset python3 tiled_render_spaces.py
    import IPython
    IPython.display.HTML(filename='/content/deepview/generated-html/deepview-mpi-viewer.html')

Let's use a reduced set of the Real State 10K dataset to further train it:

    ! git clone https://gitlab.com/Findeton/real-estate-10k-run0.git
    for i in range(10):
    ! cd deepview && DSET_NAME=re:1random RE_PATH=/content/real-estate-10k-run0/ python3 train.py

Let's show an MPI from that dataset:

    ! cd deepview && DSET_NAME=re:1random TILE_W=200 TILE_H=200 SCENE_INDEX=1 RE_PATH=/content/real-estate-10k-run0 python3 tiled_render_spaces.py
    import IPython
    IPython.display.HTML(filename='/content/deepview/generated-html/deepview-mpi-viewer.html')

You can use the full Real Estate 10K dataset to train the model, but it's large and requires pre-processing so I've split it into 39 repos! The first one is the following and you only have to replace the last number to get the rest:

https://gitlab.com/Findeton/real-estate-10g-1
