# deepview

An implementation of Google's Deepview paper https://augmentedperception.github.io/deepview/

# Train

We'll use the Spaces and the Real Estate 10K datasets to train the model. I'll use the notebook format so if the line starts with a ! bang it's bash and otherwise it's Python code.

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

You can use the full Real Estate 10K dataset to train the model, but it's large so I've split it into 39 repos. The first one is the following one and you only have to replace the last number to get the rest:

https://gitlab.com/Findeton/real-estate-10g-1
