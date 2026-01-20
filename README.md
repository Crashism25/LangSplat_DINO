# LangSplat_DINO
Extending original LangSplat by replacing CLIP with DINO.txt

Put your data in Langsplat/data/"SCENE"

The structure of the dataset should look like this:

```
SCENE
├── dslr/
│   ├── colmap/
│   │   ├── cameras.txt
│   │   ├── images.txt
│   │   └── points3D.txt
│   └── resized_images/
│       ├── image_001.jpg
│       └── ...
```

After running the complete_pipeline notebook, the data will look like this:

```
SCENE
├── images/
│   ├── image_001.jpg
│   └── ...
├── language_features_dino/
│   ├── ..._f.npy
│   ├── ..._s.npy
│   └── ...
├── output/
│   ├── point_cloud/
│   ├── cameras.json
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.txt
        ├── images.txt
        └── points3D.txt
```
The Autoencoder training stores checkpoints in autoencoder/ckpt dir and test.py generates 3d features in data/SCENE 


The checkpoints of Langsplat training with language features will be in Langsplat/output


## Key Modifications from Original LangSplat
### 1. **DinoText Vision Encoder (Replacing CLIP)**
### 2. **Optimized Preprocessing for Large-Scale Datasets**
### 3. **Modified autoencoder to handle 2048-dim to 3-dim feature vectors**


