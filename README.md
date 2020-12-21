# plan2exploreMarathonEnvs

Experimental Repro testing [plan2explore](https://github.com/ramanans1/plan2explore) with [MarathonEnvs](https://github.com/Unity-Technologies/marathon-envs)



plan2explore orginal/tensorflow copied from commit [59fa82f](https://github.com/ramanans1/plan2explore/commit/59fa82fe4d40b66b2903643ee36befe1c3ca807e)

plan2explore-pytorch from commit [13c13bd](https://github.com/yusukeurakami/plan2explore-pytorch/commit/13c13bd6c206742fd25d68ab693a5b5271b5b34a)

[marathon-envs](https://github.com/Unity-Technologies/marathon-envs) from release [v3.0.0](https://github.com/Unity-Technologies/marathon-envs/releases/tag/v3.0.0)



### Setup

1. Install conda environment:

```
conda env create -f environment.yml

conda activate p2e

cd marathon-envs
pip install .
cd ..

```

2. copy marathon-envs executable from [here](https://github.com/Unity-Technologies/marathon-envs/releases/tag/v3.0.0) to envs\

