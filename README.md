# Closed-loop MouseTracker: Live tracking of mouse behaviour


[![License MIT](https://img.shields.io/github/license/CENTURI-Hackathon-2025/mousetracker?color=green)](https://github.com/CENTURI-Hackathon-2025/mousetracker/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/mousetracker.svg?color=green)](https://pypi.org/project/mousetracker)
[![Python Version](https://img.shields.io/pypi/pyversions/mousetracker.svg?color=green)](https://python.org)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)

Closed-loop MouseTracker: Live tracking of mouse behaviour

----------------------------------

## Installation
 
You can install `mousetracker` via [pip]:

```shell
pip install mousetracker
```


To install latest development version (use this line as there is already a 'mousetracker' on PiPy) :

```shell
pip install git+https://github.com/CENTURI-Hackathon-2025/mousetracker.git
```

## Project

### Experimental context

Our experimental setup allows us film a freely moving mouse in an arena with 4 towers and track its position in live. If the mouse performs a specific action (e.g turning around a tower in the clockwise direction) it will trigger the delivery a droplet of water from the tower as a reward. Our aim with those experiment is to study the decision making process, motor control and learning of the mouse across experiments. This Hackathon project arises from our need to have an efficient tracking in live (as it is important to correctly trigger the reward).

### Project

This project's aim is to improve the tracking code we are using currently. It have been developped in python, relying on the libraries OpenCV and Numpy. It suffers from several problems:
1. The mouse can sometimes be lost by our tracker.
2. The tracking of the mouse is not working properly in the first seconds of the recording.
3. The camera lens distorts the image which biases the shape of mouse trajectory.
4. Some of our old experiment have been done with a white background, which makes the tracker follow the mouse's shadow instead of the mouse.

Additionaly, our hardware is not powerful enough to run a machine-learning based tracking in live. It would introduce too much delay during the live tracking. 

Concretely, the goal is to improve the code (or make a new one) so it would be free of those problems, and optimized to run as fast as possible.

### Tasks

This project can be divided in several tasks: 
- (Problem 1 and 2) Study how the mouse/background detection may impact the loss of the mouse by the tracker and the (speed of the background estimation, weight of the past estimations, threshold etc.)
- (Problem 3) Search for image processing methods to correct for lense distortion. It can be based on the camera specifications (e.g focus, size of the lens etc.) or on visual criterion (e.g develop a transformation such that it makes the borders of the arena parallel).
- (Problem 4) Study the different ways to detect shadows.
- Compare the new and old tracking code
    - Find metrics to compare them.
    - It might be relevant to implement a machine-learning based tracking (DeepLabCut, SLEAP etc.) just to compare it to the tracking that will be developped in the Hackathon.

Those tasks are just guidelines, and they can be rediscussed during the project.

### Available Material

- Experiment videos (~12 min of a mouse moving freely in the arena and performing the task)
- Current tracking algorithm. This script take a video as an input and displays this video with the tracking point on it.
- Current tracking algorithm embeded in the acquisition code. This one cannot be ran. It is here for an informative goal.
- Camera characteristics, as well as the parameters used during recording (focus, exposition etc.)
- Dimensions of the experimental setup.

## Contributing

Contributions are very welcome.

## License

Distributed under the terms of the [MIT] license,
"mousetracker" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[tox]: https://tox.readthedocs.io/en/latest/

[file an issue]: https://github.com/CENTURI-Hackathon-2025/mousetracker/issues

