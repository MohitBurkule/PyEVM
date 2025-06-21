# Python implementation of EVM(Eulerian Video Magnification)

This is a python implementation of eulerian video magnification《[Eulerian Video Magnification for Revealing Subtle Changes in the World](http://people.csail.mit.edu/mrub/evm/)》.
>Our goal is to reveal temporal variations in videos that are difficult or impossible to see with the naked eye and display them in an indicative manner. Our method, which we call Eulerian Video Magnification, takes a standard video sequence as input, and applies spatial decomposition, followed by temporal filtering to the frames. The resulting signal is then amplified to reveal hidden information.Using our method, we are able to visualize the flow of blood as it fills the face and also to amplify and reveal small motions. Our technique can run in real time to show phenomena occurring at temporal frequencies selected by the user.

## Docker Usage (Recommended)

The easiest way to run this application is using Docker, which handles all dependencies automatically.

### Quick Start with Docker

1. **Pull the image from Docker Hub:**
```bash
docker pull mohitburkule/evm-magnification:latest
```

2. **Run motion magnification (default):**
```bash
docker run --rm -v /path/to/your/videos:/app/input -v /path/to/output:/app/output evm-magnification /app/input/your_video.mp4
```

3. **Run color magnification:**
```bash
docker run --rm -v /path/to/your/videos:/app/input -v /path/to/output:/app/output evm-magnification /app/input/your_video.mp4 --mode color
```

### Docker Build Instructions

If you want to build the Docker image yourself:

1. **Clone the repository:**
```bash
git clone <repository-url>
cd PyEVM
```

2. **Build the Docker image:**
```bash
docker build -t evm-magnification .
```

3. **Run the container:**
```bash
docker run --rm -v /path/to/your/videos:/app/input -v /path/to/output:/app/output evm-magnification /app/input/your_video.mp4
```

### Docker Usage Examples

**Basic motion magnification:**
```bash
docker run --rm -v $(pwd):/app/input -v $(pwd):/app/output evm-magnification /app/input/baby.mp4
```

**Color magnification with custom parameters:**
```bash
docker run --rm -v $(pwd):/app/input -v $(pwd):/app/output evm-magnification /app/input/baby.mp4 --mode color --low 0.4 --high 3.0 --amplification 30
```

**Motion magnification with custom output filename:**
```bash
docker run --rm -v $(pwd):/app/input -v $(pwd):/app/output evm-magnification /app/input/baby.mp4 --output /app/output/baby_motion_magnified.avi
```

**All available options:**
```bash
docker run --rm -v $(pwd):/app/input -v $(pwd):/app/output evm-magnification /app/input/baby.mp4 --mode motion --low 0.4 --high 3.0 --levels 4 --amplification 25 --output /app/output/result.avi
```

### Command Line Arguments

- `video_path` (required): Path to the input MP4 video file
- `--mode`: Choose between 'color' or 'motion' magnification (default: motion)
- `--low`: Low frequency bound (default: 0.4)
- `--high`: High frequency bound (default: 3.0)
- `--levels`: Number of pyramid levels (default: 3)
- `--amplification`: Amplification factor (default: 20)
- `--output`: Output video path (default: /app/output/out.avi)

### File Structure for Docker

```
your-project/
├── input/          # Place your input videos here
├── output/         # Processed videos will appear here
└── ...
```

## Manual Installation (Alternative)

If you prefer to run without Docker:

### Install OpenCV3
Since the OpenCV3.X does not support Python3, you need to install opencv3 manually.

Firstly,download opencv3 for python3:
>OpenCV3 for Python3: http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv

Then install opencv3 with pip:
```
pip install opencv_python-3.1.0-cp35-cp35m-win_amd64.whl
```

### Other Libraries
* SciPy for signal processing
* NumPy for image processing
* Pympler for memory profiling

### Manual Usage
```bash
python EVM.py your_video.mp4 --mode motion --amplification 20
```

## Results

Original video：
![原图](http://img.blog.csdn.net/20160927155312178)

Color magnification：
![色彩放大](http://img.blog.csdn.net/20160927155358125)
The color of chest changes.

Motion magnification：
![运动放大](http://img.blog.csdn.net/20160927155455071)
You can see the motion of chest has been magnified.

## Requirements

- Docker (recommended)
- Or Python 3.5+ with required packages (see requirements.txt)

## Chinese version
You can read my blog for more information
>http://blog.csdn.net/tinyzhao/article/details/52681250
```