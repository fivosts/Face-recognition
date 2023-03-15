## INSTALL

Create a virtual environment and install all necessary python modules:

```
mkdir env
virtualenv -p python3.8 env
source env/bin/activate

python -m pip install -r requirements.txt

chmod +x engine.sh benchmark.sh server.sh webcam.sh
```

You can skip using `virtualenv` and you can directly use the local python installation. Python 3.7 or 3.8 is compatible for the package versions.

## RUN

There are 4 scripts:

```
benchmark.sh
webcam.sh
engine.sh
server.sh
```

### Tasks 1,2 and 3

`benchmark.sh` implements Task 1, 2 and 3. Specifically:

For **Task 1** see core function `run_engine` in `cnn_stream/AI/engine.py`.
For **Task 2** see `cnn_stream/AI/quantizer.py`. A range of static and dynamic quantizers have been implemented, but only the simplest actually works that converts model weights to fp16.
For **Task 3**, function `benchmark_engine` in `cnn_stream/AI/bootloader.py` benchmarks the CNN for a range of different batch sizes, while toggles on and off the fp16 conversion of the model. All different metrics of all configurations will be plotted in a bar plot under `plots/`. Plots are interactive HTML-based files. Open with browser.

### Tasks 4 and 5

The other 3 scripts bootstrap the three different processes for **Tasks 4** and **5**.

You can run `./webcam.sh`, `./server.sh` and `./engine.sh` in 3 different terminals.

`webcam.sh` opens the webcam, streams pictures and publishes them using MQTT.

`engine.sh` uses a MQTT subscriber and the online streamer picks up webcam frames, sends them to the CNN for processing and publishes with HTTP requests to a standalone server.

`server.sh` creates a flask-based http application with multiple functionalities. First of all, all received frames by the engine are stored chronologically in a SQL database, which will be found in `database/`. There are three main functionalities:

- See most recent frame: Shows the most recent picture picked up by the server.
- See live video: Shows live footage of processed images streamed by the engine.
- See history: Shows a video (sequence of frames) of all processed images picked up over time.

You can initialize the processes in any order you like. However, if the `engine` is running with a dead `server` you will get error messages that the `POST` requests fail. In such case, the posting daemon will sleep for a few seconds and retry. To avoid these messages initialize the `server` before the `engine`.
