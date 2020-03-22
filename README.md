# Kitchen Silence Splitter ![](https://raw.githubusercontent.com/dupontinquiries/kss/master/kss_github_logo_v2.png)

The Kitchen Silence Splitter, or KSS, is a python program that trims videos based on their volume.

## Installation - coming soon!

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## Usage

### Preparing the Workspace

Create a directory in the same folder as __kss2.py__ named __footage__, and add your video files to a folder named input.

### Setting Up the Environment

Ensure you have __ffmpeg-python__ and __moviepy__, as well as:
 - opencv-python
 - pydub
 - termcolor

### Running the Program

Run __kss2.py__ and let the program run.

```python
python kss2.py
```

The completed file will show up in the folder named __final__.

## Contributing
Feel free to fork and make something awesome!

## License
[GNU](https://www.gnu.org/licenses/gpl-3.0.en.html)


# kCrawler

kCrawler is a python program that compresses large folders of videos. It uses ffmpeg to encode videos in the [h.265 (HEVC)](http://x265.org/) format.

## Installation - coming soon!

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## Usage

### Preparing the Workspace

Locate the directory that holds your input folder.  A new folder called __program_results__ will be created in this directory once the programs starts running.

### Setting Up the Environment

Ensure you have __ffmpeg__ installed and check that it is __added it to your environment variables__.

### Running the Program

Run __kCrawler.py__ and paste the path to your folder when prompted.

```python
python kCrawler.py
Path to the workspace => your/path/here
```

Videos will start appearing in the __program_results__ folder.

## Contributing
Feel free to fork and make something awesome!

## License
[GNU](https://www.gnu.org/licenses/gpl-3.0.en.html)
